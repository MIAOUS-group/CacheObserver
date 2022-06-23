#![deny(unsafe_op_in_unsafe_fn)]

use cache_utils::calibration::{
    accumulate, calibrate_fixed_freq_2_thread, calibration_result_to_ASVP, flush_and_reload,
    get_cache_attack_slicing, load_and_flush, map_values, only_flush, only_reload, reduce,
    reload_and_flush, CalibrateOperation2T, CalibrateResult2T, CalibrationOptions, ErrorPrediction,
    ErrorPredictions, HistParams, HistogramCumSum, PotentialThresholds, ThresholdError, Verbosity,
    ASP, ASVP, AV, CFLUSH_BUCKET_NUMBER, CFLUSH_BUCKET_SIZE, CFLUSH_NUM_ITER, SP, SVP,
};
use cache_utils::mmap::MMappedMemory;
use cache_utils::{flush, maccess, noop};
use nix::sched::{sched_getaffinity, CpuSet};
use nix::unistd::Pid;

use core::arch::x86_64 as arch_x86;

use cache_utils::ip_tool::Function;
use core::cmp::min;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::process::Command;
use std::str::from_utf8;

/*
   We need to look at
   - clflush, measure reload (RAM)
   - clflush followed by prefetch L3, measure reload (pL3)
   - clflush followed by prefetch L2, measure reload (pL2)
   - clflush followed by prefetch L2, measure reload (pL1)
   - Load L1, measure reload (L1)
   - Load L1, evict from L1, measure reload (eL2)
   - Load L1, evict L1 + L2, measure reload? (eL3)
   - measure nop (nop)

   Important things to look at : detailed histograms of the diagonals
   Medians for all the core combinations. (Overlapped + separate)

   Checks that can be done :
   - Validate p vs e method to get hit from a specific cache level ?
   - Identify the timing range for the various level of the cache

   Plot system design : core & slice identification must be manual
   Generate detailed graphs with the Virtual Slices, and cores, in SCV format
   Use a python script to perform the slice and core permutations.

   [ ] Refactor IP_Tool from prefetcher reverse into cache util, generate the requisite templates
   [ ] Generate to various preparation functions.
   [ ] Add the required CalibrateOperation2T
   [ ] Result exploitation ?
   - [ ] Determine the CSV format suitable for the plots for histograms
   - [ ] Determine the CSV format suitable for the median plots
   - [ ] Output the histogram CSV
   - [ ] Output the Median CSV
   - [ ] Make the plots
*/

unsafe fn function_call(f: &Function, addr: *const u8) -> u64 {
    unsafe { (f.fun)(addr) }
}

unsafe fn prepare_RAM(p: *const u8) {
    unsafe { flush(p) };
}

unsafe fn prepare_pL3(p: *const u8) {
    unsafe { maccess(p) };
    unsafe { arch_x86::_mm_mfence() };
    unsafe { flush(p) };
    unsafe { arch_x86::_mm_mfence() };
    unsafe { arch_x86::_mm_prefetch::<{ arch_x86::_MM_HINT_T2 }>(p as *const i8) };
    unsafe { arch_x86::__cpuid_count(0, 0) };
}

unsafe fn prepare_pL2(p: *const u8) {
    unsafe { maccess(p) };
    unsafe { arch_x86::_mm_mfence() };
    unsafe { flush(p) };
    unsafe { arch_x86::_mm_mfence() };
    unsafe { arch_x86::_mm_prefetch::<{ arch_x86::_MM_HINT_T1 }>(p as *const i8) };
    unsafe { arch_x86::__cpuid_count(0, 0) };
}

unsafe fn prepare_pL1(p: *const u8) {
    unsafe { maccess(p) };
    unsafe { arch_x86::_mm_mfence() };
    unsafe { flush(p) };
    unsafe { arch_x86::_mm_mfence() };
    unsafe { arch_x86::_mm_prefetch::<{ arch_x86::_MM_HINT_T0 }>(p as *const i8) };
    unsafe { arch_x86::__cpuid_count(0, 0) };
}

unsafe fn prepare_L1(p: *const u8) {
    unsafe { only_reload(p) };
}

unsafe fn prepare_eL2(p: *const u8) {
    unimplemented!()
}

unsafe fn prepare_eL3(p: *const u8) {
    unimplemented!()
}

unsafe fn multiple_access(p: *const u8) {
    unsafe {
        maccess::<u8>(p);
        maccess::<u8>(p);
        arch_x86::_mm_mfence();
        maccess::<u8>(p);
        arch_x86::_mm_mfence();
        maccess::<u8>(p);
        arch_x86::_mm_mfence();
        maccess::<u8>(p);
        maccess::<u8>(p);
    }
}

const SIZE: usize = 2 << 20;
const MAX_SEQUENCE: usize = 2048 * 64;

#[derive(Clone, Copy, Hash, Eq, PartialEq, Debug)]
struct ASV {
    pub attacker: u8,
    pub slice: u8,
    pub victim: u8,
}

struct ResultAnalysis {
    // indexed by bucket size
    pub miss: Vec<u32>,
    pub miss_cum_sum: Vec<u32>,
    pub miss_total: u32,
    pub hit: Vec<u32>,
    pub hit_cum_sum: Vec<u32>,
    pub hit_total: u32,
    pub error_miss_less_than_hit: Vec<u32>,
    pub error_hit_less_than_miss: Vec<u32>,
    pub min_error_hlm: u32,
    pub min_error_mlh: u32,
}

// Split the threshold and error in two separate structs ?

#[derive(Debug, Clone, Copy)]
struct Threshold {
    pub error_rate: f32,
    pub threshold: usize,
    // extend with other possible algorithm ?
    pub is_hlm: bool,
    pub num_true_hit: u32,
    pub num_false_hit: u32,
    pub num_true_miss: u32,
    pub num_false_miss: u32,
}

fn main() {
    let measure_reload =
        cache_utils::ip_tool::Function::try_new(1, 0, cache_utils::ip_tool::TIMED_MACCESS).unwrap();
    let measure_nop =
        cache_utils::ip_tool::Function::try_new(1, 0, cache_utils::ip_tool::TIMED_NOP).unwrap();
    // Grab a slice of memory

    let core_per_socket_out = Command::new("sh")
        .arg("-c")
        .arg("lscpu | grep socket | cut -b 22-")
        .output()
        .expect("Failed to detect cpu count");
    //println!("{:#?}", core_per_socket_str);

    let core_per_socket_str = from_utf8(&core_per_socket_out.stdout).unwrap();

    //println!("Number of cores per socket: {}", cps_str);

    let core_per_socket: u8 = core_per_socket_str[0..(core_per_socket_str.len() - 1)]
        .parse()
        .unwrap_or(0);

    println!("Number of cores per socket: {}", core_per_socket);

    let m = MMappedMemory::new(SIZE, true, false, |i: usize| i as u8);
    let array = m.slice();

    let cache_line_size = 64;

    // Generate core iterator
    let mut core_pairs: Vec<(usize, usize)> = Vec::new();
    let old = sched_getaffinity(Pid::from_raw(0)).unwrap();

    for i in 0..CpuSet::count() {
        for j in 0..CpuSet::count() {
            if old.is_set(i).unwrap() && old.is_set(j).unwrap() {
                core_pairs.push((i, j));
                println!("{},{}", i, j);
            }
        }
    }

    // operations
    // Call calibrate 2T \o/

    let verbose_level = Verbosity::RawResult;

    let pointer = (&array[0]) as *const u8;
    if pointer as usize & (cache_line_size - 1) != 0 {
        panic!("not aligned nicely");
    }

    let operations = [
        CalibrateOperation2T {
            prepare: prepare_RAM,
            op: function_call,
            name: "RAM_load",
            display_name: "Load from RAM",
            t: &measure_reload,
        },
        CalibrateOperation2T {
            prepare: prepare_pL3,
            op: function_call,
            name: "pL3_load",
            display_name: "Load from L3 (prefetch)",
            t: &measure_reload,
        },
        CalibrateOperation2T {
            prepare: prepare_pL2,
            op: function_call,
            name: "pL2_load",
            display_name: "Load from L2 (prefetch)",
            t: &measure_reload,
        },
        CalibrateOperation2T {
            prepare: prepare_pL1,
            op: function_call,
            name: "pL1_load",
            display_name: "Load from L1 (prefetch)",
            t: &measure_reload,
        },
        CalibrateOperation2T {
            prepare: prepare_L1,
            op: function_call,
            name: "L1_load",
            display_name: "Load from L1 (Reload)",
            t: &measure_reload,
        },
        CalibrateOperation2T {
            prepare: noop::<u8>,
            op: function_call,
            name: "pL3_load",
            display_name: "Load from L3 (prefetch)",
            t: &measure_nop,
        },
    ];

    let r = unsafe {
        calibrate_fixed_freq_2_thread(
            pointer,
            64,                                      // FIXME : MAGIC
            min(array.len(), MAX_SEQUENCE) as isize, // MAGIC
            &mut core_pairs.into_iter(),
            &operations,
            CalibrationOptions {
                hist_params: HistParams {
                    bucket_number: CFLUSH_BUCKET_NUMBER,
                    bucket_size: CFLUSH_BUCKET_SIZE,
                    iterations: CFLUSH_NUM_ITER,
                },
                verbosity: verbose_level,
                optimised_addresses: true,
            },
            core_per_socket,
        )
    };

    //let mut analysis = HashMap::<ASV, ResultAnalysis>::new();

    let miss_name = "clflush_miss_n";
    let hit_name = "clflush_remote_hit";

    let miss_index = operations
        .iter()
        .position(|op| op.name == miss_name)
        .unwrap();
    let hit_index = operations
        .iter()
        .position(|op| op.name == hit_name)
        .unwrap();

    let slicing = get_cache_attack_slicing(core_per_socket, cache_line_size);

    let h = if let Some(s) = slicing {
        |addr: usize| -> usize { slicing.unwrap().hash(addr) }
    } else {
        panic!("No slicing function known");
    };

    /* Analysis Flow
        Vec<CalibrationResult2T> (or Vec<CalibrationResult>) -> Corresponding ASVP + Analysis (use the type from two_thread_cal, or similar)
        ASVP,Analysis -> ASVP,Thresholds,Error
        ASVP,Analysis -> ASP,Analysis (mobile victim) -> ASP, Threshold, Error -> ASVP detailed Threshold,Error in ASP model
        ASVP,Analysis -> SP, Analysis (mobile A and V) -> SP, Threshold, Error -> ASVP detailed Threshold,Error in SP model
        ASVP,Analysis -> AV, Analysis (legacy attack)  -> AV, Threshold, Error -> ASVP detailed Threshold,Error in AV model
        ASVP,Analysis -> Global Analysis            -> Global Threshold, Error -> ASVP detailed Threshold,Error in Global Model
        The last step is done as a apply operation on original ASVP Analysis using the new Thresholds.

        This model correspond to an attacker that can chose its core and its victim core, and has slice knowledge
        ASVP,Thresholds,Error -> Best AV selection for average error. HashMap<AV,(ErrorPrediction,HashMap<ASVP,Threshold,Error>)>

        This model corresponds to an attacker that can chose its own core, measure victim location, and has slice knowledge.
        ASVP,Thresholds,Error -> Best A  selection for average error. HashMap<AV,(ErrorPrediction,HashMap<ASVP,Threshold,Error>)>

        Also compute best AV pair for AV model

        What about chosing A but no knowing V at all, from ASP detiled analysis ?




        Compute for each model averages, worst and best cases ?

    */

    let new_analysis: Result<HashMap<ASVP, ErrorPredictions>, nix::Error> =
        calibration_result_to_ASVP(
            r,
            |cal_1t_res| {
                ErrorPredictions::predict_errors(HistogramCumSum::from_calibrate(
                    cal_1t_res, hit_index, miss_index,
                ))
            },
            &h,
        );

    // Analysis aka HashMap<subset of ASVP, ErrorPredictions> --------------------------------------

    let asvp_analysis = match new_analysis {
        Ok(a) => a,
        Err(e) => panic!("Error: {}", e),
    };

    /*    asvp_analysis[&ASVP {
        attacker: 0,
        slice: 0,
        victim: 0,
        page: pointer as usize,
    }]
        .debug();*/

    let asp_analysis = accumulate(
        asvp_analysis.clone(),
        |asvp: ASVP| ASP {
            attacker: asvp.attacker,
            slice: asvp.slice,
            page: asvp.page,
        },
        || ErrorPredictions::empty(CFLUSH_BUCKET_NUMBER),
        |accumulator: &mut ErrorPredictions, error_preds: ErrorPredictions, _key, _rkey| {
            *accumulator += error_preds;
        },
    );

    let sp_analysis = accumulate(
        asp_analysis.clone(),
        |asp: ASP| SP {
            slice: asp.slice,
            page: asp.page,
        },
        || ErrorPredictions::empty(CFLUSH_BUCKET_NUMBER),
        |accumulator: &mut ErrorPredictions, error_preds: ErrorPredictions, _key, _rkey| {
            *accumulator += error_preds;
        },
    );

    // This one is the what would happen if you ignored slices
    let av_analysis = accumulate(
        asvp_analysis.clone(),
        |asvp: ASVP| AV {
            attacker: asvp.attacker,
            victim: asvp.victim,
        },
        || ErrorPredictions::empty(CFLUSH_BUCKET_NUMBER),
        |accumulator: &mut ErrorPredictions, error_preds: ErrorPredictions, _key, _rkey| {
            *accumulator += error_preds;
        },
    );

    let global_analysis = accumulate(
        av_analysis.clone(),
        |_av: AV| (),
        || ErrorPredictions::empty(CFLUSH_BUCKET_NUMBER),
        |accumulator: &mut ErrorPredictions, error_preds: ErrorPredictions, _key, _rkey| {
            *accumulator += error_preds;
        },
    )
    .remove(&())
    .unwrap();

    // Thresholds aka HashMap<subset of ASVP,ThresholdError> ---------------------------------------

    let asvp_threshold_errors: HashMap<ASVP, ThresholdError> = map_values(
        asvp_analysis.clone(),
        |error_predictions: ErrorPredictions, _| {
            PotentialThresholds::minimizing_total_error(error_predictions)
                .median()
                .unwrap()
        },
    );

    let asp_threshold_errors =
        map_values(asp_analysis, |error_predictions: ErrorPredictions, _| {
            PotentialThresholds::minimizing_total_error(error_predictions)
                .median()
                .unwrap()
        });

    let sp_threshold_errors = map_values(sp_analysis, |error_predictions: ErrorPredictions, _| {
        PotentialThresholds::minimizing_total_error(error_predictions)
            .median()
            .unwrap()
    });

    let av_threshold_errors = map_values(av_analysis, |error_predictions: ErrorPredictions, _| {
        PotentialThresholds::minimizing_total_error(error_predictions)
            .median()
            .unwrap()
    });

    let gt_threshold_error = PotentialThresholds::minimizing_total_error(global_analysis)
        .median()
        .unwrap();

    // ASVP detailed Threshold,Error in strict subset of ASVP model --------------------------------
    // HashMap<ASVP, (Thershold ?)Error>,
    // with the same threshold for all the ASVP sharing the same value of an ASVP subset.

    let asp_detailed_errors: HashMap<ASVP, ThresholdError> = map_values(
        asvp_analysis.clone(),
        |error_pred: ErrorPredictions, asvp: &ASVP| {
            let asp = ASP {
                attacker: asvp.attacker,
                slice: asvp.slice,
                page: asvp.page,
            };
            let threshold = asp_threshold_errors[&asp].threshold;
            let error = error_pred.histogram.error_for_threshold(threshold);
            ThresholdError { threshold, error }
        },
    );

    let sp_detailed_errors: HashMap<ASVP, ThresholdError> = map_values(
        asvp_analysis.clone(),
        |error_pred: ErrorPredictions, asvp: &ASVP| {
            let sp = SP {
                slice: asvp.slice,
                page: asvp.page,
            };
            let threshold = sp_threshold_errors[&sp].threshold;
            let error = error_pred.histogram.error_for_threshold(threshold);
            ThresholdError { threshold, error }
        },
    );

    let av_detailed_errors: HashMap<ASVP, ThresholdError> = map_values(
        asvp_analysis.clone(),
        |error_pred: ErrorPredictions, asvp: &ASVP| {
            let av = AV {
                attacker: asvp.attacker,
                victim: asvp.victim,
            };
            let threshold = av_threshold_errors[&av].threshold;
            let error = error_pred.histogram.error_for_threshold(threshold);
            ThresholdError { threshold, error }
        },
    );

    let gt_detailed_errors: HashMap<ASVP, ThresholdError> =
        map_values(asvp_analysis.clone(), |error_pred: ErrorPredictions, _| {
            let threshold = gt_threshold_error.threshold;
            let error = error_pred.histogram.error_for_threshold(threshold);
            ThresholdError { threshold, error }
        });

    // Best core selections

    let asvp_best_av_errors: HashMap<AV, (ErrorPrediction, HashMap<SP, ThresholdError>)> =
        accumulate(
            asvp_threshold_errors.clone(),
            |asvp: ASVP| AV {
                attacker: asvp.attacker,
                victim: asvp.victim,
            },
            || (ErrorPrediction::default(), HashMap::new()),
            |acc: &mut (ErrorPrediction, HashMap<SP, ThresholdError>),
             threshold_error,
             asvp: ASVP,
             av| {
                assert_eq!(av.attacker, asvp.attacker);
                assert_eq!(av.victim, asvp.victim);
                let sp = SP {
                    slice: asvp.slice,
                    page: asvp.page,
                };
                acc.0 += threshold_error.error;
                acc.1.insert(sp, threshold_error);
            },
        );

    let asvp_best_a_errors: HashMap<usize, (ErrorPrediction, HashMap<SVP, ThresholdError>)> =
        accumulate(
            asvp_threshold_errors.clone(),
            |asvp: ASVP| asvp.attacker,
            || (ErrorPrediction::default(), HashMap::new()),
            |acc: &mut (ErrorPrediction, HashMap<SVP, ThresholdError>),
             threshold_error,
             asvp: ASVP,
             attacker| {
                assert_eq!(attacker, asvp.attacker);
                let svp = SVP {
                    slice: asvp.slice,
                    page: asvp.page,
                    victim: asvp.victim,
                };
                acc.0 += threshold_error.error;
                acc.1.insert(svp, threshold_error);
            },
        );

    let asp_best_a_errors: HashMap<usize, (ErrorPrediction, HashMap<SVP, ThresholdError>)> =
        accumulate(
            asp_detailed_errors.clone(),
            |asvp: ASVP| asvp.attacker,
            || (ErrorPrediction::default(), HashMap::new()),
            |acc: &mut (ErrorPrediction, HashMap<SVP, ThresholdError>),
             threshold_error,
             asvp: ASVP,
             attacker| {
                assert_eq!(attacker, asvp.attacker);
                let svp = SVP {
                    slice: asvp.slice,
                    page: asvp.page,
                    victim: asvp.victim,
                };
                acc.0 += threshold_error.error;
                acc.1.insert(svp, threshold_error);
            },
        );

    //let av_best_av_errors
    let av_best_a_erros: HashMap<usize, (ErrorPrediction, HashMap<SVP, ThresholdError>)> =
        accumulate(
            av_detailed_errors.clone(),
            |asvp: ASVP| asvp.attacker,
            || (ErrorPrediction::default(), HashMap::new()),
            |acc: &mut (ErrorPrediction, HashMap<SVP, ThresholdError>),
             threshold_error,
             asvp: ASVP,
             attacker| {
                assert_eq!(attacker, asvp.attacker);
                let svp = SVP {
                    slice: asvp.slice,
                    page: asvp.page,
                    victim: asvp.victim,
                };
                acc.0 += threshold_error.error;
                acc.1.insert(svp, threshold_error);
            },
        );

    // Find best index in each model...

    // CSV output logic

    /* moving parts :
       - order of lines
       - columns and columns header.
       - Probably should be a macro ?
       Or something taking a Vec of Column and getter, plus a vec (or iterator) of 'Keys'
    */

    let mut keys = asvp_threshold_errors.keys().collect::<Vec<&ASVP>>();
    keys.sort_unstable_by(|a: &&ASVP, b: &&ASVP| {
        if a.page > b.page {
            Ordering::Greater
        } else if a.page < b.page {
            Ordering::Less
        } else if a.slice > b.slice {
            Ordering::Greater
        } else if a.slice < b.slice {
            Ordering::Less
        } else if a.attacker > b.attacker {
            Ordering::Greater
        } else if a.attacker < b.attacker {
            Ordering::Less
        } else if a.victim > b.victim {
            Ordering::Greater
        } else if a.victim < b.victim {
            Ordering::Less
        } else {
            Ordering::Equal
        }
    });

    // In theory there should be a way of making such code much more modular.

    let error_header = |name: &str| {
        format!(
            "{}ErrorRate,{}Errors,{}Measures,{}TrueHit,{}TrueMiss,{}FalseHit,{}FalseMiss",
            name, name, name, name, name, name, name
        )
    };

    let header = |name: &str| {
        format!(
            "{}_Threshold,{}_MFH,{}_GlobalErrorRate,{}",
            name,
            name,
            name,
            error_header(&format!("{}_ASVP", name))
        )
    };

    println!(
        "Analysis:Page,Slice,Attacker,Victim,ASVP_Threshold,ASVP_MFH,{},{},{},{},{}",
        error_header("ASVP_"),
        header("ASP"),
        header("SP"),
        header("AV"),
        header("GT")
    );

    let format_error = |error_pred: &ErrorPrediction| {
        format!(
            "{},{},{},{},{},{},{}",
            error_pred.error_rate(),
            error_pred.total_error(),
            error_pred.total(),
            error_pred.true_hit,
            error_pred.true_miss,
            error_pred.false_hit,
            error_pred.false_miss
        )
    };

    let format_detailed_model = |global: &ThresholdError, detailed: &ThresholdError| {
        assert_eq!(global.threshold, detailed.threshold);
        format!(
            "{},{},{},{}",
            global.threshold.bucket_index,
            global.threshold.miss_faster_than_hit,
            global.error.error_rate(),
            format_error(&detailed.error)
        )
    };

    for key in keys {
        print!(
            "Analysis:{},{},{},{},",
            key.page, key.slice, key.attacker, key.victim
        );
        let threshold_error = asvp_threshold_errors[key];
        print!(
            "{},{},{},",
            threshold_error.threshold.bucket_index,
            threshold_error.threshold.miss_faster_than_hit,
            format_error(&threshold_error.error)
        );

        let asp_global = &asp_threshold_errors[&ASP {
            attacker: key.attacker,
            slice: key.slice,
            page: key.page,
        }];
        let asp_detailed = &asp_detailed_errors[key];
        print!("{},", format_detailed_model(asp_global, asp_detailed));

        let sp_global = &sp_threshold_errors[&SP {
            slice: key.slice,
            page: key.page,
        }];
        let sp_detailed = &sp_detailed_errors[key];
        print!("{},", format_detailed_model(sp_global, sp_detailed));

        let av_global = &av_threshold_errors[&AV {
            attacker: key.attacker,
            victim: key.victim,
        }];
        let av_detailed = &av_detailed_errors[key];
        print!("{},", format_detailed_model(av_global, av_detailed));

        let gt_global = &gt_threshold_error;
        let gt_detailed = &gt_detailed_errors[key];
        print!("{},", format_detailed_model(gt_global, gt_detailed));
        println!();
    }

    //The two other CSV are summaries that allowdetermining the best case. Index in the first CSV for the detailed info.
    // Second CSV output logic:

    // Build keys
    let mut keys = asvp_best_av_errors.keys().collect::<Vec<&AV>>();
    keys.sort_unstable_by(|a: &&AV, b: &&AV| {
        if a.attacker > b.attacker {
            Ordering::Greater
        } else if a.attacker < b.attacker {
            Ordering::Less
        } else if a.victim > b.victim {
            Ordering::Greater
        } else if a.victim < b.victim {
            Ordering::Less
        } else {
            Ordering::Equal
        }
    });

    // Print header
    println!(
        "AVAnalysis:Attacker,Victim,{},{}",
        error_header("AVSP_AVAverage_"),
        error_header("AV_AVAverage_")
    );
    //print lines

    for av in keys {
        println!(
            "AVAnalysis:{attacker},{victim},{AVSP},{AV}",
            attacker = av.attacker,
            victim = av.victim,
            AVSP = format_error(&asvp_best_av_errors[av].0),
            AV = format_error(&av_threshold_errors[av].error),
        );
    }

    // Third CSV output logic:

    // Build keys
    let mut keys = asvp_best_a_errors.keys().collect::<Vec<&usize>>();
    keys.sort_unstable();

    println!(
        "AttackerAnalysis:Attacker,{},{},{}",
        error_header("AVSP_AAverage_"),
        error_header("ASP_AAverage_"),
        error_header("AV_AAverage_"),
    );

    for attacker in keys {
        println!(
            "AttackerAnalysis:{attacker},{AVSP},{ASP},{AV}",
            attacker = attacker,
            AVSP = format_error(&asvp_best_a_errors[&attacker].0),
            ASP = format_error(&asp_best_a_errors[&attacker].0),
            AV = format_error(&av_best_a_erros[&attacker].0)
        );
    }
}
