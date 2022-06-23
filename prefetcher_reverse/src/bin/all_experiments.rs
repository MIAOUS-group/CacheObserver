/*
 Objective : run an exploration of patterns of a length given as an arg and test all the possible ones,
 Then proceed with some analysis.

 Probably will use library functions for a lot of it
 (Auto pattern generation belongs in lib.rs, the analysis part may be a little bit more subtle)

 Output, detailed CSV, and well chosen slices + summaries ?

Alternatively, limit to 3 accesses ?

*/

use cache_utils::ip_tool::{Function, TIMED_MACCESS};
use itertools::Itertools;
use nix::sched::sched_yield;
use prefetcher_reverse::{
    pattern_helper, FullPageDualProbeResults, PatternAccess, Prober, PAGE_CACHELINE_LEN,
};

pub const NUM_ITERATION: u32 = 1 << 10;
pub const WARMUP: u32 = 100;

struct Params {
    limit: usize,
    same_ip: bool,
    unique_ip: bool,
}

fn print_tagged_csv(tag: &str, results: Vec<FullPageDualProbeResults>, len: usize) {
    // Print Header,
    println!("{}Functions:i,Addr", tag);
    if !results.is_empty() {
        let first = &results[0];
        for (i, p) in first.pattern.iter().enumerate() {
            println!("{}Functions:{},{:p}", tag, i, p.function.ip)
        }
    }
    println!(
        "{}:{}ProbeAddr,Probe_SF_H,Probe_SF_HR,Probe_SR_H,Probe_SR_HR,Probe_FF_H,Probe_FF_HR",
        tag,
        (0..len)
            .map(|i| {
                format!(
                    "Offset_{i},\
            Offset_{i}_SF_H,Offset_{i}_SF_HR,\
            Offset_{i}_SR_H,Offset_{i}_SR_HR,\
            Offset_{i}_FF_H,Offset_{i}_FF_HR,",
                    i = i
                )
            })
            .format(""),
    );
    // Print each line,
    // TODO : double check with the impl in lib.rs how to extract the various piece of info.
    for res in results {
        assert_eq!(res.pattern.len(), len);

        for probe_addr in 0..PAGE_CACHELINE_LEN {
            let sf_h = res.single_probe_results[probe_addr].flush.probe_result;
            let sr_h = res.single_probe_results[probe_addr].load.probe_result;
            let ff_h = res.full_flush_results.probe_result[probe_addr];
            println!(
                "{}:{}{},{},{},{},{},{},{}",
                tag,
                (0..len)
                    .map(|i| {
                        let sf_h = res.single_probe_results[probe_addr].flush.pattern_result[i];
                        let sf_hr = sf_h as f32 / res.num_iteration as f32;
                        let sr_h = res.single_probe_results[probe_addr].load.pattern_result[i];
                        let sr_hr = sr_h as f32 / res.num_iteration as f32;
                        let ff_h = res.full_flush_results.pattern_result[i];
                        let ff_hr = ff_h as f32 / res.num_iteration as f32;
                        format!(
                            "{},{},{},{},{},{},{},",
                            res.pattern[i].offset, sf_h, sf_hr, sr_h, sr_hr, ff_h, ff_hr
                        )
                    })
                    .format(""),
                probe_addr,
                sf_h,
                sf_h as f32 / res.num_iteration as f32,
                sr_h,
                sr_h as f32 / res.num_iteration as f32,
                ff_h,
                ff_h as f32 / res.num_iteration as f32
            );
        }
    }
}

fn exp(
    i: usize,
    tag: &str,
    patterns: &Vec<Vec<usize>>,
    same_ip: bool,
    unique_ip: bool,
    prober: &mut Prober<1>,
) {
    if same_ip {
        let single_reload = Function::try_new(1, 0, TIMED_MACCESS).unwrap();
        let mut results = Vec::new();
        for pattern in patterns {
            eprintln!("Single IP pattern: {:?}", pattern);
            let single_ip_pattern = pattern_helper(pattern, &single_reload);
            let result = prober.full_page_probe(single_ip_pattern, NUM_ITERATION, WARMUP);
            results.push(result);
            sched_yield().unwrap();
        }
        print_tagged_csv(&format!("SingleIP{}", tag), results, i);
        // generate the vec with a single IP
    }
    if unique_ip {
        let mut functions = Vec::new();
        let rounded_i = i.next_power_of_two();
        for j in 0..i {
            functions.push(Function::try_new(rounded_i, j, TIMED_MACCESS).unwrap());
        }
        let mut results = Vec::new();
        for pattern in patterns {
            eprintln!("Unique IP pattern: {:?}", pattern);

            let unique_ip_pattern = pattern
                .iter()
                .enumerate()
                .map(|(i, &offset)| PatternAccess {
                    function: &functions[i],
                    offset,
                })
                .collect();
            let result = prober.full_page_probe(unique_ip_pattern, NUM_ITERATION, WARMUP);
            results.push(result);
            sched_yield().unwrap();
        }
        print_tagged_csv(&format!("UniqueIPs{}", tag), results, i);
    }
}

/* TODO change access patterns
- We want patterns for i,j in [0,64]^2
A (i,i+k,j)
B (i,i-k,j)
C (i,j,j+k)
D (i,j,j-k)

with k in 1,2,3,8, plus possibly others.
4 access patterns will probably come in later

In addition consider base + stride + len patterns, with a well chosen set of length, strides and bases
len to be considered 2,3,4
Identifiers :
E 2
 */

fn main() {
    // TODO Argument parsing
    let args = Params {
        limit: 2,
        same_ip: true,
        unique_ip: true,
    };

    let mut experiments: Vec<(String, usize, usize, Box<dyn Fn(usize, usize) -> Vec<usize>>)> = vec![];
    for class in [
        (
            "",
            Box::new(|k: usize| {

                let f = Box::new(move |i, j| {
                    let mut v = vec![
                        i,j,
                    ];
                    v.truncate(k);
                    v
                }) as Box<dyn Fn(usize, usize) -> Vec<usize>>;
                let i_limit = if k > 0 {
                    PAGE_CACHELINE_LEN
                } else {
                    1
                };
                let j_limit = if k > 1 {
                    PAGE_CACHELINE_LEN
                } else {
                    1
                };
                (i_limit,j_limit,f)
            }) as Box<dyn Fn(usize) -> (usize, usize, Box<dyn Fn(usize, usize) -> Vec<usize>>)>,
        ),
    ] {
        for k in [0,1,2] {
            let exp = class.1(k);
            experiments.push((format!("{}{}", class.0, k),exp.0, exp.1, exp.2));
        }
    }

    for class in [
        (
            "A",
            Box::new(|k| {
                Box::new(move |i, j| vec![i, (i + k) % PAGE_CACHELINE_LEN, j])
                    as Box<dyn Fn(usize, usize) -> Vec<usize>>
            }) as Box<dyn Fn(usize) -> Box<dyn Fn(usize, usize) -> Vec<usize>>>,
        ),
        (
            "B",
            Box::new(|k| {
                Box::new(move |i, j| vec![i, (i - k) % PAGE_CACHELINE_LEN, j])
                    as Box<dyn Fn(usize, usize) -> Vec<usize>>
            }) as Box<dyn Fn(usize) -> Box<dyn Fn(usize, usize) -> Vec<usize>>>,
        ),
        (
            "C",
            Box::new(|k| {
                Box::new(move |i, j| vec![i, j, (j + k) % PAGE_CACHELINE_LEN])
                    as Box<dyn Fn(usize, usize) -> Vec<usize>>
            }) as Box<dyn Fn(usize) -> Box<dyn Fn(usize, usize) -> Vec<usize>>>,
        ),
        (
            "D",
            Box::new(|k| {
                Box::new(move |i, j| vec![i, j, (j - k) % PAGE_CACHELINE_LEN])
                    as Box<dyn Fn(usize, usize) -> Vec<usize>>
            }) as Box<dyn Fn(usize) -> Box<dyn Fn(usize, usize) -> Vec<usize>>>,
        ),
    ] {
        for k in [1, 2, 3, 4, 8] {
            experiments.push((format!("{}{}", class.0, k), PAGE_CACHELINE_LEN, PAGE_CACHELINE_LEN, class.1(k)));
        }
    }

    for class in [(
        "E",
        Box::new(|len: usize| {
            Box::new(move |base, stride| {
                let mut res = vec![base];
                for i in 0..len {
                    res.push((base + stride * (i + 1)) % PAGE_CACHELINE_LEN)
                }
                res
            }) as Box<dyn Fn(usize, usize) -> Vec<usize>>
        }) as Box<dyn Fn(usize) -> Box<dyn Fn(usize, usize) -> Vec<usize>>>,
    )] {
        for len in [2, 3, 4] {
            experiments.push((format!("{}{}", class.0, len), PAGE_CACHELINE_LEN, PAGE_CACHELINE_LEN, class.1(len)));
        }
    }

    for class in [
        (
            "F",
            Box::new(|k: isize| {
                Box::new(move |i, j| {
                    vec![
                        i,
                        (i as isize + k + PAGE_CACHELINE_LEN as isize) as usize
                            % PAGE_CACHELINE_LEN,
                        j,
                        (i as isize + 2 * k + PAGE_CACHELINE_LEN as isize) as usize
                            % PAGE_CACHELINE_LEN,
                    ]
                }) as Box<dyn Fn(usize, usize) -> Vec<usize>>
            }) as Box<dyn Fn(isize) -> Box<dyn Fn(usize, usize) -> Vec<usize>>>,
        ),
    ] {
        for k in [4 as isize, 3, 2, 1, -1, -2, -3, -4] {
            experiments.push((format!("{}{}", class.0, k), PAGE_CACHELINE_LEN, PAGE_CACHELINE_LEN, class.1(k)));
        }
    }

    let mut prober = Prober::<1>::new(63).unwrap();

    for experiment in experiments {
        let tag = &experiment.0;
        let mut patterns = vec![];
        for i in 0..experiment.1 {
            for j in 0..experiment.2 {
                patterns.push(experiment.3(i, j))
            }
        }
        let i = patterns[0].len();
        exp(i, tag, &patterns, args.same_ip, args.unique_ip, &mut prober);
    }
}
