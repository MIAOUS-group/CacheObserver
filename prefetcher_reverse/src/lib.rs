#![feature(global_asm)]
#![deny(unsafe_op_in_unsafe_fn)]

use std::fmt::{Display, Error, Formatter};
use std::iter::{Cycle, Peekable};
use std::ops::Range;
use std::{thread, time};

use nix::sys::stat::stat;
use rand::seq::SliceRandom;

use basic_timing_cache_channel::{
    CalibrationStrategy, TopologyAwareError, TopologyAwareTimingChannel,
};
use cache_side_channel::CacheStatus::{Hit, Miss};
use cache_side_channel::{
    set_affinity, CacheStatus, ChannelHandle, CoreSpec, MultipleAddrCacheSideChannel,
    SingleAddrCacheSideChannel,
};
use cache_utils::calibration::{only_reload, Threshold, PAGE_LEN};
use cache_utils::ip_tool::Function;
use cache_utils::mmap::MMappedMemory;
use cache_utils::rdtsc_nofence;
use flush_flush::{FFHandle, FFPrimitives, FlushAndFlush};
use flush_reload::naive::{NFRHandle, NaiveFlushAndReload};
use flush_reload::{FRHandle, FRPrimitives, FlushAndReload};

use crate::Probe::{Flush, FullFlush, Load};

// NB these may need to be changed / dynamically measured.
pub const CACHE_LINE_LEN: usize = 64;
pub const PAGE_CACHELINE_LEN: usize = PAGE_LEN / CACHE_LINE_LEN;

pub const CALIBRATION_STRAT: CalibrationStrategy = CalibrationStrategy::ASVP;

pub struct Prober<const GS: usize> {
    pages: Vec<MMappedMemory<u8>>,
    ff_handles: Vec<Vec<FFHandle>>,
    fr_handles: Vec<Vec<FRHandle>>,
    //fr_handles: Vec<Vec<NFRHandle>>,
    page_indexes: Peekable<Cycle<Range<usize>>>,
    ff_channel: FlushAndFlush,
    fr_channel: FlushAndReload,
    //fr_channel: NaiveFlushAndReload,
    delay: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Probe {
    Load(usize),
    Flush(usize),
    FullFlush,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbeType {
    Load,
    Flush,
    FullFlush,
}

#[derive(Debug, Clone)]
pub struct PatternAccess<'a> {
    pub function: &'a Function,
    pub offset: usize,
}

#[derive(Debug)]
pub struct ProbePattern<'a> {
    pub pattern: Vec<PatternAccess<'a>>,
    pub probe: Probe,
}

#[derive(Debug)]
pub enum ProberError {
    NoMem(nix::Error),
    TopologyError(TopologyAwareError),
    Nix(nix::Error),
}

/**
Result of running a probe pattern num_iteration times,
*/
pub type SinglePR = u64;
pub type FullPR = Vec<u64>;

#[derive(Debug)]
pub enum ProbeResult {
    Load(SinglePR),
    Flush(SinglePR),
    FullFlush(FullPR),
}

#[derive(Debug)]
pub struct ProbePatternResult {
    pub num_iteration: u32,
    pub pattern_result: Vec<u64>,
    pub probe_result: ProbeResult,
}

#[derive(Debug)]
pub struct DPRItem<PR> {
    pub pattern_result: Vec<u64>,
    pub probe_result: PR,
}

#[derive(Debug)]
pub struct DualProbeResult {
    pub probe_offset: usize,
    pub load: DPRItem<SinglePR>,
    pub flush: DPRItem<SinglePR>,
}

#[derive(Debug)]
pub struct FullPageDualProbeResults<'a> {
    pub pattern: Vec<PatternAccess<'a>>,
    pub num_iteration: u32,
    pub single_probe_results: Vec<DualProbeResult>,
    pub full_flush_results: DPRItem<FullPR>,
}

#[derive(Debug, Clone)]
pub struct SingleProbeResult {
    pub probe_offset: usize,
    pub pattern_result: Vec<u64>,
    pub probe_result: u64,
}

#[derive(Debug)]
pub struct FullPageSingleProbeResult<'a, const GS: usize> {
    pub pattern: Vec<PatternAccess<'a>>,
    pub probe_type: ProbeType,
    pub num_iteration: u32,
    pub results: Vec<SingleProbeResult>,
}

fn delay(d: u64) {
    let mut t = unsafe { rdtsc_nofence() };
    let end = t + d;
    while t < end {
        t = unsafe { rdtsc_nofence() };
    }
}

// Helper function
/**
This function is a helper that determine what is the maximum stride for a pattern of len accesses
starting at a given offset, both forward and backward.

Special case for length 0.
 */
fn max_stride(offset: usize, len: usize) -> (isize, isize) {
    if len == 0 {
        (0, 0)
    } else {
        let min = -((offset / (len * CACHE_LINE_LEN)) as isize);
        let max = ((PAGE_LEN - offset) / (len * CACHE_LINE_LEN)) as isize;
        (min, max)
    }
}

impl<const GS: usize> Prober<GS> {
    // turn page into page groups, which can vary in size.
    // calibrate on all pages in a group with offsets within the groups.
    // keep track of the max offset
    pub fn new(num_pages: usize) -> Result<Prober<GS>, ProberError> {
        let mut vec = Vec::new();
        let mut handles = Vec::new();
        let (mut ff_channel, cpuset, core) = match FlushAndFlush::new_any_single_core() {
            Ok(res) => res,
            Err(err) => {
                return Err(ProberError::TopologyError(err));
            }
        };
        let old_affinity = match set_affinity(&ff_channel.main_core()) {
            Ok(old) => old,
            Err(nixerr) => return Err(ProberError::Nix(nixerr)),
        };
        /*let mut fr_channel = NaiveFlushAndReload::new(Threshold {
            bucket_index: 315,
            miss_faster_than_hit: false,
        });*/
        let mut fr_channel = match FlushAndReload::new(core, core, CALIBRATION_STRAT) {
            Ok(res) => res,
            Err(err) => {
                return Err(ProberError::TopologyError(err));
            }
        };

        for i in 0..num_pages {
            let mut p = match MMappedMemory::<u8>::try_new(PAGE_LEN * GS, false, false, |j| {
                (j / CACHE_LINE_LEN + i * PAGE_CACHELINE_LEN) as u8
            }) {
                Ok(p) => p,
                Err(e) => {
                    return Err(ProberError::NoMem(e));
                }
            };
            let page_addresses = ((0..(PAGE_LEN * GS)).step_by(CACHE_LINE_LEN))
                .map(|offset| &p[offset] as *const u8);
            let ff_page_handles = unsafe { ff_channel.calibrate(page_addresses.clone()) }.unwrap();
            let fr_page_handles = unsafe { fr_channel.calibrate_single(page_addresses) }.unwrap();

            vec.push(p);
            handles.push((ff_page_handles, fr_page_handles));
        }

        let mut page_indexes = (0..(handles.len())).cycle().peekable();

        handles.shuffle(&mut rand::thread_rng());

        let mut ff_handles = Vec::new();
        let mut fr_handles = Vec::new();

        for (ff_handle, fr_handle) in handles {
            ff_handles.push(ff_handle);
            fr_handles.push(fr_handle);
        }

        Ok(Prober {
            pages: vec,
            ff_handles,
            fr_handles,
            page_indexes,
            ff_channel,
            fr_channel,
            delay: 0,
        })
    }

    pub fn set_delay(&mut self, delay: u64) {
        self.delay = delay;
    }

    fn probe_pattern_once(
        &mut self,
        pattern: &ProbePattern,
        result: Option<&mut ProbePatternResult>,
    ) {
        enum ProbeOutput {
            Single(CacheStatus),
            Full(Vec<(*const u8, CacheStatus)>),
        }

        self.page_indexes.next();
        let page_index = *self.page_indexes.peek().unwrap();

        let mut ff_handles = self.ff_handles[page_index]
            .iter_mut() /*.rev()*/
            .collect();

        unsafe { self.ff_channel.prepare(&mut ff_handles) };

        let mut pattern_res = vec![0; pattern.pattern.len()];
        for (i, access) in pattern.pattern.iter().enumerate() {
            let h = &mut self.fr_handles[page_index][access.offset];
            let pointer: *const u8 = h.to_const_u8_pointer();
            pattern_res[i] = unsafe { (access.function.fun)(pointer) };
            // TODO IP : This is where the pattern access need to be done using pattern.function instead.
            //pattern_res[i] = unsafe { self.fr_channel.test_debug(h, false) }.unwrap().1;
            delay(self.delay);
            //pattern_res[i] = unsafe { self.fr_channel.test_single(h, false) }.unwrap();
            //unsafe { only_reload(h.to_const_u8_pointer()) };
        }

        let mut probe_out = match pattern.probe {
            Load(offset) => {
                let h = &mut self.fr_handles[page_index][offset];
                ProbeOutput::Single(unsafe { self.fr_channel.test_single(h, false) }.unwrap())
            }
            Flush(offset) => {
                let h = &mut self.ff_handles[page_index][offset];
                ProbeOutput::Single(unsafe { self.ff_channel.test_single(h, false) }.unwrap())
            }
            Probe::FullFlush => {
                ProbeOutput::Full(unsafe { self.ff_channel.test(&mut ff_handles, true).unwrap() })
            }
        };

        if let Some(result_ref) = result {
            result_ref.num_iteration += 1;

            match result_ref.probe_result {
                ProbeResult::Load(ref mut r) | ProbeResult::Flush(ref mut r) => {
                    if let ProbeOutput::Single(status) = probe_out {
                        if status == Hit {
                            *r += 1;
                        }
                    } else {
                        panic!()
                    }
                }
                ProbeResult::FullFlush(ref mut v) => {
                    if let ProbeOutput::Full(vstatus) = probe_out {
                        for (i, status) in vstatus.iter().enumerate() {
                            if status.1 == Hit {
                                v[/*63 -*/ i] += 1;
                            }
                        }
                    } else {
                        panic!()
                    }
                }
            }

            for (i, res) in pattern_res.into_iter().enumerate() {
                //if res == Hit {
                result_ref.pattern_result[i] += res;
                //}
            }
        }
    }

    pub fn probe_pattern(
        &mut self,
        pattern: &ProbePattern,
        num_iteration: u32,
        warmup: u32,
    ) -> ProbePatternResult {
        let mut result = ProbePatternResult {
            num_iteration: 0,
            pattern_result: vec![0; pattern.pattern.len()],
            probe_result: match pattern.probe {
                Load(_) => ProbeResult::Load(0),
                Flush(_) => ProbeResult::Flush(0),
                Probe::FullFlush => ProbeResult::FullFlush(vec![0; PAGE_CACHELINE_LEN * GS]),
            },
        };
        for _ in 0..warmup {
            self.probe_pattern_once(pattern, None);
        }

        for _ in 0..num_iteration {
            self.probe_pattern_once(pattern, Some(&mut result));
        }

        result
    }

    fn full_page_probe_helper<'a>(
        &mut self,
        pattern: &mut ProbePattern<'a>,
        probe_type: ProbeType,
        num_iteration: u32,
        warmup: u32,
    ) -> FullPageSingleProbeResult<'a, GS> {
        let mut result = FullPageSingleProbeResult {
            pattern: pattern.pattern.clone(),
            probe_type,
            num_iteration,
            results: vec![
                SingleProbeResult {
                    probe_offset: 0,
                    pattern_result: vec![],
                    probe_result: 0
                };
                64
            ],
        };
        for offset in (0..(PAGE_CACHELINE_LEN * GS))
        /*.rev()*/
        {
            // Reversed FIXME
            pattern.probe = match probe_type {
                ProbeType::Load => Probe::Load(offset),
                ProbeType::Flush => Probe::Flush(offset),
                ProbeType::FullFlush => FullFlush,
            };
            let r = self.probe_pattern(pattern, num_iteration, warmup);
            result.results[offset] = SingleProbeResult {
                probe_offset: offset,
                pattern_result: r.pattern_result,
                probe_result: match r.probe_result {
                    ProbeResult::Load(r) => r,
                    ProbeResult::Flush(r) => r,
                    ProbeResult::FullFlush(r) => r[offset],
                },
            };
        }
        result
    }

    pub fn full_page_probe<'a>(
        &mut self,
        pattern: Vec<PatternAccess<'a>>,
        num_iteration: u32,
        warmup: u32,
    ) -> FullPageDualProbeResults<'a> {
        let mut probe_pattern = ProbePattern {
            pattern: pattern,
            probe: Probe::FullFlush,
        };
        let res_flush = self.full_page_probe_helper(
            &mut probe_pattern,
            ProbeType::Flush,
            num_iteration,
            warmup,
        );
        let res_load =
            self.full_page_probe_helper(&mut probe_pattern, ProbeType::Load, num_iteration, warmup);
        probe_pattern.probe = FullFlush;
        let res_full_flush = self.probe_pattern(&probe_pattern, num_iteration, warmup);
        // TODO results

        FullPageDualProbeResults {
            pattern: probe_pattern.pattern,
            num_iteration,
            single_probe_results: res_flush
                .results
                .into_iter()
                .enumerate()
                .zip(res_load.results.into_iter())
                .map(|((offset, flush), load)| DualProbeResult {
                    probe_offset: offset,
                    load: DPRItem {
                        pattern_result: load.pattern_result,
                        probe_result: load.probe_result,
                    },
                    flush: DPRItem {
                        pattern_result: flush.pattern_result,
                        probe_result: flush.probe_result,
                    },
                })
                .collect(),
            full_flush_results: DPRItem {
                pattern_result: res_full_flush.pattern_result,
                probe_result: match res_full_flush.probe_result {
                    ProbeResult::FullFlush(r) => r,
                    _ => {
                        unreachable!()
                    }
                },
            },
        }
    }
}

impl<'a> Display for FullPageDualProbeResults<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut indices = vec![None; self.single_probe_results.len()];
        let pat_len = self.pattern.len();
        let divider = (self.single_probe_results.len() * self.num_iteration as usize) as f32;
        for (i, access) in self.pattern.iter().enumerate() {
            indices[access.offset] = Some(i);
        }
        // Display header
        let mut r = writeln!(
            f,
            "{:11} | {:^37} | {:^37} | {:^37}",
            "", "Single Flush", "Single Load", "Full Flush"
        );
        match r {
            Ok(_) => {}
            Err(e) => {
                return Err(e);
            }
        }
        let mut r = writeln!(f, "{:^3} {:>7} | {:>8} {:^9} {:>8} {:^9} | {:>8} {:^9} {:>8} {:^9} | {:>8} {:^9} {:>8} {:^9}",
                       "pat", "offset",
               "SF Ac H", "SF Ac HR", "SF Pr H", "SF Pr HR",
               "SR Ac H", "SR Ac HR", "SR Pr H", "SR Pr HR",
               "FF Ac H", "FF Ac HR", "FF Pr H", "FF Pr HR");
        match r {
            Ok(_) => {}
            Err(e) => {
                return Err(e);
            }
        }

        for i in 0..(self.single_probe_results.len()) {
            let index = indices[i];

            let (pat, sf_ac_h, sf_ac_hr, sr_ac_h, sr_ac_hr, ff_ac_h, ff_ac_hr) = match index {
                None => (
                    String::from(""),
                    String::from(""),
                    String::from(""),
                    String::from(""),
                    String::from(""),
                    String::from(""),
                    String::from(""),
                ),
                Some(index) => {
                    let pat = format!("{:3}", index);
                    let sf_ac: u64 = self
                        .single_probe_results
                        .iter()
                        .map(|d| d.flush.pattern_result[index])
                        .sum();
                    let sf_ac_h = format!("{:8}", sf_ac);
                    let sf_ac_hr = format!("{:9.7}", sf_ac as f32 / divider);

                    let sr_ac: u64 = self
                        .single_probe_results
                        .iter()
                        .map(|d| d.load.pattern_result[index])
                        .sum();
                    let sr_ac_h = format!("{:8}", sr_ac);
                    let sr_ac_hr = format!("{:9.7}", sr_ac as f32 / divider);

                    let ff_ac = self.full_flush_results.pattern_result[index];
                    let ff_ac_h = format!("{:8}", ff_ac);
                    let ff_ac_hr = format!("{:9.7}", ff_ac as f32 / self.num_iteration as f32);
                    (pat, sf_ac_h, sf_ac_hr, sr_ac_h, sr_ac_hr, ff_ac_h, ff_ac_hr)
                }
            };

            let sf_pr = self.single_probe_results[i].flush.probe_result;
            let sf_pr_h = format!("{:8}", sf_pr);
            let sf_pr_hr = format!("{:9.7}", sf_pr as f32 / self.num_iteration as f32);

            let sr_pr = self.single_probe_results[i].load.probe_result;
            let sr_pr_h = format!("{:8}", sr_pr);
            let sr_pr_hr = format!("{:9.7}", sr_pr as f32 / self.num_iteration as f32);

            let ff_pr = self.full_flush_results.probe_result[i];
            let ff_pr_h = format!("{:8}", ff_pr);
            let ff_pr_hr = format!("{:9.7}", ff_pr as f32 / self.num_iteration as f32);

            r = writeln!(f, "{:>3} {:>7} | {:>8} {:^9} {:>8} {:^9} | {:>8} {:^9} {:>8} {:^9} | {:>8} {:^9} {:>8} {:^9}",
                             pat, i,
                             sf_ac_h, sf_ac_hr, sf_pr_h, sf_pr_hr,
                             sr_ac_h, sr_ac_hr, sr_pr_h, sr_pr_hr,
                             ff_ac_h, ff_ac_hr, ff_pr_h, ff_pr_hr);
            match r {
                Ok(_) => {}
                Err(e) => {
                    return Err(e);
                }
            };
            // display lines
        }
        write!(f, "Num_iteration: {}", self.num_iteration)
    }
}

pub fn reference_patterns() -> [(&'static str, Vec<usize>); 9] {
    [
        ("Pattern 1", vec![0, 1, 2, 3]),
        ("Pattern 2", vec![0, 1]),
        ("Pattern 3", vec![0, 19]),
        ("Pattern 4 (I)", vec![0, 1, 2, 44]),
        ("Pattern 4 (II)", vec![63, 62, 61, 19]),
        ("Pattern 5 (I)", vec![0, 1, 2, 63, 62, 44]),
        ("Pattern 5 (II)", vec![63, 62, 61, 0, 1, 2, 19]),
        ("Pattern 5 (III)", vec![63, 62, 61, 0, 1, 2, 44]),
        ("Pattern 5 (IV)", vec![0, 1, 2, 63, 62, 61, 19]),
    ]
}

pub fn pattern_helper<'a>(offsets: &Vec<usize>, function: &'a Function) -> Vec<PatternAccess<'a>> {
    offsets
        .into_iter()
        .map(|i| PatternAccess {
            function,
            offset: *i,
        })
        .collect()
}
