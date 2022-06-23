#![deny(unsafe_op_in_unsafe_fn)]

// TODO

// Common logic for the ability to calibrate along slices
// Core issues should be orthogonal
// Extend to multithread ?

// Should be used by F+F and non Naive F+R

//use crate::naive::NaiveTimingChannelHandle;
use cache_side_channel::table_side_channel::{
    MultipleTableCacheSideChannel, SingleTableCacheSideChannel, TableAttackResult,
    TableCacheSideChannel,
};
use cache_side_channel::SideChannelError::AddressNotReady;
use cache_side_channel::{
    BitIterator, CacheStatus, ChannelFatalError, ChannelHandle, CoreSpec, CovertChannel,
    MultipleAddrCacheSideChannel, SideChannelError, SingleAddrCacheSideChannel,
};
use cache_utils::calibration::{
    accumulate, calibrate_fixed_freq_2_thread, calibration_result_to_ASVP,
    get_cache_attack_slicing, get_vpn, map_values, only_flush, only_reload, CalibrateOperation2T,
    CalibrationOptions, ErrorPrediction, ErrorPredictions, HashMap, HistParams, HistogramCumSum,
    PotentialThresholds, Slice, Threshold, ThresholdError, Verbosity, ASVP, AV,
    CFLUSH_BUCKET_NUMBER, CFLUSH_BUCKET_SIZE, CFLUSH_NUM_ITER, CLFLUSH_NUM_ITERATION_AV, PAGE_LEN,
    SP, VPN,
};
use cache_utils::complex_addressing::{CacheAttackSlicing, CacheSlicing};
use cache_utils::mmap::MMappedMemory;
use cache_utils::{find_core_per_socket, flush, maccess, noop};
use nix::sched::sched_getaffinity;
use nix::sched::CpuSet;
use nix::unistd::Pid;
use std::collections::HashSet;
use std::fmt;
use std::fmt::{Debug, Formatter};
use std::ptr::slice_from_raw_parts;

pub mod naive;

const CACHE_LINE_LENGTH: usize = 64; // FIXME MAGIC to be autodetected.

pub trait TimingChannelPrimitives: Debug + Send + Sync + Default {
    unsafe fn attack(&self, addr: *const u8) -> u64;
    const NEED_RESET: bool;
}

#[derive(Debug)]
pub struct TopologyAwareTimingChannelHandle {
    threshold: Threshold,
    vpn: VPN, // what is this field used for
    addr: *const u8,
    ready: bool,
    calibration_epoch: usize,
}

pub struct CovertChannelHandle<T: MultipleAddrCacheSideChannel>(T::Handle);

impl ChannelHandle for TopologyAwareTimingChannelHandle {
    fn to_const_u8_pointer(&self) -> *const u8 {
        self.addr
    }
}

#[derive(Debug)]
pub enum TopologyAwareError {
    NoSlicing,
    Nix(nix::Error),
    NeedRecalibration,
}

#[derive(Debug, Copy, Clone)]
pub enum CalibrationStrategy {
    ASVP,
    AV,
    AVSockets,
}

#[derive(Debug)]
enum ThresholdStrat {
    ASVP(HashMap<SP, ThresholdError>),
    AV(ThresholdError),
    AVSockets(ThresholdError),
}

pub struct TopologyAwareTimingChannel<T: TimingChannelPrimitives> {
    slicing: CacheAttackSlicing,
    main_core: usize,   // aka attacker
    helper_core: usize, // aka victim
    t: T,
    thresholds: ThresholdStrat,
    addresses: HashSet<*const u8>,
    preferred_address: HashMap<VPN, *const u8>,
    calibration_epoch: usize,
    calibration_strategy: CalibrationStrategy,
}

unsafe impl<T: TimingChannelPrimitives + Send> Send for TopologyAwareTimingChannel<T> {}
unsafe impl<T: TimingChannelPrimitives + Sync> Sync for TopologyAwareTimingChannel<T> {}

impl<T: TimingChannelPrimitives> TopologyAwareTimingChannel<T> {
    pub fn new(
        main_core: usize,
        helper_core: usize,
        strat: CalibrationStrategy,
    ) -> Result<Self, TopologyAwareError> {
        if let Some(slicing) = get_cache_attack_slicing(find_core_per_socket(), CACHE_LINE_LENGTH) {
            let ret = Self {
                thresholds: match strat {
                    CalibrationStrategy::ASVP => ThresholdStrat::ASVP(Default::default()),
                    CalibrationStrategy::AV => ThresholdStrat::AV(Default::default()),
                    CalibrationStrategy::AVSockets => ThresholdStrat::AVSockets(Default::default()),
                },
                addresses: Default::default(),
                slicing,
                main_core,
                helper_core,
                preferred_address: Default::default(),
                t: Default::default(),
                calibration_epoch: 0,
                calibration_strategy: strat,
            };
            Ok(ret)
        } else {
            Err(TopologyAwareError::NoSlicing)
        }
    }

    // Takes a buffer / list of addresses or pages
    // Takes a list of core pairs
    // Run optimized calibration and processes results
    fn calibration_for_core_pairs<'a>(
        t: &T,
        core_pairs: impl Iterator<Item = (usize, usize)> + Clone,
        pages: impl Iterator<Item = &'a [u8]>,
        strat: CalibrationStrategy,
    ) -> Result<HashMap<AV, (ErrorPrediction, ThresholdStrat)>, TopologyAwareError> {
        let core_per_socket = find_core_per_socket();

        let operations = [
            CalibrateOperation2T {
                prepare: maccess::<u8>,
                op: T::attack,
                name: "hit",
                display_name: "hit",
                t: &t,
            },
            CalibrateOperation2T {
                prepare: flush,
                op: T::attack,
                name: "miss",
                display_name: "miss",
                t: &t,
            },
        ];
        const HIT_INDEX: usize = 0;
        const MISS_INDEX: usize = 1;

        let mut calibrate_results2t_vec = Vec::new();

        let slicing = match get_cache_attack_slicing(core_per_socket, CACHE_LINE_LENGTH) {
            Some(s) => s,
            None => {
                return Err(TopologyAwareError::NoSlicing);
            }
        };
        let h = |addr: usize| slicing.hash(addr);

        for page in pages {
            // FIXME Cache line size is magic
            let mut r = unsafe {
                calibrate_fixed_freq_2_thread(
                    &page[0] as *const u8,
                    CACHE_LINE_LENGTH,
                    page.len() as isize,
                    &mut core_pairs.clone(),
                    &operations,
                    CalibrationOptions {
                        hist_params: HistParams {
                            bucket_number: CFLUSH_BUCKET_NUMBER,
                            bucket_size: CFLUSH_BUCKET_SIZE,
                            iterations: match strat {
                                CalibrationStrategy::ASVP => CFLUSH_NUM_ITER,
                                CalibrationStrategy::AV => CLFLUSH_NUM_ITERATION_AV,
                                CalibrationStrategy::AVSockets => CLFLUSH_NUM_ITERATION_AV >> 1,
                            },
                        },
                        verbosity: Verbosity::NoOutput,
                        optimised_addresses: true,
                    },
                    core_per_socket,
                )
            };
            calibrate_results2t_vec.append(&mut r);
        }
        // FIXME analysis for various strategies

        match strat {
            CalibrationStrategy::ASVP => {
                let analysis: HashMap<ASVP, ThresholdError> = calibration_result_to_ASVP(
                    calibrate_results2t_vec,
                    |cal_1t_res| {
                        let e = ErrorPredictions::predict_errors(HistogramCumSum::from_calibrate(
                            cal_1t_res, HIT_INDEX, MISS_INDEX,
                        ));
                        PotentialThresholds::minimizing_total_error(e)
                            .median()
                            .unwrap()
                    },
                    &h,
                )
                .map_err(|e| TopologyAwareError::Nix(e))?;
                let asvp_best_av_errors: HashMap<AV, (ErrorPrediction, ThresholdStrat)> =
                    accumulate(
                        analysis,
                        |asvp: ASVP| AV {
                            attacker: asvp.attacker,
                            victim: asvp.victim,
                        },
                        || {
                            (
                                ErrorPrediction::default(),
                                ThresholdStrat::ASVP(HashMap::new()),
                            )
                        },
                        |acc: &mut (ErrorPrediction, ThresholdStrat),
                         threshold_error,
                         asvp: ASVP,
                         av: AV| {
                            assert_eq!(av.attacker, asvp.attacker);
                            assert_eq!(av.victim, asvp.victim);
                            let sp = SP {
                                slice: asvp.slice,
                                page: asvp.page,
                            };
                            acc.0 += threshold_error.error;
                            if let ThresholdStrat::ASVP(hashmap) = &mut acc.1 {
                                hashmap.insert(sp, threshold_error);
                            }
                        },
                    );
                Ok(asvp_best_av_errors)
            }
            CalibrationStrategy::AV => {
                let analysis: HashMap<ASVP, ErrorPredictions> = calibration_result_to_ASVP(
                    calibrate_results2t_vec,
                    |cal_1t_res| {
                        let e = ErrorPredictions::predict_errors(HistogramCumSum::from_calibrate(
                            cal_1t_res, HIT_INDEX, MISS_INDEX,
                        ));
                        e
                    },
                    &h,
                )
                .map_err(|e| TopologyAwareError::Nix(e))?;
                let av_analysis = accumulate(
                    analysis,
                    |asvp: ASVP| AV {
                        attacker: asvp.attacker,
                        victim: asvp.victim,
                    },
                    || ErrorPredictions::empty(CFLUSH_BUCKET_NUMBER),
                    |accumulator: &mut ErrorPredictions,
                     error_preds: ErrorPredictions,
                     _key,
                     _rkey| {
                        *accumulator += error_preds;
                    },
                );
                let av_threshold_errors: HashMap<AV, (ErrorPrediction, ThresholdStrat)> =
                    map_values(av_analysis, |error_predictions: ErrorPredictions, _| {
                        let threshold_error =
                            PotentialThresholds::minimizing_total_error(error_predictions)
                                .median()
                                .unwrap();
                        (threshold_error.error, ThresholdStrat::AV(threshold_error))
                    });
                Ok(av_threshold_errors)
                // Now consolidate over all slices.
                // The accumulator is ErrorPredictions ? (Check if it can be summed, otherwise revert to histograms, accumulate and then predict errors)
                // For each AV : build the generic Threshold + error, (using max ?) then determine the best core.
            }
            CalibrationStrategy::AVSockets => {
                let analysis: HashMap<ASVP, ErrorPredictions> = calibration_result_to_ASVP(
                    calibrate_results2t_vec,
                    |cal_1t_res| {
                        let e = ErrorPredictions::predict_errors(HistogramCumSum::from_calibrate(
                            cal_1t_res, HIT_INDEX, MISS_INDEX,
                        ));
                        e
                    },
                    &h,
                )
                .map_err(|e| TopologyAwareError::Nix(e))?;
                unimplemented!("Requires socket identification support") // Will stay unimplemented for now (10/10/2021)
            }
        }
    }

    pub fn new_with_core_pairs(
        core_pairs: impl Iterator<Item = (usize, usize)> + Clone,
        strat: CalibrationStrategy,
    ) -> Result<(Self, usize, usize), TopologyAwareError> {
        let m = MMappedMemory::new(PAGE_LEN, false, false, |i| i as u8);
        let array: &[u8] = m.slice();

        let t = Default::default();

        let res = Self::calibration_for_core_pairs(&t, core_pairs, vec![array].into_iter(), strat)?;

        let mut best_error_rate = 1.0;
        let mut best_av = Default::default();

        // Select the proper core

        for (av, (global_error_pred, thresholds)) in res.iter() {
            if global_error_pred.error_rate() < best_error_rate {
                best_av = *av;
                best_error_rate = global_error_pred.error_rate();
            }
        }
        Self::new(best_av.attacker, best_av.victim, strat)
            .map(|this| (this, best_av.attacker, best_av.victim))

        // Set no threshold as calibrated on local array that will get dropped.
    }

    pub fn new_any_single_core() -> Result<(Self, CpuSet, usize), TopologyAwareError> {
        // Generate core iterator
        let mut core_pairs: Vec<(usize, usize)> = Vec::new();

        let old = sched_getaffinity(Pid::from_raw(0)).unwrap();

        for i in 0..CpuSet::count() {
            if old.is_set(i).unwrap() {
                core_pairs.push((i, i));
            }
        }

        // Generate all single core pairs

        // Call out to private constructor that takes a core pair list, determines best and makes the choice.
        // The private constructor will set the correct affinity for main (attacker thread)

        Self::new_with_core_pairs(core_pairs.into_iter(), CalibrationStrategy::ASVP).map(
            |(channel, attacker, victim)| {
                assert_eq!(attacker, victim);
                (channel, old, attacker)
            },
        )
    }

    pub fn new_any_two_core(
        distinct: bool,
        strat: CalibrationStrategy,
    ) -> Result<(Self, CpuSet, usize, usize), TopologyAwareError> {
        let old = sched_getaffinity(Pid::from_raw(0)).unwrap();

        let mut core_pairs: Vec<(usize, usize)> = Vec::new();

        for i in 0..CpuSet::count() {
            if old.is_set(i).unwrap() {
                for j in 0..CpuSet::count() {
                    if old.is_set(j).unwrap() {
                        if i != j || !distinct {
                            core_pairs.push((i, j));
                        }
                    }
                }
            }
        }

        Self::new_with_core_pairs(core_pairs.into_iter(), strat).map(
            |(channel, attacker, victim)| {
                if distinct {
                    assert_ne!(attacker, victim);
                }
                (channel, old, attacker, victim)
            },
        )
    }

    fn get_slice(&self, addr: *const u8) -> Slice {
        // This will not work well if slicing is not known FIXME
        self.slicing.hash(addr as usize)
    }

    pub fn set_cores(&mut self, main: usize, helper: usize) -> Result<(), TopologyAwareError> {
        let old_main = self.main_core;
        let old_helper = self.helper_core;

        self.main_core = main;
        self.helper_core = helper;

        match self.recalibrate(self.calibration_strategy) {
            Ok(()) => Ok(()),
            Err(e) => {
                self.main_core = old_main;
                self.helper_core = old_helper;
                Err(e)
            }
        }
    }

    fn recalibrate(&mut self, strat: CalibrationStrategy) -> Result<(), TopologyAwareError> {
        // unset readiness status.
        // Call calibration with core pairs with a single core pair
        // Use results \o/ (or error out)

        self.addresses.clear();

        let core_pairs = vec![(self.main_core, self.helper_core)];

        let mut res = match &self.thresholds {
            ThresholdStrat::ASVP(thresholds_asvp) => {
                let pages: HashSet<&[u8]> = thresholds_asvp
                    .keys()
                    .map(|sp: &SP| unsafe {
                        &*slice_from_raw_parts(sp.page as *const u8, PAGE_LEN)
                    })
                    .collect();
                Self::calibration_for_core_pairs(
                    &self.t,
                    core_pairs.into_iter(),
                    pages.into_iter(),
                    strat,
                )?
            }
            ThresholdStrat::AV(_) | ThresholdStrat::AVSockets(_) => {
                let m = MMappedMemory::new(PAGE_LEN, false, false, |i| i as u8);
                let array: &[u8] = m.slice();
                let mut hashset = HashSet::new();
                hashset.insert(array);
                let pages = hashset;
                Self::calibration_for_core_pairs(
                    &self.t,
                    core_pairs.into_iter(),
                    pages.into_iter(),
                    strat,
                )?
            }
        }; // TODO add the ability to switch calibration strategy.
           // This allows selecting core using non SP strategies, and switching to better strategies later on.

        self.calibration_strategy = strat;
        self.thresholds = res
            .remove(&AV {
                attacker: self.main_core,
                victim: self.helper_core,
            })
            .unwrap()
            .1;
        self.calibration_epoch += 1;
        Ok(())
    }

    unsafe fn test_one_impl(
        &self,
        handle: &mut TopologyAwareTimingChannelHandle,
        reset: bool,
    ) -> Result<CacheStatus, SideChannelError> {
        if handle.calibration_epoch != self.calibration_epoch {
            return Err(SideChannelError::NeedRecalibration);
        }
        let time = unsafe { self.t.attack(handle.addr) };
        if T::NEED_RESET && reset {
            unsafe { flush(handle.addr) };
        }
        if handle.threshold.is_hit(time) {
            Ok(CacheStatus::Hit)
        } else {
            Ok(CacheStatus::Miss)
        }
    }

    unsafe fn test_impl(
        &self,
        addresses: &mut Vec<&mut TopologyAwareTimingChannelHandle>,
        limit: u32,
        reset: bool,
    ) -> Result<Vec<(*const u8, CacheStatus)>, SideChannelError> {
        let mut result = Vec::new();
        let mut tmp = Vec::new();
        let mut i = 0;
        for addr in addresses {
            let r = unsafe { self.test_one_impl(addr, false) };
            tmp.push((addr.to_const_u8_pointer(), r));
            i += 1;
            if i == limit {
                break;
            }
        }
        for (addr, r) in tmp {
            match r {
                Ok(status) => {
                    result.push((addr, status));
                }
                Err(e) => {
                    return Err(e);
                }
            }
            if T::NEED_RESET && reset {
                unsafe { flush(addr) };
            }
        }
        Ok(result)
    }

    unsafe fn prepare_one_impl(
        &self,
        handle: &mut TopologyAwareTimingChannelHandle,
    ) -> Result<(), SideChannelError> {
        if handle.calibration_epoch != self.calibration_epoch {
            return Err(SideChannelError::NeedRecalibration);
        }
        unsafe { flush(handle.addr) };
        handle.ready = true;
        Ok(())
    }

    unsafe fn prepare_impl(
        &mut self,
        addresses: &mut Vec<&mut TopologyAwareTimingChannelHandle>,
        limit: u32,
    ) -> Result<(), SideChannelError> {
        // Iterate on addresse prparig them, error early exit
        let mut i = 0;
        for handle in addresses {
            match unsafe { self.prepare_one_impl(handle) } {
                Ok(_) => {}
                Err(e) => {
                    return Err(e);
                }
            }
            i += 1;
            if i == limit {
                break;
            }
        }
        Ok(())
    }
}

impl<T: TimingChannelPrimitives> Debug for TopologyAwareTimingChannel<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("Topology Aware Channel")
            .field("thresholds", &self.thresholds)
            .field("addresses", &self.addresses)
            .field("slicing", &self.slicing)
            .field("main_core", &self.main_core)
            .field("helper_core", &self.helper_core)
            .field("preferred_addresses", &self.preferred_address)
            .field("calibration_epoch", &self.calibration_epoch)
            .field("primitive", &self.t)
            .finish()
    }
}

impl<T: TimingChannelPrimitives> CoreSpec for TopologyAwareTimingChannel<T> {
    fn main_core(&self) -> CpuSet {
        let mut main = CpuSet::new();
        main.set(self.main_core).unwrap();
        main
    }

    fn helper_core(&self) -> CpuSet {
        let mut helper = CpuSet::new();
        helper.set(self.helper_core).unwrap();
        helper
    }
}

impl<T: TimingChannelPrimitives> MultipleAddrCacheSideChannel for TopologyAwareTimingChannel<T> {
    type Handle = TopologyAwareTimingChannelHandle;
    const MAX_ADDR: u32 = 0;

    unsafe fn test<'a>(
        &mut self,
        addresses: &mut Vec<&'a mut Self::Handle>,
        reset: bool,
    ) -> Result<Vec<(*const u8, CacheStatus)>, SideChannelError>
    where
        Self::Handle: 'a,
    {
        unsafe { self.test_impl(addresses, Self::MAX_ADDR, reset) }
    }

    unsafe fn prepare<'a>(
        &mut self,
        addresses: &mut Vec<&'a mut Self::Handle>,
    ) -> Result<(), SideChannelError>
    where
        Self::Handle: 'a,
    {
        unsafe { self.prepare_impl(addresses, Self::MAX_ADDR) }
    }

    fn victim(&mut self, operation: &dyn Fn()) {
        operation(); // TODO use a different helper core ?
    }

    // this function tolerates multiple handle on the same cache line
    // should the invariant be fixed to one handle per line & calibration epoch ?
    unsafe fn calibrate(
        &mut self,
        addresses: impl IntoIterator<Item = *const u8> + Clone,
    ) -> Result<Vec<Self::Handle>, ChannelFatalError> {
        let core_pair = vec![(self.main_core, self.helper_core)];

        match &mut self.thresholds {
            ThresholdStrat::ASVP(thresholds_asvp) => {
                let pages = addresses
                    .clone()
                    .into_iter()
                    .map(|addr: *const u8| unsafe {
                        let p = get_vpn(addr) as *const u8;
                        let ret = &*slice_from_raw_parts(p, PAGE_LEN);
                        (p, ret)
                    })
                    .collect::<HashMap<*const u8, &[u8]>>();
                let mut res = match Self::calibration_for_core_pairs(
                    &self.t,
                    core_pair.into_iter(),
                    pages.into_iter().map(|(k, v)| v),
                    self.calibration_strategy,
                ) {
                    Err(e) => {
                        return Err(ChannelFatalError::Oops);
                    }
                    Ok(r) => r,
                };

                assert_eq!(res.keys().count(), 1);

                let t = match res
                    .remove(&AV {
                        attacker: self.main_core,
                        victim: self.helper_core,
                    })
                    .unwrap()
                    .1
                {
                    ThresholdStrat::ASVP(t) => t,
                    _ => {
                        unreachable!()
                    }
                };

                for (sp, threshold) in t {
                    thresholds_asvp.insert(sp, threshold);
                }
            }
            ThresholdStrat::AV(_) | ThresholdStrat::AVSockets(_) => {}
        }
        let mut result = vec![];
        for addr in addresses {
            let vpn = get_vpn(addr);
            let slice = self.slicing.hash(addr as usize);
            let handle = TopologyAwareTimingChannelHandle {
                threshold: match &self.thresholds {
                    ThresholdStrat::ASVP(thresholds_asvp) => {
                        thresholds_asvp
                            .get(&SP { slice, page: vpn })
                            .unwrap()
                            .threshold
                    }
                    ThresholdStrat::AV(te) | ThresholdStrat::AVSockets(te) => te.threshold,
                },
                vpn,
                addr,
                ready: false,
                calibration_epoch: self.calibration_epoch,
            };
            result.push(handle);
        }

        Ok(result)
    }
}

impl<T: TimingChannelPrimitives> SingleAddrCacheSideChannel for TopologyAwareTimingChannel<T> {
    type Handle = TopologyAwareTimingChannelHandle;

    unsafe fn test_single(
        &mut self,
        handle: &mut Self::Handle,
        reset: bool,
    ) -> Result<CacheStatus, SideChannelError> {
        unsafe { self.test_one_impl(handle, reset) }
    }

    unsafe fn prepare_single(&mut self, handle: &mut Self::Handle) -> Result<(), SideChannelError> {
        unsafe { self.prepare_one_impl(handle) }
    }

    fn victim_single(&mut self, operation: &dyn Fn()) {
        self.victim(operation)
    }

    unsafe fn calibrate_single(
        &mut self,
        addresses: impl IntoIterator<Item = *const u8> + Clone,
    ) -> Result<Vec<Self::Handle>, ChannelFatalError> {
        unsafe { self.calibrate(addresses) }
    }
}

impl<T: TimingChannelPrimitives> CovertChannel for TopologyAwareTimingChannel<T> {
    type CovertChannelHandle = CovertChannelHandle<TopologyAwareTimingChannel<T>>;
    const BIT_PER_PAGE: usize = 1;

    unsafe fn transmit<'a>(
        &self,
        handle: &mut Self::CovertChannelHandle,
        bits: &mut BitIterator<'a>,
    ) {
        let page = handle.0.addr;

        if let Some(b) = bits.next() {
            if b {
                unsafe { only_reload(page) };
            } else {
                unsafe { only_flush(page) };
            }
        }
    }

    unsafe fn receive(&self, handle: &mut Self::CovertChannelHandle) -> Vec<bool> {
        let r = unsafe { self.test_one_impl(&mut handle.0, false) }; // transmit does the reload / flush as needed.
        match r {
            Err(e) => panic!("{:?}", e),
            Ok(status) => {
                let received = status == CacheStatus::Hit;
                return vec![received];
            }
        }
    }

    unsafe fn ready_page(&mut self, page: *const u8) -> Result<Self::CovertChannelHandle, ()> {
        let vpn: VPN = get_vpn(page);
        // Check if the page has already been readied. If so should error out ?
        if let Some(preferred) = self.preferred_address.get(&vpn) {
            return Err(());
        }

        let (best_slice, te) = match &mut self.thresholds {
            ThresholdStrat::ASVP(thresholds_asvp) => {
                if thresholds_asvp.iter().filter(|kv| kv.0.page == vpn).count() == 0 {
                    // ensure calibration
                    let core_pair = vec![(self.main_core, self.helper_core)];

                    let as_slice = unsafe { &*slice_from_raw_parts(vpn as *const u8, PAGE_LEN) };
                    let pages = vec![as_slice];
                    let mut res = match Self::calibration_for_core_pairs(
                        &self.t,
                        core_pair.into_iter(),
                        pages.into_iter(),
                        self.calibration_strategy,
                    ) {
                        Err(e) => {
                            return Err(());
                        }
                        Ok(r) => r,
                    };
                    assert_eq!(res.keys().count(), 1);

                    let t = match res
                        .remove(&AV {
                            attacker: self.main_core,
                            victim: self.helper_core,
                        })
                        .unwrap()
                        .1
                    {
                        ThresholdStrat::ASVP(t) => t,
                        _ => {
                            unreachable!()
                        }
                    };

                    for (sp, threshold) in t {
                        thresholds_asvp.insert(sp, threshold);
                    }
                }
                let mut best_slice = 0;
                let mut te = None;
                //self.thresholds = ThresholdStrat::ASVP(thresholds_asvp);
                let mut best_error_rate = 1.0;
                for (sp, threshold_error) in thresholds_asvp.iter().filter(|kv| kv.0.page == vpn) {
                    if threshold_error.error.error_rate() < best_error_rate {
                        best_error_rate = threshold_error.error.error_rate();
                        best_slice = sp.slice;
                        te = Some(threshold_error)
                    }
                }
                (best_slice, te.unwrap().clone())
            }
            ThresholdStrat::AV(te) | ThresholdStrat::AVSockets(te) => {
                let te = te.clone();
                (self.get_slice(page), te)
            }
        };

        for i in 0..PAGE_LEN {
            let addr = unsafe { page.offset(i as isize) };
            if self.get_slice(addr) == best_slice {
                self.preferred_address.insert(vpn, addr);
                // Create the right handle
                let mut handle = Self::CovertChannelHandle {
                    0: TopologyAwareTimingChannelHandle {
                        threshold: te.threshold,
                        vpn,
                        addr,
                        ready: false,
                        calibration_epoch: self.calibration_epoch,
                    },
                };
                let r = unsafe { self.prepare_one_impl(&mut handle.0) }.unwrap();

                return Ok(handle);
            }
        }

        Err(())
    }
}

impl<T: TimingChannelPrimitives> TableCacheSideChannel<TopologyAwareTimingChannelHandle>
    for TopologyAwareTimingChannel<T>
{
    unsafe fn tcalibrate(
        &mut self,
        addresses: impl IntoIterator<Item = *const u8> + Clone,
    ) -> Result<Vec<TopologyAwareTimingChannelHandle>, ChannelFatalError> {
        unsafe { self.tcalibrate_multi(addresses) }
    }

    unsafe fn attack<'a, 'b, 'c, 'd>(
        &'a mut self,
        addresses: &'b mut Vec<&'c mut TopologyAwareTimingChannelHandle>,
        victim: &'d dyn Fn(),
        num_iteration: u32,
    ) -> Result<Vec<TableAttackResult>, ChannelFatalError>
    where
        TopologyAwareTimingChannelHandle: 'c,
    {
        unsafe { self.attack_multi(addresses, victim, num_iteration) }
    }
}

// Extra helper for single address per page variants.
#[derive(Debug)]
pub struct SingleChannel<T: SingleAddrCacheSideChannel> {
    inner: T,
}

impl<T: SingleAddrCacheSideChannel> SingleChannel<T> {
    pub fn new(inner: T) -> Self {
        Self { inner }
    }
}

impl<T: SingleAddrCacheSideChannel> CoreSpec for SingleChannel<T> {
    fn main_core(&self) -> CpuSet {
        self.inner.main_core()
    }

    fn helper_core(&self) -> CpuSet {
        self.inner.helper_core()
    }
}

impl<T: SingleAddrCacheSideChannel> SingleAddrCacheSideChannel for SingleChannel<T> {
    type Handle = T::Handle;

    unsafe fn test_single(
        &mut self,
        handle: &mut Self::Handle,
        reset: bool,
    ) -> Result<CacheStatus, SideChannelError> {
        unsafe { self.inner.test_single(handle, reset) }
    }

    unsafe fn prepare_single(&mut self, handle: &mut Self::Handle) -> Result<(), SideChannelError> {
        unsafe { self.inner.prepare_single(handle) }
    }

    fn victim_single(&mut self, operation: &dyn Fn()) {
        self.inner.victim_single(operation)
    }

    unsafe fn calibrate_single(
        &mut self,
        addresses: impl IntoIterator<Item = *const u8> + Clone,
    ) -> Result<Vec<Self::Handle>, ChannelFatalError> {
        unsafe { self.inner.calibrate_single(addresses) }
    }
}

impl<T: SingleAddrCacheSideChannel>
    TableCacheSideChannel<<SingleChannel<T> as SingleAddrCacheSideChannel>::Handle>
    for SingleChannel<T>
{
    unsafe fn tcalibrate(
        &mut self,
        addresses: impl IntoIterator<Item = *const u8> + Clone,
    ) -> Result<Vec<<SingleChannel<T> as SingleAddrCacheSideChannel>::Handle>, ChannelFatalError>
    {
        unsafe { self.inner.tcalibrate_single(addresses) }
    }

    unsafe fn attack<'a, 'b, 'c, 'd>(
        &'a mut self,
        addresses: &'b mut Vec<&'c mut <SingleChannel<T> as SingleAddrCacheSideChannel>::Handle>,
        victim: &'d dyn Fn(),
        num_iteration: u32,
    ) -> Result<Vec<TableAttackResult>, ChannelFatalError>
    where
        <SingleChannel<T> as SingleAddrCacheSideChannel>::Handle: 'c,
    {
        unsafe { self.inner.attack_single(addresses, victim, num_iteration) }
    }
}

/*
impl<T: MultipleAddrCacheSideChannel + Sync + Send> CovertChannel for SingleChannel<T> {
    type Handle = CovertChannelHandle<T>;
    const BIT_PER_PAGE: usize = 1;

    unsafe fn transmit<'a>(&self, handle: &mut Self::Handle, bits: &mut BitIterator<'a>) {
        unimplemented!()
    }

    unsafe fn receive(&self, handle: &mut Self::Handle) -> Vec<bool> {
        let r = unsafe { self.test_single(handle) };
        match r {
            Err(e) => panic!("{:?}", e),
            Ok(status_vec) => {
                assert_eq!(status_vec.len(), 1);
                let received = status_vec[0].1 == Hit;
                //println!("Received {} on page {:p}", received, page);
                return vec![received];
            }
        }
    }

    unsafe fn ready_page(&mut self, page: *const u8) -> Self::Handle {
        unimplemented!()
    }
}
*/

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
