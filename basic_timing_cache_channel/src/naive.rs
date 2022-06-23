use crate::TimingChannelPrimitives;
use cache_side_channel::table_side_channel::{
    SingleTableCacheSideChannel, TableAttackResult, TableCacheSideChannel,
};
use cache_side_channel::{
    BitIterator, CacheStatus, ChannelFatalError, ChannelHandle, CoreSpec, CovertChannel,
    MultipleAddrCacheSideChannel, SideChannelError, SingleAddrCacheSideChannel,
};
use cache_utils::calibration::{get_vpn, only_flush, only_reload, HashMap, Threshold, VPN};
use cache_utils::flush;
use nix::sched::sched_getaffinity;
use nix::sched::CpuSet;
use nix::unistd::Pid;
use std::fmt::Debug;

// Parameters required : The threshold.
#[derive(Debug)]
pub struct NaiveTimingChannel<T: TimingChannelPrimitives> {
    threshold: Threshold,
    current: HashMap<VPN, *const u8>,
    main_core: CpuSet,
    helper_core: CpuSet,
    channel_primitive: T,
}

pub struct NaiveTimingChannelHandle {
    vpn: VPN, // what is this field used for
    addr: *const u8,
}

impl ChannelHandle for NaiveTimingChannelHandle {
    fn to_const_u8_pointer(&self) -> *const u8 {
        self.addr
    }
}

unsafe impl<T: TimingChannelPrimitives + Send> Send for NaiveTimingChannel<T> {}
unsafe impl<T: TimingChannelPrimitives + Sync> Sync for NaiveTimingChannel<T> {}

impl<T: TimingChannelPrimitives> NaiveTimingChannel<T> {
    pub fn new(threshold: Threshold) -> Self {
        Self {
            threshold,
            current: Default::default(),
            main_core: sched_getaffinity(Pid::from_raw(0)).unwrap(),
            helper_core: sched_getaffinity(Pid::from_raw(0)).unwrap(),
            channel_primitive: Default::default(),
        }
    }

    pub fn set_cores(&mut self, main_core: usize, helper_core: usize) {
        self.main_core = CpuSet::new();
        self.main_core.set(main_core).unwrap();

        self.helper_core = CpuSet::new();
        self.helper_core.set(helper_core).unwrap();
    }

    pub fn unready_page(
        &mut self,
        handle: NaiveTimingChannelHandle,
    ) -> Result<*const u8, ChannelFatalError> {
        //if let Some(addr) = self.current.remove(&handle.vpn) {
        Ok(handle.addr)
        //} else {
        //    Err(ChannelFatalError::Oops)
        //}
    }

    unsafe fn test_one_impl(
        &self,
        handle: &mut NaiveTimingChannelHandle,
        //limit: u32,
        reset: bool,
    ) -> Result<CacheStatus, SideChannelError> {
        // This should be handled in prepare / unprepare
        let t = unsafe { self.channel_primitive.attack(handle.addr) };
        if T::NEED_RESET && reset {
            unsafe { flush(handle.addr) };
        }
        if self.threshold.is_hit(t) {
            Ok(CacheStatus::Hit)
        } else {
            Ok(CacheStatus::Miss)
        }
    }

    unsafe fn test_impl(
        &self,
        handles: &mut Vec<&mut NaiveTimingChannelHandle>,
        limit: u32,
        reset: bool,
    ) -> Result<Vec<(*const u8, CacheStatus)>, SideChannelError> {
        let mut result = Vec::new();
        let mut tmp = Vec::new();
        let mut i = 0;
        for addr in handles {
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

    // The former invariant of one handle per page has been removed
    // Now tolerates as many handles per cache line as wanted
    // should the invariant be fixed into one handle per cache line ?

    unsafe fn calibrate_impl(
        &mut self,
        addr: *const u8,
    ) -> Result<NaiveTimingChannelHandle, ChannelFatalError> {
        let vpn = get_vpn(addr);
        /*if self.current.get(&vpn).is_some() {
            return Err(ChannelFatalError::Oops);
        } else {
            self.current.insert(vpn, addr);*/
        Ok(NaiveTimingChannelHandle { vpn, addr })
        //}
    }

    unsafe fn prepare_one_impl(
        &mut self,
        handle: &mut NaiveTimingChannelHandle,
    ) -> Result<(), SideChannelError> {
        unsafe { flush(handle.addr) };
        Ok(())
    }

    unsafe fn prepare_impl(
        &mut self,
        addresses: &mut Vec<&mut NaiveTimingChannelHandle>,
        limit: u32,
    ) -> Result<(), SideChannelError> {
        // Iterate on addresses preparing them, error early exit
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

    pub unsafe fn test_debug(
        &self,
        handle: &mut NaiveTimingChannelHandle,
        reset: bool,
    ) -> Result<(CacheStatus, u64), SideChannelError> {
        // This should be handled in prepare / unprepare
        let t = unsafe { self.channel_primitive.attack(handle.addr) };
        if T::NEED_RESET && reset {
            unsafe { flush(handle.addr) };
        }
        if self.threshold.is_hit(t) {
            Ok((CacheStatus::Hit, t))
        } else {
            Ok((CacheStatus::Miss, t))
        }
    }
}

impl<T: TimingChannelPrimitives> CoreSpec for NaiveTimingChannel<T> {
    fn main_core(&self) -> CpuSet {
        self.main_core
    }

    fn helper_core(&self) -> CpuSet {
        self.helper_core
    }
}

impl<T: TimingChannelPrimitives + Send + Sync> CovertChannel for NaiveTimingChannel<T> {
    type CovertChannelHandle = NaiveTimingChannelHandle;
    const BIT_PER_PAGE: usize = 1;

    unsafe fn transmit<'a>(
        &self,
        handle: &mut Self::CovertChannelHandle,
        bits: &mut BitIterator<'a>,
    ) {
        if let Some(b) = bits.next() {
            if b {
                unsafe { only_reload(handle.addr) };
            } else {
                unsafe { only_flush(handle.addr) };
            }
        }
    }

    unsafe fn receive(&self, handle: &mut Self::CovertChannelHandle) -> Vec<bool> {
        let r = unsafe { self.test_one_impl(handle, false) };
        match r {
            Err(e) => panic!(),
            Ok(status) => match status {
                CacheStatus::Hit => vec![true],
                CacheStatus::Miss => vec![false],
            },
        }
    }

    unsafe fn ready_page(&mut self, page: *const u8) -> Result<Self::CovertChannelHandle, ()> {
        unsafe { self.calibrate_impl(page) }.map_err(|_| ())
    }
}

impl<T: TimingChannelPrimitives> SingleAddrCacheSideChannel for NaiveTimingChannel<T> {
    type Handle = NaiveTimingChannelHandle;

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
        operation()
    }

    /// # Safety
    ///
    /// addr needs to be a valid pointer
    unsafe fn calibrate_single(
        &mut self,
        addresses: impl IntoIterator<Item = *const u8> + Clone,
    ) -> Result<Vec<Self::Handle>, ChannelFatalError> {
        let mut result = vec![];
        for addr in addresses {
            match unsafe { self.calibrate_impl(addr) } {
                Ok(handle) => result.push(handle),
                Err(e) => {
                    return Err(e);
                }
            }
        }
        Ok(result)
    }
}

impl<T: TimingChannelPrimitives> MultipleAddrCacheSideChannel for NaiveTimingChannel<T> {
    type Handle = NaiveTimingChannelHandle;
    const MAX_ADDR: u32 = 1;

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
        operation()
    }

    unsafe fn calibrate(
        &mut self,
        addresses: impl IntoIterator<Item = *const u8> + Clone,
    ) -> Result<Vec<Self::Handle>, ChannelFatalError> {
        let mut result = vec![];
        for addr in addresses {
            match unsafe { self.calibrate_impl(addr) } {
                Ok(handle) => result.push(handle),
                Err(e) => {
                    return Err(e);
                }
            }
        }
        Ok(result)
    }
}

impl<T: TimingChannelPrimitives> TableCacheSideChannel<NaiveTimingChannelHandle>
    for NaiveTimingChannel<T>
{
    unsafe fn tcalibrate(
        &mut self,
        addresses: impl IntoIterator<Item = *const u8> + Clone,
    ) -> Result<Vec<NaiveTimingChannelHandle>, ChannelFatalError> {
        unsafe { self.tcalibrate_single(addresses) }
    }

    unsafe fn attack<'a, 'b, 'c, 'd>(
        &'a mut self,
        addresses: &'b mut Vec<&'c mut NaiveTimingChannelHandle>,
        victim: &'d dyn Fn(),
        num_iteration: u32,
    ) -> Result<Vec<TableAttackResult>, ChannelFatalError>
    where
        NaiveTimingChannelHandle: 'c,
    {
        unsafe { self.attack_single(addresses, victim, num_iteration) }
    }
}
// Include a helper code to get global threshold model ?

// TODO
