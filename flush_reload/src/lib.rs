#![deny(unsafe_op_in_unsafe_fn)]

pub mod naive;

use basic_timing_cache_channel::{
    SingleChannel, TimingChannelPrimitives, TopologyAwareTimingChannel,
};

use cache_side_channel::MultipleAddrCacheSideChannel;
use cache_utils::calibration::only_reload;

#[derive(Debug, Default)]
pub struct FRPrimitives {}

impl TimingChannelPrimitives for FRPrimitives {
    unsafe fn attack(&self, addr: *const u8) -> u64 {
        unsafe { only_reload(addr) }
    }
    const NEED_RESET: bool = true;
}

pub type FlushAndReload = TopologyAwareTimingChannel<FRPrimitives>;

pub type FRHandle = <FlushAndReload as MultipleAddrCacheSideChannel>::Handle;

pub type SingleFlushAndReload = SingleChannel<FlushAndReload>;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
