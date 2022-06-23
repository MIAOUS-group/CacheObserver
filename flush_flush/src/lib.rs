#![deny(unsafe_op_in_unsafe_fn)]

pub mod naive;

use basic_timing_cache_channel::{
    SingleChannel, TimingChannelPrimitives, TopologyAwareTimingChannel,
};

use cache_side_channel::MultipleAddrCacheSideChannel;
use cache_utils::calibration::only_flush;

#[derive(Debug, Default)]
pub struct FFPrimitives {}

impl TimingChannelPrimitives for FFPrimitives {
    unsafe fn attack(&self, addr: *const u8) -> u64 {
        unsafe { only_flush(addr) }
    }
    const NEED_RESET: bool = false;
}

pub type FlushAndFlush = TopologyAwareTimingChannel<FFPrimitives>;

pub type FFHandle = <FlushAndFlush as MultipleAddrCacheSideChannel>::Handle;

pub type SingleFlushAndFlush = SingleChannel<FlushAndFlush>;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
