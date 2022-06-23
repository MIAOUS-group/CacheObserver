use crate::FRPrimitives;
use basic_timing_cache_channel::naive::NaiveTimingChannel;
use cache_side_channel::SingleAddrCacheSideChannel;

pub type NaiveFlushAndReload = NaiveTimingChannel<FRPrimitives>;

pub type NFRHandle = <NaiveFlushAndReload as SingleAddrCacheSideChannel>::Handle;
