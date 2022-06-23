#![feature(specialization)]
#![feature(unsafe_block_in_unsafe_fn)]
#![deny(unsafe_op_in_unsafe_fn)]

use bit_field::BitField;
use nix::sched::{sched_getaffinity, sched_setaffinity, CpuSet};
use nix::unistd::Pid;
use std::fmt::Debug;

pub mod table_side_channel;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum CacheStatus {
    Hit,
    Miss,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum ChannelFatalError {
    Oops,
}

#[derive(Debug)]
pub enum SideChannelError {
    NeedRecalibration,
    FatalError(ChannelFatalError),
    AddressNotReady(*const u8),
    AddressNotCalibrated(*const u8),
}

pub trait ChannelHandle {
    fn to_const_u8_pointer(&self) -> *const u8;
}

pub trait CoreSpec {
    fn main_core(&self) -> CpuSet;
    fn helper_core(&self) -> CpuSet;
}

pub fn restore_affinity(cpu_set: &CpuSet) {
    sched_setaffinity(Pid::from_raw(0), &cpu_set).unwrap();
}

#[must_use = "This result must be used to restore affinity"]
pub fn set_affinity(cpu_set: &CpuSet) -> Result<CpuSet, nix::Error> {
    let old = sched_getaffinity(Pid::from_raw(0))?;
    sched_setaffinity(Pid::from_raw(0), &cpu_set)?;
    Ok(old)
}

pub trait SingleAddrCacheSideChannel: CoreSpec + Debug {
    type Handle: ChannelHandle;
    //type SingleChannelFatalError: Debug;
    /// # Safety
    ///
    /// addr must be a valid pointer to read.
    unsafe fn test_single(
        &mut self,
        handle: &mut Self::Handle,
        reset: bool,
    ) -> Result<CacheStatus, SideChannelError>;
    /// # Safety
    ///
    /// addr must be a valid pointer to read.
    unsafe fn prepare_single(&mut self, handle: &mut Self::Handle) -> Result<(), SideChannelError>;
    fn victim_single(&mut self, operation: &dyn Fn());
    /// # Safety
    ///
    /// addresses must contain only valid pointers to read.
    unsafe fn calibrate_single(
        &mut self,
        addresses: impl IntoIterator<Item = *const u8> + Clone,
    ) -> Result<Vec<Self::Handle>, ChannelFatalError>;
}

pub trait MultipleAddrCacheSideChannel: CoreSpec + Debug {
    type Handle: ChannelHandle;
    const MAX_ADDR: u32;
    /// # Safety
    ///
    /// addresses must contain only valid pointers to read.
    unsafe fn test<'a, 'b, 'c>(
        &'a mut self,
        addresses: &'b mut Vec<&'c mut Self::Handle>,
        reset: bool,
    ) -> Result<Vec<(*const u8, CacheStatus)>, SideChannelError>
    where
        Self::Handle: 'c;

    /// # Safety
    ///
    /// addresses must contain only valid pointers to read.
    unsafe fn prepare<'a, 'b, 'c>(
        &'a mut self,
        addresses: &'b mut Vec<&'c mut Self::Handle>,
    ) -> Result<(), SideChannelError>
    where
        Self::Handle: 'c;
    fn victim(&mut self, operation: &dyn Fn());

    /// # Safety
    ///
    /// addresses must contain only valid pointers to read.
    unsafe fn calibrate(
        &mut self,
        addresses: impl IntoIterator<Item = *const u8> + Clone,
    ) -> Result<Vec<Self::Handle>, ChannelFatalError>;
}

/*
impl<T: MultipleAddrCacheSideChannel> SingleAddrCacheSideChannel for T {
    type Handle = <Self as MultipleAddrCacheSideChannel>::Handle;

    unsafe fn test_single(
        &mut self,
        handle: &mut Self::Handle,
        reset: bool,
    ) -> Result<CacheStatus, SideChannelError> {
        let mut handles = vec![handle];
        unsafe { self.test(&mut handles, reset) }.map(|v| v[0].1)
    }

    unsafe fn prepare_single(&mut self, handle: &mut Self::Handle) -> Result<(), SideChannelError> {
        let mut handles = vec![handle];
        unsafe { self.prepare(&mut handles) }
    }

    fn victim_single(&mut self, operation: &dyn Fn()) {
        self.victim(operation);
    }

    unsafe fn calibrate_single(
        &mut self,
        addresses: impl IntoIterator<Item = *const u8> + Clone,
    ) -> Result<Vec<Self::Handle>, ChannelFatalError> {
        unsafe { self.calibrate(addresses) }
    }
}
*/
// From covert_channel_evaluation
pub trait CovertChannel: Send + Sync + CoreSpec + Debug {
    type CovertChannelHandle;
    const BIT_PER_PAGE: usize;
    unsafe fn transmit(&self, handle: &mut Self::CovertChannelHandle, bits: &mut BitIterator);
    unsafe fn receive(&self, handle: &mut Self::CovertChannelHandle) -> Vec<bool>;
    unsafe fn ready_page(&mut self, page: *const u8) -> Result<Self::CovertChannelHandle, ()>; // TODO Error Type
}

pub struct BitIterator<'a> {
    bytes: &'a Vec<u8>,
    byte_index: usize,
    bit_index: u8,
}

impl<'a> BitIterator<'a> {
    pub fn new(bytes: &'a Vec<u8>) -> BitIterator<'a> {
        BitIterator {
            bytes,
            byte_index: 0,
            bit_index: 0,
        }
    }

    pub fn atEnd(&self) -> bool {
        self.byte_index >= self.bytes.len()
    }
}

impl Iterator for BitIterator<'_> {
    type Item = bool;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(b) = self.bytes.get(self.byte_index) {
            let r = (b >> (u8::BIT_LENGTH - 1 - self.bit_index as usize)) & 1 != 0;
            self.bit_index += 1;
            self.byte_index += self.bit_index as usize / u8::BIT_LENGTH;
            self.bit_index = self.bit_index % u8::BIT_LENGTH as u8;
            Some(r)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
