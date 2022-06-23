#![feature(ptr_internals)]
#![feature(linked_list_cursors)]
#![feature(global_asm)]
#![allow(clippy::missing_safety_doc)]
#![deny(unsafe_op_in_unsafe_fn)]

use core::arch::x86_64 as arch_x86;
use core::ptr;

pub mod cache_info;
mod calibrate_2t;
pub mod calibration;
pub mod complex_addressing;
pub mod frequency;
pub mod ip_tool;
pub mod mmap;
pub mod prefetcher;

// rdtsc no fence
pub unsafe fn rdtsc_nofence() -> u64 {
    unsafe { arch_x86::_rdtsc() }
}
// rdtsc (has mfence before and after)
pub unsafe fn rdtsc_fence() -> u64 {
    unsafe { arch_x86::_mm_mfence() };
    let tsc: u64 = unsafe { arch_x86::_rdtsc() };
    unsafe { arch_x86::_mm_mfence() };
    tsc
}

pub unsafe fn maccess<T>(p: *const T) {
    unsafe { ptr::read_volatile(p) };
}

// flush (cflush)
pub unsafe fn flush(p: *const u8) {
    unsafe { arch_x86::_mm_clflush(p) };
}

pub fn noop<T>(_: *const T) {}

pub fn find_core_per_socket() -> u8 {
    // FIXME error handling
    use std::process::Command;
    use std::str::from_utf8;

    let core_per_socket_out = Command::new("sh")
        .arg("-c")
        .arg("lscpu | grep socket | cut -b 22-")
        .output()
        .expect("Failed to detect cpu count");
    //println!("{:#?}", core_per_socket_str);

    let core_per_socket_str = from_utf8(&core_per_socket_out.stdout).unwrap();

    //println!("Number of cores per socket: {}", cps_str);

    let core_per_socket: u8 = core_per_socket_str[0..(core_per_socket_str.len() - 1)] // FIXME, for cases such as '   24  '
        .parse()
        .unwrap_or(0);
    core_per_socket
}

// future enhancements
// prefetch
// long nop (64 nops)
