#![deny(unsafe_op_in_unsafe_fn)]
use basic_timing_cache_channel::TopologyAwareError;
use cache_side_channel::CacheStatus::Hit;
use cache_side_channel::{
    set_affinity, ChannelHandle, CoreSpec, MultipleAddrCacheSideChannel, SingleAddrCacheSideChannel,
};
use cache_utils::calibration::PAGE_LEN;
use cache_utils::ip_tool::{Function, TIMED_MACCESS};
use cache_utils::maccess;
use cache_utils::mmap;
use cache_utils::mmap::MMappedMemory;
use flush_flush::{FFHandle, FFPrimitives, FlushAndFlush};
use nix::Error;
use prefetcher_reverse::{
    pattern_helper, PatternAccess, Prober, CACHE_LINE_LEN, PAGE_CACHELINE_LEN,
};
use rand::seq::SliceRandom;
use std::iter::Cycle;

pub const NUM_ITERATION: usize = 1 << 10;
pub const NUM_PAGES: usize = 256;

// TODO negative stride
fn generate_pattern(offset: usize, len: usize, stride: isize) -> Option<Vec<usize>> {
    let end = (offset as isize + stride * len as isize) * CACHE_LINE_LEN as isize;
    if end < 0 || end > PAGE_LEN as isize {
        return None;
    }
    let mut res = Vec::with_capacity(len);
    let mut addr = offset as isize;
    for _ in 0..len {
        res.push(addr as usize);
        addr += stride;
    }
    Some(res)
}

fn execute_pattern(
    channel: &mut FlushAndFlush,
    page_handles: &mut Vec<&mut <FlushAndFlush as MultipleAddrCacheSideChannel>::Handle>,
    pattern: &Vec<usize>,
) -> Vec<bool> {
    for offset in pattern {
        let pointer = page_handles[*offset].to_const_u8_pointer();
        unsafe { maccess(pointer) };
    }

    let measures = unsafe { channel.test(page_handles, true) };

    let mut res = vec![false; PAGE_CACHELINE_LEN];

    for (i, status) in measures.unwrap().into_iter().enumerate() {
        res[i] = status.1 == Hit;
    }
    res
}

fn execute_pattern_probe1(
    channel: &mut FlushAndFlush,
    page_handles: &mut Vec<&mut <FlushAndFlush as MultipleAddrCacheSideChannel>::Handle>,
    pattern: &Vec<usize>,
    probe_offset: usize,
) -> bool {
    for offset in pattern {
        let pointer = page_handles[*offset].to_const_u8_pointer();
        unsafe { maccess(pointer) };
    }

    let measure = unsafe { channel.test_single(&mut page_handles[probe_offset], true) };

    measure.unwrap() == Hit
}

fn main() {
    /*
    let mut vec = Vec::new();
    let mut handles = Vec::new();
    let (mut channel, cpuset, core) = FlushAndFlush::new_any_single_core().unwrap();
    let old_affinity = set_affinity(&channel.main_core());
    for i in 0..NUM_PAGES {
        let mut p = MMappedMemory::<u8>::new(PAGE_LEN, false);
        for j in 0..PAGE_LEN {
            p[j] = (i * PAGE_CACHELINE_LEN + j) as u8;
        }
        let page_addresses =
            ((0..PAGE_LEN).step_by(CACHE_LINE_LEN)).map(|offset| &p[offset] as *const u8);
        let page_handles = unsafe { channel.calibrate(page_addresses) }.unwrap();
        println!("{:p}", page_handles[0].to_const_u8_pointer());
        vec.push(p);
        handles.push(page_handles);
    }
    println!();

    let mut page_indexes = (0..(handles.len())).cycle();

    handles.shuffle(&mut rand::thread_rng());
    let mut handles_mutref = Vec::new();
    for page in handles.iter_mut() {
        handles_mutref.push(
            page.iter_mut()
                .collect::<Vec<&mut <FlushAndFlush as MultipleAddrCacheSideChannel>::Handle>>(),
        );
    }

    // Use an std::iter::Cycle iterator for pages.

    /*
    TODO List :
    Calibration & core selection (select one or two cores with optimal error)
    Then allocate a bunch of pages, and do accesses on each of them.

    (Let's start with stride patterns: for len in 0..16, and then for stride in 1..maxs_stride(len),
    generate a vec of addresses and get the victim to execute, then dump all the page)

    Sanity check on one pattern : do full dump, vs do dump per address.

    Both can be done using the FlushFlush channel

     */

    let pattern = generate_pattern(1, 4, 4).unwrap();
    println!("{:?}", pattern);
    let mut probe_all_result_first = [0; PAGE_CACHELINE_LEN];
    for _ in 0..NUM_ITERATION {
        let page_index = page_indexes.next().unwrap();
        unsafe { channel.prepare(&mut handles_mutref[page_index]) }.unwrap();
        let res = execute_pattern(&mut channel, &mut handles_mutref[page_index], &pattern);
        for j in 0..PAGE_CACHELINE_LEN {
            if res[j] {
                probe_all_result_first[j] += 1;
            }
        }
    }
    let mut probe1_result = [0; PAGE_CACHELINE_LEN];
    for i in 0..PAGE_CACHELINE_LEN {
        for _ in 0..NUM_ITERATION {
            let page_index = page_indexes.next().unwrap();
            unsafe { channel.prepare(&mut handles_mutref[page_index]) }.unwrap();
            let res =
                execute_pattern_probe1(&mut channel, &mut handles_mutref[page_index], &pattern, i);
            if res {
                probe1_result[i] += 1;
            }
        }
    }
    let mut probe_all_result = [0; PAGE_CACHELINE_LEN];
    for _ in 0..NUM_ITERATION {
        let page_index = page_indexes.next().unwrap();
        unsafe { channel.prepare(&mut handles_mutref[page_index]) }.unwrap();
        let res = execute_pattern(&mut channel, &mut handles_mutref[page_index], &pattern);
        for j in 0..PAGE_CACHELINE_LEN {
            if res[j] {
                probe_all_result[j] += 1;
            }
        }
    }

    for i in 0..PAGE_CACHELINE_LEN {
        println!(
            "{:2} {:4} {:4} {:4}",
            i, probe_all_result_first[i], probe1_result[i], probe_all_result[i]
        );
    }

    let pattern = generate_pattern(0, 3, 12).unwrap();
    println!("{:?}", pattern);
    let mut probe_all_result_first = [0; PAGE_CACHELINE_LEN];
    for _ in 0..NUM_ITERATION {
        let page_index = page_indexes.next().unwrap();
        unsafe { channel.prepare(&mut handles_mutref[page_index]) }.unwrap();
        let res = execute_pattern(&mut channel, &mut handles_mutref[page_index], &pattern);
        for j in 0..PAGE_CACHELINE_LEN {
            if res[j] {
                probe_all_result_first[j] += 1;
            }
        }
    }
    let mut probe1_result = [0; PAGE_CACHELINE_LEN];
    for i in 0..PAGE_CACHELINE_LEN {
        for _ in 0..NUM_ITERATION {
            let page_index = page_indexes.next().unwrap();
            unsafe { channel.prepare(&mut handles_mutref[page_index]) }.unwrap();
            let res =
                execute_pattern_probe1(&mut channel, &mut handles_mutref[page_index], &pattern, i);
            if res {
                probe1_result[i] += 1;
            }
        }
    }
    let mut probe_all_result = [0; PAGE_CACHELINE_LEN];
    for _ in 0..NUM_ITERATION {
        let page_index = page_indexes.next().unwrap();
        unsafe { channel.prepare(&mut handles_mutref[page_index]) }.unwrap();
        let res = execute_pattern(&mut channel, &mut handles_mutref[page_index], &pattern);
        for j in 0..PAGE_CACHELINE_LEN {
            if res[j] {
                probe_all_result[j] += 1;
            }
        }
    }*/

    let reload = Function::try_new(1, 0, TIMED_MACCESS).unwrap();

    let pattern = pattern_helper(&generate_pattern(0, 3, 12).unwrap(), &reload);
    let pattern4 = pattern_helper(&generate_pattern(0, 4, 12).unwrap(), &reload);
    let mut new_prober = Prober::<1>::new(63).unwrap();
    let result = new_prober.full_page_probe(pattern.clone(), NUM_ITERATION as u32, 100);
    println!("{}", result);
    //println!("{:#?}", result);

    let result2 = new_prober.full_page_probe(pattern, NUM_ITERATION as u32, 100);
    println!("{}", result2);
    let result4 = new_prober.full_page_probe(pattern4, NUM_ITERATION as u32, 100);
    println!("{}", result4);
    let pattern5 = pattern_helper(&generate_pattern(0, 5, 8).unwrap(), &reload);
    let result5 = new_prober.full_page_probe(pattern5, NUM_ITERATION as u32, 100);
    println!("{}", result5);

    let pattern5 = pattern_helper(&generate_pattern(0, 5, 4).unwrap(), &reload);
    let result5 = new_prober.full_page_probe(pattern5, NUM_ITERATION as u32, 100);
    println!("{}", result5);

    let pattern = pattern_helper(&generate_pattern(0, 10, 4).unwrap(), &reload);
    let result = new_prober.full_page_probe(pattern, NUM_ITERATION as u32, 100);
    println!("{}", result);

    let pattern = pattern_helper(&generate_pattern(0, 6, 8).unwrap(), &reload);
    let result = new_prober.full_page_probe(pattern, NUM_ITERATION as u32, 100);
    println!("{}", result);

    let pattern = pattern_helper(&generate_pattern(2, 6, 0).unwrap(), &reload);
    let result = new_prober.full_page_probe(pattern, NUM_ITERATION as u32, 100);
    println!("{}", result);

    let pattern = pattern_helper(&vec![0, 0, 8, 8, 16, 16, 24, 24], &reload);
    let result = new_prober.full_page_probe(pattern, NUM_ITERATION as u32, 100);
    println!("{}", result);

    /*
    for i in 0..PAGE_CACHELINE_LEN {
        println!(
            "{:2} {:4} {:4} {:4}",
            i, probe_all_result_first[i], probe1_result[i], probe_all_result[i]
        );
    }

    println!("{:?}", generate_pattern(0, 5, 1));
    println!("{:?}", generate_pattern(5, 0, 1));
    println!("{:?}", generate_pattern(1, 5, 5));
    println!("{:?}", generate_pattern(0, 16, 16));
     */
}
