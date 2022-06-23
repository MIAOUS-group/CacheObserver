use cache_utils::ip_tool::{Function, TIMED_MACCESS};
use cache_utils::{flush, maccess};
use prefetcher_reverse::{pattern_helper, Prober, PAGE_CACHELINE_LEN};
use std::arch::x86_64 as arch_x86;

pub const NUM_ITERATION: usize = 1 << 10;

unsafe extern "C" fn prefetch_l2(p: *const u8) -> u64 {
    maccess(p);
    arch_x86::_mm_mfence();
    flush(p);
    arch_x86::_mm_mfence();
    arch_x86::_mm_prefetch::<{ arch_x86::_MM_HINT_T1 }>(p as *const i8);
    arch_x86::__cpuid_count(0, 0);
    0
}

unsafe extern "C" fn prefetch_l3(p: *const u8) -> u64 {
    maccess(p);
    arch_x86::_mm_mfence();
    flush(p);
    arch_x86::_mm_mfence();
    arch_x86::_mm_prefetch::<{ arch_x86::_MM_HINT_T2 }>(p as *const i8);
    arch_x86::__cpuid_count(0, 0);
    0
}

unsafe extern "C" fn prefetch_l1(p: *const u8) -> u64 {
    maccess(p);
    arch_x86::_mm_mfence();
    flush(p);
    arch_x86::_mm_mfence();
    arch_x86::_mm_prefetch::<{ arch_x86::_MM_HINT_T0 }>(p as *const i8);
    arch_x86::__cpuid_count(0, 0);
    0
}

fn exp(stride: usize, num_steps: i32, delay: u64, reload: &Function) {
    let mut prober = Prober::<2>::new(63).unwrap();
    prober.set_delay(delay);
    let limit = if num_steps < 0 {
        PAGE_CACHELINE_LEN + stride + 2
    } else {
        stride * num_steps as usize
    };
    let pattern = (2usize..limit).step_by(stride).collect::<Vec<_>>();
    let p = pattern_helper(&pattern, reload);

    let pl2 = Function {
        fun: prefetch_l2,
        ip: prefetch_l2 as *const u8,
        end: prefetch_l2 as *const u8,
        size: 0,
    };

    let pl3 = Function {
        fun: prefetch_l3,
        ip: prefetch_l3 as *const u8,
        end: prefetch_l3 as *const u8,
        size: 0,
    };

    let pl1 = Function {
        fun: prefetch_l1,
        ip: prefetch_l1 as *const u8,
        end: prefetch_l1 as *const u8,
        size: 0,
    };

    let mut pattern_pl2 = pattern_helper(&(0..(2 * PAGE_CACHELINE_LEN)).collect(), &pl2);
    pattern_pl2.extend(p.iter().cloned());

    let mut pattern_pl3 = pattern_helper(&(0..(2 * PAGE_CACHELINE_LEN)).collect(), &pl3);
    pattern_pl3.extend(p.iter().cloned());

    let mut pattern_pl1 = pattern_helper(&(0..(2 * PAGE_CACHELINE_LEN)).collect(), &pl1);
    pattern_pl1.extend(p.iter().cloned());

    println!("With no sw prefetch");
    let result = prober.full_page_probe(p, NUM_ITERATION as u32, 100);
    println!("{}", result);
    println!("With L2 sw prefetch");
    let result = prober.full_page_probe(pattern_pl2, NUM_ITERATION as u32, 100);
    println!("{}", result);
    println!("With L3 sw prefetch");
    let result = prober.full_page_probe(pattern_pl3, NUM_ITERATION as u32, 100);
    println!("{}", result);
    println!("With L1 sw prefetch");
    let result = prober.full_page_probe(pattern_pl1, NUM_ITERATION as u32, 100);
    println!("{}", result);
}

fn main() {
    //let reload = Function::try_new(1, 0, TIMED_MACCESS).unwrap();
    let mut reloads = Vec::new();
    for i in 0..3 {
        reloads.push(Function::try_new(4, i, TIMED_MACCESS).unwrap());
    }
    for (index, stride) in [2, 3, 4].iter().enumerate() {
        let reload = &reloads[index];
        for delay_shift in [5, 12] {
            let limit = ((PAGE_CACHELINE_LEN + 32) / stride) as i32;
            //for num_steps in -1..limit {
            let num_steps = limit;
            println!(
                "Stride: {}, Limit: {}, Delay: {}",
                *stride,
                num_steps,
                1 << delay_shift
            );
            exp(*stride, num_steps, 1 << delay_shift, &reload);
            //}
        }
    }
}
