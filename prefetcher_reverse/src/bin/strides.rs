use cache_utils::ip_tool::{Function, TIMED_MACCESS};
use prefetcher_reverse::{pattern_helper, Prober, PAGE_CACHELINE_LEN};

pub const NUM_ITERATION: usize = 1 << 10;

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

    let result = prober.full_page_probe(p, NUM_ITERATION as u32, 100);
    println!("{}", result);
}

fn main() {
    let reload = Function::try_new(1, 0, TIMED_MACCESS).unwrap();

    for stride in [5, 7, 8] {
        for delay_shift in [5, 12, 20] {
            //let stride = 8;
            let limit = (PAGE_CACHELINE_LEN / stride) as i32 + 2;
            //for num_steps in -1..limit {
            let num_steps = limit;
            println!(
                "Stride: {}, Limit: {}, Delay: {}",
                stride,
                num_steps,
                1 << delay_shift
            );
            exp(stride, num_steps, 1 << delay_shift, &reload);
            //}
        }
    }
}
