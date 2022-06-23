use cache_utils::ip_tool::{Function, TIMED_MACCESS};
use prefetcher_reverse::{pattern_helper, Prober, PAGE_CACHELINE_LEN};

pub const NUM_ITERATION: usize = 1 << 10;

fn exp(delay: u64, reload: &Function) {
    let mut prober = Prober::<2>::new(63).unwrap();
    prober.set_delay(delay);
    let pattern = (0usize..(PAGE_CACHELINE_LEN * 2usize)).collect::<Vec<usize>>();
    let p = pattern_helper(&pattern, reload);

    let result = prober.full_page_probe(p, NUM_ITERATION as u32, 100);
    println!("{}", result);
}

fn main() {
    let reload = Function::try_new(1, 0, TIMED_MACCESS).unwrap();

    for delay in [0, 5, 10, 50] {
        println!("Delay after each access: {} us", delay);
        exp(delay, &reload);
    }
}
