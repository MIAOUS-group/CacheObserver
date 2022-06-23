CacheObserver - monitor what happens in the cache when doing memory accesses
============================================================================

This framework, derived from https://github.com/MIAOUS-group/calibration-done-right,
is built to help reverse engineer prefetchers on Intel CPUs.

The main entry point of the framework is the `prefetcher_reverse` crate.

The code presented runs under Fedora 30, and can also be made to run on Ubuntu 18.04 LTS with minor tweaks

(Notably, lib cpupower may also be called lib cpufreq)

## Usage

Requires rust nightly features. Install rust nightly using rustup,
known working versions are listed at the end of the document.

This tool needs access to MSR and thus requires sudo access.
The setup.sh script disables turbo boost and makes sure the frequency is set to the max
non-boosted frequency.

One can run all the experiments with the following instructions :

```
cd prefetcher_reverse
mkdir results-xxx
cd results-xxx
sudo ../setup.sh
../run-msr-all.sh 15
../run-msr-all.sh 14
../run-msr-all.sh 13
../run-msr-all.sh 12
../run-msr-all.sh 0
# Do not forget to re-enable turbo-boost and set the cpupower frequency governor back
```

This results in a set of log files that can then be analyzed.

**Note for default settings, this results in several GB worth of logs**

## General Architecture

`prefetcher_reverse` is where the experiments used to reverse engineer prefetcher lives.
It contains the Prober structure, along with binaries generating patterns for the experiments
to run and feeding them to the Prober struct. 

The `analysis` folder contains the scripts we used to turn the logs into figures.
To be documented. We used Julia with the Plots and PGFPlotsX backend to generate figures.

The flow is to first use `extract_analysis_csv.sh` to extract the CSV for each experiment from the logs.

Then one can use the makeplots Julia scripts (those are unfortunately not optimized and may run for several hours, as the LaTeX backend is not thread-safe and generates many figures).

Those scripts expect to find the CSVs at a specific path and require their output folder
by MSR 420 (0x1A4) values to exist beforehand (so 15,14,13,12,0 must exist beforehand).
They are still quite rough and undocumented, rough edges are to be expected.
(A better version could be released if the paper is accepted)

The resulting figures can then be sorted into subfolders for easier browsing, and the change colormap script can be used to tweak the tikz file colormaps for use in papers

Crates originally from the *Calibration done right* framework, slightly modified :

- `basic_timing_cache_channel` contains generic implementations of Naive and Optimised cache side channels, that just require providing the actual operation used
- `cache_side_channel` defines the interface cache side channels have to implement
- `cache_utils` contains utilities related to cache attacks
- `cpuid` is a small crate that handles CPU microarchitecture identification and provides info about what is known about it
- `flush_flush` and `flush_reload` are tiny crates that use `basic_timing_cache_channel` to export Flush+Flush and Flush+Reload primitives
- `turn_lock`  is the synchronization primitive used by `cache_utils`


### Rust versions

Known good nightly :

- rustc 1.54.0-nightly (eab201df7 2021-06-09)
- rustc 1.55.0-nightly (885399992 2021-07-06)
