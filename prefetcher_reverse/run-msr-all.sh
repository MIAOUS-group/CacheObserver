#!/bin/bash
PREFETCH_MSR=$1
sudo wrmsr -a 0x1a4 $PREFETCH_MSR
sudo rdmsr -a 0x1a4
cargo run --bin all_experiments --release > with-${PREFETCH_MSR}-prefetcher.log


