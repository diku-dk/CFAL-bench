#!/usr/bin/env python3
#
# Compute the GFLOP/s for each nbody runtime, passed on stdin.

import sys

# Total GFLOPS for all workloads.
GFLOPS=1800

numbers=[float(x) for x in sys.stdin.readlines()]

for x in numbers:
    print(GFLOPS/x)
