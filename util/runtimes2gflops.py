#!/usr/bin/env python3
#
# Convert runtime numbers passed on stdin (one per line) to GFLOP/s on stdout
# (also one per line). Requires two command line parameters: the benchmark name
# and workload.
#
# Example:
#
# $ python3 runtimes2gflops.py MG A < mg_baseline_gpu_A.runtimes

import sys

# Total GFLOPS per benchmark and workload.
GFLOPS = {
    'nbody': {'n=1000': 1800,
              'n=10000': 1800,
              'n=000000': 1800},
    'MG': {
        'A': 3.625,
        'B': 18.125,
        'C': 145
    },
    'FlashAttention': {
        '16384-64': 68.72,
        '32768-64': 274.88,
        '8192-128': 34.36,
        '16384-128': 127.44
    }
}

_, benchmark, workload = sys.argv

numbers=[float(x) for x in sys.stdin.readlines()]

for x in numbers:
    print(GFLOPS[benchmark][workload]/x)
