#!/usr/bin/env python3
#
# Convert CSV files with runtime results into a table.

import csv
import sys
import collections
import numpy as np

# Our workloads have been chosen to have the same total FLOPs.
GFLOPS=1800

results={}

# Read data from CSV files and populare 'results' dict.
for f in sys.argv[1:]:
    r = csv.reader(open(f, 'r'), delimiter=',')
    for (prog,workload,seconds) in r:
        if not prog in results:
            results[prog] = {}
        if not workload in results[prog]:
            results[prog][workload] = []
        results[prog][workload].append(float(seconds))

def entry(runtimes):
    runtimes_sum = np.sum(runtimes)
    runtimes_mean = runtimes_sum / len(runtimes)
    gflop_s = GFLOPS / runtimes
    gflop_per_s_mean = GFLOPS / runtimes_mean
    return '{:10.2f} {:6}'.format(gflop_per_s_mean,
                                  '±' + '{:.2f}'.format(np.std(gflop_s)))

workloads=['n=1000', 'n=10000', 'n=100000']
fstr='{:20} {:>20} {:>20} {:>20}'
print(fstr.format('', *workloads))
for prog in results:
    entries=[entry(np.array(results[prog][workloads[0]])),
             entry(np.array(results[prog][workloads[1]])),
             entry(np.array(results[prog][workloads[2]]))]
    print(fstr.format(prog, *entries))
