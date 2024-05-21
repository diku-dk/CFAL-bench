#!/usr/bin/env python3
#
# This simple script computes GFLOP/s from the runtime measurements in
# a 'futhark bench'-generated .json file. Requires the GFLOPS per
# dataset to be passed on the command line.

import sys
import json
import numpy as np

jsonf = sys.argv[1]
prog = sys.argv[2]
json = json.load(open(jsonf, 'r'))
for (dataset,v) in zip(sys.argv[3::2], sys.argv[4::2]):
    gflops = float(v)
    runtimes = np.array(json[prog]['datasets'][dataset]['runtimes']) / 1e6
    runtimes_sum = np.sum(runtimes)
    runtimes_mean = runtimes_sum / len(runtimes)
    gflop_s = gflops / runtimes
    gflop_per_s_mean = gflops / runtimes_mean
    print('Dataset {:20} mean GFLOP/s: {:10.2f}±{:.2}'
          .format(dataset+':', gflop_per_s_mean, np.std(gflop_s)))
