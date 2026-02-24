#!/usr/bin/env python3
#
# Accepts numbers on stdin, one per line, and reports standard statistical
# quantities. The numbers might for example be the .runtimes files produced by
# the job.sh scripts (possibly translated to GFLOP/s first if you want).

import sys
import numpy as np

numbers=[float(x) for x in sys.stdin.readlines()]

print('           mean:', np.mean(numbers))
print('         median:', np.median(numbers))
print('         stddev:', np.std(numbers))
print('  sample stddev:', np.std(numbers, ddof=1))
print('       variance:', np.var(numbers))
