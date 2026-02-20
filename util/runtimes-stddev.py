#!/usr/bin/env python3
#
# Compute standard deviation for the numbers (one per line) read from stdin.

import sys
import numpy as np

numbers=[float(x) for x in sys.stdin.readlines()]

print(np.std(numbers))
