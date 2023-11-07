#!/usr/bin/env python3

import sys
import struct

tokens = sys.stdin.read().split()
num_points = len(tokens)//2

sys.stdout.buffer.write(b'b')
sys.stdout.buffer.write(b'\2')
sys.stdout.buffer.write(b'\2')
sys.stdout.buffer.write(b' f64')
sys.stdout.buffer.write(struct.pack('<Q', num_points))
sys.stdout.buffer.write(struct.pack('<Q', 2))
for i in range(0, num_points):
    sys.stdout.buffer.write(struct.pack('<d', float(tokens[i*2])))
    sys.stdout.buffer.write(struct.pack('<d', float(tokens[i*2+1])))
