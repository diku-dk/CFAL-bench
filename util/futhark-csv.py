#!/usr/bin/env python3
#
# Convert a JSON file produced by 'futhark bench' to a three-column
# CSV file. The first column is a user-provided name, the second is
# the name of the dataset, and the last is a runtime in seconds.
# Prints the CSV on stdout and expects the JSON on stdin.

import json
import sys

_, benchname = sys.argv

json = json.load(sys.stdin)

if len(json) != 1:
    print('JSON file must only contain a single program', file=sys.stderr)
    sys.exit(1)

prog=next(iter(json))
for d in json[prog]['datasets']:
    runtimes = json[prog]['datasets'][d]['runtimes']
    for r in runtimes:
        print(f'{benchname},{d},{r/1e6}')
