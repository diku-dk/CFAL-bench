# Cheat CFLAGS.
CFLAGS='-Ofast -march=native -mtune=native'
FUTHARK_BACKEND?=multicore

.PHONY: run

run:
	CFLAGS=$(CFLAGS) futhark bench --backend=multicore nbody.fut --json nbody.json
