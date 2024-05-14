# Cheat CFLAGS.
CFLAGS='-Ofast -march=native -mtune=native'
FUTHARK_BACKEND?=multicore

.PHONY: run

run:
	CFLAGS=$(CFLAGS) futhark bench --backend=$(FUTHARK_BACKEND) nbody.fut --json nbody.json
