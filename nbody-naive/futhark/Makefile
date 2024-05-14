# Cheat CFLAGS.
CFLAGS='-Ofast -march=native -mtune=native'

.PHONY: run_multicore run_ispc run_cuda

run_multicore:
	CFLAGS=$(CFLAGS) futhark bench --backend=multicore nbody.fut --json nbody_multicore.json

run_ispc:
	CFLAGS=$(CFLAGS) futhark bench --backend=ispc nbody.fut --json nbody_ispc.json

run_cuda:
	CFLAGS=$(CFLAGS) futhark bench --backend=cuda nbody.fut --json nbody_cuda.json
