# Cheat CFLAGS.
CFLAGS='-Ofast -march=native -mtune=native'

.PHONY: run_multicore run_ispc run_cuda

run_multicore:
	CFLAGS=$(CFLAGS) futhark bench --backend=multicore LocVolCalib.fut --json LocVolCalib_multicore.json --no-tuning

run_ispc:
	CFLAGS=$(CFLAGS) futhark bench --backend=ispc LocVolCalib.fut --json LocVolCalib_ispc.json --no-tuning

run_cuda:
	CFLAGS=$(CFLAGS) futhark bench --backend=cuda LocVolCalib.fut --json LocVolCalib_cuda.json
