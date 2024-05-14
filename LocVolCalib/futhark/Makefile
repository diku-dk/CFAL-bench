# Cheat CFLAGS.
CFLAGS='-Ofast -march=native -mtune=native'
BENCHMARK=LocVolCalib

.PHONY: run_multicore run_ispc run_cuda

run_multicore:
	CFLAGS=$(CFLAGS) futhark bench --backend=multicore $(BENCHMARK).fut --json $(BENCHMARK)_multicore.json --no-tuning

run_ispc:
	CFLAGS=$(CFLAGS) futhark bench --backend=ispc $(BENCHMARK).fut --json $(BENCHMARK)_ispc.json --no-tuning

run_cuda:
	CFLAGS=$(CFLAGS) futhark bench --backend=cuda $(BENCHMARK).fut --json $(BENCHMARK)_cuda.json
