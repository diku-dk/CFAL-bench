# Common Makefile definitions for benchmarking Futhark programs. Set
# BENCHMARK before importing this.

CFLAGS='-Ofast -march=native -mtune=native'

.PHONY: run_multicore run_ispc run_cuda

run_multicore:
	CFLAGS=$(CFLAGS) futhark bench --backend=multicore $(BENCHMARK).fut --json $(BENCHMARK)_multicore.json --no-tuning
	CFLAGS=$(CFLAGS) futhark bench --backend=multicore $(BENCHMARK).fut --json $(BENCHMARK)_multicore_1.json --no-tuning --pass-option=--num-threads=1

run_ispc:
	CFLAGS=$(CFLAGS) futhark bench --backend=ispc $(BENCHMARK).fut --json $(BENCHMARK)_ispc.json --no-tuning
	CFLAGS=$(CFLAGS) futhark bench --backend=ispc $(BENCHMARK).fut --json $(BENCHMARK)_ispc_1.json --no-tuning --pass-option=--num-threads=1

run_cuda:
	CFLAGS=$(CFLAGS) futhark bench --backend=cuda $(BENCHMARK).fut --json $(BENCHMARK)_cuda.json
