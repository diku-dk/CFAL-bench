all: data/1M_circle_16384.dat data/1M_quadratic_2147483648.dat data/1M_rectangle_16384.dat

data/%:
	@mkdir -p data
	python3 input2fut.py < ../input/$* > $@

# Cheat CFLAGS.
CFLAGS='-Ofast -march=native -mtune=native'
BENCHMARK=quickhull

.PHONY: run_multicore run_ispc run_cuda

run_multicore:
	CFLAGS=$(CFLAGS) futhark bench --backend=multicore $(BENCHMARK).fut --json $(BENCHMARK)_multicore.json --no-tuning

run_ispc:
	CFLAGS=$(CFLAGS) futhark bench --backend=ispc $(BENCHMARK).fut --json $(BENCHMARK)_ispc.json --no-tuning

run_cuda:
	CFLAGS=$(CFLAGS) futhark bench --backend=cuda $(BENCHMARK).fut --json $(BENCHMARK)_cuda.json
