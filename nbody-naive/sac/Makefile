SAC = ${HOME}/sac2c/build/sac2c_d
SAC_FLAGS = 

.PHONY: all clean bench_seq bench_omp bench

all: seq mt cuda

seq: bin/nbody_seq
seq: SAC_FLAGS += -t seq -Xc -Ofast
mt: bin/nbody_mt
mt: SAC_FLAGS += -t mt_pth -Xc -Ofast
cuda: bin/nbody_cuda
cuda: SAC_FLAGS = -t cuda -Xc -O3

bin/nbody%: src/nbody.sac
	$(SAC) -D N=${N} \
		   -D ITER=${ITER} \
		   $(SAC_FLAGS) $< -o $@

bench_seq:
	sbatch bench_seq.sh 10000 10 10 out

bench_mt:
	sbatch bench_mt.sh 10000 10 10 out 32

bench: bench_seq bench_mt

clean:
	$(RM) bin/*
