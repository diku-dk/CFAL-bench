SAC = sac2c_p
SAC_FLAGS = -maxwlur 2

.PHONY: all clean bench_seq bench_omp bench

all: seq mt

seq: bin/nbody_seq
seq: SAC_FLAGS += -t seq
mt: bin/nbody_mt
mt: SAC_FLAGS += -t mt_pth
cuda: bin/nbody_cuda
cuda: SAC_FLAGS = -t cuda

bin/nbody%: src/nbody.sac
	$(SAC) $(SAC_FLAGS) $< -o $@

bench_seq:
	sbatch bench_seq.sh 10000 10 10 out

bench_mt:
	sbatch bench_mt.sh 10000 10 10 out 32

bench: bench_seq bench_mt

clean:
	$(RM) bin/*
