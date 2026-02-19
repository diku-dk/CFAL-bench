SAC = sac2c
SAC_FLAGS = -maxwlur 2

SEQ_LPATH  := host/seq
MT_LPATH   := host/mt-pth
CUDA_LPATH := host/cuda

.PHONY: all clean bench_seq bench_omp bench

all: seq mt

seq: bin/nbody_seq 
mt: bin/nbody_mt
cuda: bin/nbody_cuda

bin/nbody_seq: src/nbody.sac $(SEQ_LPATH)/libVect3DMod.so
	$(SAC) $(SAC_FLAGS) $< -o $@

bin/nbody_mt: src/nbody.sac $(MT_LPATH)/libVect3DMod.so
	$(SAC) -t mt_pth  -mt_bind simple $(SAC_FLAGS) $< -o $@

bin/nbody_cuda: src/nbody.sac $(CUDA_LPATH)/libVect3DMod.so
	$(SAC) -t cuda $(SAC_FLAGS) $< -o $@

bench_seq:
	sbatch bench_seq.sh 10000 10 10 out

bench_mt:
	sbatch bench_mt.sh 10000 10 10 out 32

bench: bench_seq bench_mt

$(SEQ_LPATH)/lib%Mod.so: src/%.sac
	$(SAC) $(SAC_FLAGS) $<

$(MT_LPATH)/lib%Mod.so: src/%.sac
	$(SAC) -t mt_pth $(SAC_FLAGS) $<

$(CUDA_LPATH)/lib%Mod.so: src/%.sac
	$(SAC) -t cuda $(SAC_FLAGS) $<

clean:
	$(RM) bin/*
