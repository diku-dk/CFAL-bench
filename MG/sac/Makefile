SEQ_LPATH  := host/seq
MT_LPATH   := host/mt-pth
CUDA_LPATH := host/cuda

DYLIB := so                          # adjust to your system (e.g. so)

SAC2C := sac2c_p                  # may need to adjust if sac2c is not in your PATH
SAC2CFLAGS := -maxwlur 27 -v3 # when changing you may want to include -check c

all: seq test mt cuda

test: bin/RNG_test bin/input_test

seq: bin/MG_${CLASS}_seq

mt: bin/MG_${CLASS}_mt

cuda: bin/MG_${CLASS}_cuda
cuda: SAC2CFLAGS += -nomemrt

bin/MG_${CLASS}_seq: src/MG.sac $(SEQ_LPATH)/libInputMod.$(DYLIB)
	mkdir -p bin
	$(SAC2C) -D CLASS_${CLASS}=1 $(SAC2CFLAGS) $< -o$@
	$(RM) bin/MG_seq.*

bin/MG_${CLASS}_mt: src/MG.sac $(MT_LPATH)/libInputMod.$(DYLIB)
	mkdir -p bin
	$(SAC2C) -D CLASS_${CLASS}=1 -tmt_pth $(SAC2CFLAGS) $< -o$@
	$(RM) bin/MG_mt.*

bin/MG_${CLASS}_cuda: src/MG.sac $(CUDA_LPATH)/libInputMod.$(DYLIB)
	mkdir -p bin
	$(SAC2C) -D CLASS_${CLASS}=1 -tcuda_man $(SAC2CFLAGS) $< -o$@
	$(RM) bin/MG_cuda.*

bin/RNG_test: src/RNG_test.sac $(SEQ_LPATH)/libRNGMod.$(DYLIB)
	mkdir -p bin
	$(SAC2C) $(SAC2CFLAGS) $< -o $@

bin/input_test: src/input_test.sac $(SEQ_LPATH)/libInputMod.$(DYLIB)
	mkdir -p bin
	$(SAC2C) $(SAC2CFLAGS) $< -o $@

bench: bench_seq bench_mt bench_cuda

bench_seq:
	sbatch bench_seq.sh B 10 out

bench_mt:
	sbatch bench_mt.sh B 10 out 32

bench_cuda:
	sbatch bench_cuda.sh B 10 out

$(SEQ_LPATH)/libInputMod.$(DYLIB): $(SEQ_LPATH)/libRNGMod.$(DYLIB)

$(MT_LPATH)/libInputMod.$(DYLIB): $(MT_LPATH)/libRNGMod.$(DYLIB)

$(CUDA_LPATH)/libInputMod.$(DYLIB): $(CUDA_LPATH)/libRNGMod.$(DYLIB)



$(SEQ_LPATH)/lib%Mod.$(DYLIB): src/%.sac
	$(SAC2C) $(SAC_FLAGS) $<

$(MT_LPATH)/lib%Mod.$(DYLIB): src/%.sac
	$(SAC2C) $(SAC_FLAGS) -tmt_pth $<

$(CUDA_LPATH)/lib%Mod.$(DYLIB): src/%.sac
	$(SAC2C) $(SAC_FLAGS) -tcuda_man $<


clean:
	$(RM) bin/*
	$(RM) -rf host tree
