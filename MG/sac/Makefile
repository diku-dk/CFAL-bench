SEQ_LPATH  := host/seq
MT_LPATH   := host/mt-pth
CUDA_LPATH := host/cuda

DYLIB := so                          # adjust to your system (e.g. so)

SAC2C := sac2c_p                  # may need to adjust if sac2c is not in your PATH
SAC2CFLAGS := -maxwlur 27

all: seq test mt cuda

test: bin/RNG_test bin/input_test

seq: bin/MG_${CLASS}_seq

# No -mt_bind simple here! I don't know why, but this sometimes decreases
# the performance by 10x on NUMA architectures
mt: bin/MG_${CLASS}_mt
mt: SAC2CFLAGS += -t mt_pth -minmtsize 32768 #-mt_bind simple

cuda: bin/MG_${CLASS}_cuda
cuda: SAC2CFLAGS += -nomemrt

bin/MG_${CLASS}_seq: src/MG.sac $(SEQ_LPATH)/libInputMod.$(DYLIB)
	mkdir -p bin
	$(SAC2C) -D CLASS_${CLASS}=1 $(SAC2CFLAGS) $< -o$@
	$(RM) bin/MG_seq.*

bin/MG_${CLASS}_mt: src/MG.sac $(MT_LPATH)/libInputMod.$(DYLIB)
	mkdir -p bin
	$(SAC2C) -D CLASS_${CLASS}=1 $(SAC2CFLAGS) $< -o$@
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

$(SEQ_LPATH)/libInputMod.$(DYLIB): $(SEQ_LPATH)/libRNGMod.$(DYLIB)

$(MT_LPATH)/libInputMod.$(DYLIB): $(MT_LPATH)/libRNGMod.$(DYLIB)

$(CUDA_LPATH)/libInputMod.$(DYLIB): $(CUDA_LPATH)/libRNGMod.$(DYLIB)

$(SEQ_LPATH)/lib%Mod.$(DYLIB): src/%.sac
	$(SAC2C) $(SAC_FLAGS) $<

$(MT_LPATH)/lib%Mod.$(DYLIB): src/%.sac
	$(SAC2C) $(SAC_FLAGS) -t mt_pth $<

$(CUDA_LPATH)/lib%Mod.$(DYLIB): src/%.sac
	$(SAC2C) $(SAC_FLAGS) -tcuda_man $<


clean:
	$(RM) bin/*
	$(RM) -rf host tree
