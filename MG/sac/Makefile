SEQ_LPATH  := host/seq
MT_LPATH   := host/mt-pth
CUDA_LPATH := host/cuda

DYLIB := dylib                          # adjust to your system (e.g. so)

SAC2C := sac2c                          # may need to adjust if sac2c is not in your PATH
SAC2CFLAGS := -maxwlur 27 -Xc -Ofast    # when changing you may want to include -check c

all: bin/MG_seq

mt: bin/MG_mt

cuda: bin/MG_cuda

bin/MG_seq: src/MG.sac $(SEQ_LPATH)/libInitMod.$(DYLIB)
	mkdir -p bin
	$(SAC2C) $(SAC2CFLAGS) $< -o$@
	$(RM) bin/MG_seq.*

bin/MG_mt: src/MG.sac $(MT_LPATH)/libInitMod.$(DYLIB)
	mkdir -p bin
	$(SAC2C) -tmt_pth $(SAC2CFLAGS) $< -o$@
	$(RM) bin/MG_mt.*

bin/MG_cuda: src/MG.sac $(CUDA_LPATH)/libInitMod.$(DYLIB)
	mkdir -p bin
	$(SAC2C) -tcuda $(SAC2CFLAGS) $< -o$@
	$(RM) bin/MG_cuda.*

$(SEQ_LPATH)/libInitMod.$(DYLIB): $(SEQ_LPATH)/libRNGMod.$(DYLIB)

$(MT_LPATH)/libInitMod.$(DYLIB): $(MT_LPATH)/libRNGMod.$(DYLIB)

$(CUDA_LPATH)/libInitMod.$(DYLIB): $(CUDA_LPATH)/libRNGMod.$(DYLIB)



$(SEQ_LPATH)/lib%Mod.$(DYLIB): src/%.sac
	$(SAC2C) $<

$(MT_LPATH)/lib%Mod.$(DYLIB): src/%.sac
	$(SAC2C) -tmt_pth $<

$(CUDA_LPATH)/lib%Mod.$(DYLIB): src/%.sac
	$(SAC2C) -tcuda $<


clean:
	$(RM) bin/*
	$(RM) -rf host tree
