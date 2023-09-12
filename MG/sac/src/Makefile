SEQ_LPATH  := host/seq
MT_LPATH   := host/mt-pth
CUDA_LPATH := host/cuda

DYLIB := dylib

MG_seq: MG.sac $(SEQ_LPATH)/libInitMod.$(DYLIB)
	sac2c -maxwlur 27 $< -o$@

MG_mt: MG.sac $(MT_LPATH)/libInitMod.$(DYLIB)
	sac2c -tmt_pth -maxwlur 27 $< -o$@

MG_cuda: MG.sac $(CUDA_LPATH)/libInitMod.$(DYLIB)
	sac2c -tcuda -maxwlur 27 $< -o$@

$(SEQ_LPATH)/libInitMod.$(DYLIB): $(SEQ_LPATH)/libRNGMod.$(DYLIB)

$(MT_LPATH)/libInitMod.$(DYLIB): $(MT_LPATH)/libRNGMod.$(DYLIB)

$(CUDA_LPATH)/libInitMod.$(DYLIB): $(CUDA_LPATH)/libRNGMod.$(DYLIB)



$(SEQ_LPATH)/lib%Mod.$(DYLIB): %.sac
	sac2c $<

$(MT_LPATH)/lib%Mod.$(DYLIB): %.sac
	sac2c -tmt_pth $<

$(CUDA_LPATH)/lib%Mod.$(DYLIB): %.sac
	sac2c -tcuda $<

