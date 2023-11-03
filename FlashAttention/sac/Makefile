SAC = ~/repos/sac2c/build_CFAL/sac2c_d
SAC_FLAGS = 

.PHONY: all seq mt cuda clean

all: seq mt cuda

seq: bin/flash_seq
seq: SAC_FLAGS += -Xc -Ofast 
mt: bin/flash_mt
mt: SAC_FLAGS += -t mt_pth -Xc -Ofast
cuda: bin/flash_cuda
cuda: SAC_FLAGS += -t cuda -Xc -O3 -noemrl -profile c

bin/flash_%: src/flash.sac
	$(SAC) $(SAC_FLAGS) $< -o $@

clean:
	$(RM) bin/*
