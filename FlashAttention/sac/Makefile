SAC = sac2c
SAC_FLAGS = #-profile m

.PHONY: all seq mt cuda clean

all: seq mt cuda

seq: bin/flash_seq
mt: bin/flash_mt
mt: SAC_FLAGS += -t mt_pth -mt_bind simple
cuda: bin/flash_cuda
cuda: SAC_FLAGS += -t cuda

bin/flash_%: src/flash.sac
	$(SAC) $(SAC_FLAGS) $< -o $@

clean:
	$(RM) bin/*
