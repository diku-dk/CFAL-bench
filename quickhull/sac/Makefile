SAC = ~/repos/sac2c/build_wlir/sac2c_d
SAC_FLAGS = #-profile m#-check c #-profile m

.PHONY: all seq mt cuda clean

all: seq mt cuda

seq: bin/quickhull_seq
#seq: SAC_FLAGS += -Xc -Ofast 
mt: bin/quickhull_mt
#mt: SAC_FLAGS += -t mt_pth -Xc -Ofast
cuda: bin/quickhull_cuda
cuda: SAC_FLAGS += -t cuda -Xc -O3 #-profile c

bin/quickhull_%: src/quickhull.sac
	$(SAC) $(SAC_FLAGS) $< -o $@

clean:
	$(RM) bin/*
