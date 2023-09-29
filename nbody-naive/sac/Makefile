SAC = /home/thomas/repos/sac2c/build_CFAL/sac2c_d
SAC_FLAGS = -Xc -Ofast -d cccall -Xc -flto

.PHONY: all clean

all: seq mt cuda

seq: bin/nbody_seq
mt: bin/nbody_mt
mt: SAC_FLAGS += -t mt_pth
cuda: bin/nbody_cuda
cuda: SAC_FLAGS = -t cuda -Xc -O3 -doPFD

bin/nbody%: src/nbody.sac
	$(SAC) -D N=4096 \
		   -D ITER=10 \
		   $(SAC_FLAGS) $< -o $@

clean:
	$(RM) bin/*
