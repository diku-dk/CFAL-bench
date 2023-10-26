SAC = sac2c_p
SAC_FLAGS = -Xc -Ofast

.PHONY: all clean run_small run_medium run_large

all: seq mt cuda

seq: bin/small_seq bin/medium_seq bin/large_cuda
mt: bin/small_mt bin/medium_mt bin/large_mt
mt: SAC_FLAGS += -t mt_pth -minmtsize 12
cuda: bin/small_cuda bin/medium_cuda bin/large_cuda
cuda: SAC_FLAGS = -t cuda -Xc -O3

bin/small_%: src/VolCalib.sac
	$(SAC) -D OUTER=16 \
		   -D NUM_X=32 \
		   -D NUM_Y=256 \
		   -D NUM_T=256 \
		   -D S0=0.03 \
		   -D T=5.0 \
		   -D ALPHA=0.2 \
		   -D NU=0.6 \
		   -D BETA=0.5 \
		   $(SAC_FLAGS) $< -o $@

bin/medium_%: src/VolCalib.sac
	$(SAC) -D OUTER=128 \
		   -D NUM_X=256 \
		   -D NUM_Y=32 \
		   -D NUM_T=64 \
		   -D S0=0.03 \
		   -D T=5.0 \
		   -D ALPHA=0.2 \
		   -D NU=0.6 \
		   -D BETA=0.5 \
		   $(SAC_FLAGS) $< -o $@

bin/large_%: src/VolCalib.sac
	$(SAC) -D OUTER=256 \
		   -D NUM_X=256 \
		   -D NUM_Y=256 \
		   -D NUM_T=64 \
		   -D S0=0.03 \
		   -D T=5.0 \
		   -D ALPHA=0.2 \
		   -D NU=0.6 \
		   -D BETA=0.5 \
		   $(SAC_FLAGS) $< -o $@

clean:
	$(RM) bin/*
