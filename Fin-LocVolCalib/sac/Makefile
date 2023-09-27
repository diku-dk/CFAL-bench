# Cuda won't parallelise

SAC = sac2c_d
SAC_FLAGS = -Xc -Ofast

.PHONY: all clean run_small run_medium run_large

all: seq mt

seq: bin/small bin/medium bin/large
mt: bin/small_mt bin/medium_mt bin/large_mt
mt: SAC_FLAGS += -t mt_pth
# TODO Code won't cudarize, have a look at the GPU backend

bin/smal%: src/VolCalib.sac
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

bin/mediu%: src/VolCalib.sac
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

bin/larg%: src/VolCalib.sac
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

run_small:
	bin/small
	bin/small_mt -mt 4

run_medium:
	bin/medium
	bin/medium_mt -mt 4

run_large:
	bin/large
	bin/large_mt -mt 4

clean:
	$(RM) bin/*
