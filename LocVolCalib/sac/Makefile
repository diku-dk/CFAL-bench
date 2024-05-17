SAC = sac2c_p
SAC_FLAGS = #-Xc -Ofast  #-profile m#-maxwlur 8 #-check c -ecc 
CC = gcc
CFLAGS = -Ofast -march=native -mtune=native -Wall -Wextra -Werror

.PHONY: all float clean

all: seq mt cuda

# Tridiagonal solver does not vectorize, so there is little use 
# in reducing the accuracy.
float: seq_float mt_float cuda_float

seq: bin/small_seq bin/medium_seq bin/large_seq
mt: bin/small_mt bin/medium_mt bin/large_mt
mt: SAC_FLAGS += -t mt_pth -minmtsize 1 -mt_bind simple
cuda: bin/small_cuda bin/medium_cuda bin/large_cuda
cuda: SAC_FLAGS = -t cuda -Xc -O3 -noemrl -profile c

seq_float: bin/small_seq_float bin/medium_seq_float bin/large_seq_float
mt_float: bin/small_mt_float bin/medium_mt_float bin/large_mt_float
mt_float: SAC_FLAGS += -t mt_pth -minmtsize 1 -mt_bind simple
cuda_float: bin/small_cuda_float bin/medium_cuda_float bin/large_cuda_float
cuda_float: SAC_FLAGS = -t cuda -Xc -O3

bin/tridag_v.o: src/tridag_v.c
	$(CC) -D REAL=float \
		  -D NUM_X=256 \
		  -D VEC=8 \
		  $(CFLAGS) -c $< -o $@

bin/large_manual: src/manual.sac bin/tridag_v.o
	$(SAC) -D OUTER=256 \
		   -D NUM_X=256 \
		   -D NUM_Y=256 \
		   -D NUM_T=64 \
		   -D S0=0.03f \
		   -D T=5.0f \
		   -D ALPHA=0.2f \
		   -D NU=0.6f \
		   -D BETA=0.5f \
		   $(SAC_FLAGS) -t mt_pth $< -o $@

bin/small_%: src/VolCalib.sac
	$(SAC) -D DOUBLE=1 \
		   -D OUTER=16 \
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
	$(SAC) -D DOUBLE=1 \
	       -D OUTER=128 \
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
	$(SAC) -D DOUBLE=1 \
	       -D OUTER=256 \
		   -D NUM_X=256 \
		   -D NUM_Y=256 \
		   -D NUM_T=64 \
		   -D S0=0.03 \
		   -D T=5.0 \
		   -D ALPHA=0.2 \
		   -D NU=0.6 \
		   -D BETA=0.5 \
		   $(SAC_FLAGS) $< -o $@

bin/small_%_float: src/VolCalib.sac
	$(SAC) -D OUTER=16 \
		   -D NUM_X=32 \
		   -D NUM_Y=256 \
		   -D NUM_T=256 \
		   -D S0=0.03f \
		   -D T=5.0f \
		   -D ALPHA=0.2f \
		   -D NU=0.6f \
		   -D BETA=0.5f \
		   $(SAC_FLAGS) $< -o $@

bin/medium_%_float: src/VolCalib.sac
	$(SAC) -D OUTER=128 \
		   -D NUM_X=256 \
		   -D NUM_Y=32 \
		   -D NUM_T=64 \
		   -D S0=0.03f \
		   -D T=5.0f \
		   -D ALPHA=0.2f \
		   -D NU=0.6f \
		   -D BETA=0.5f \
		   $(SAC_FLAGS) $< -o $@

bin/large_%_float: src/VolCalib.sac
	$(SAC) -D OUTER=256 \
		   -D NUM_X=256 \
		   -D NUM_Y=256 \
		   -D NUM_T=64 \
		   -D S0=0.03f \
		   -D T=5.0f \
		   -D ALPHA=0.2f \
		   -D NU=0.6f \
		   -D BETA=0.5f \
		   $(SAC_FLAGS) $< -o $@

clean:
	$(RM) bin/*
