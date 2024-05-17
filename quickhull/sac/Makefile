SAC = ~/sac2c/build/sac2c_d -d cccall
SAC_FLAGS = 

.PHONY: all seq 

all: seq 

# No non-sequential version yet
seq: bin/quickhull

bin/quickhull: src/quickhull.sac
	$(SAC) $(SAC_FLAGS) $< -o $@

clean:
	$(RM) bin/*
