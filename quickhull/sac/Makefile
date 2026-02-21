SAC = sac2c
SAC_FLAGS = 

.PHONY: all seq clean input

all: seq 
input: input/100M_quadratic input/100M_rectangle input/100M_circle

# No non-sequential version yet
seq: bin/quickhull

bin/quickhull: src/quickhull.sac
	$(SAC) $(SAC_FLAGS) $< -o $@

../input/%.dat:
	cd .. && $(MAKE) input/$*.dat

# Implicit rule does not work if it is a dependency...
../points2sac: ../points2sac.c
	$(LINK.c) $^ $(LOADLIBES) $(LDLIBS) -o $@

../gen_points: ../gen_points.c
	$(LINK.c) $^ $(LOADLIBES) $(LDLIBS) -o $@

input/%: ../input/%.dat ../points2sac ../gen_points
	../points2sac $< $@
		

clean:
	$(RM) bin/*
