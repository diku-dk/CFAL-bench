C = sac2c_d -maxwlur 27 -check c
FLAGS = -Xc -Ofast

all: bin/MG

RNG: src/RNG.sac
	$(C) $(FLAGS) $<

Init: src/Init.sac RNG
	$(C) $(FLAGS) $<

bin/RNG: src/RNG.sac # define DEBUG on commandline?
	$(C) $(FLAGS) $< -o $@

bin/init: src/init.sac RNG # define DEBUG on commandline?
	$(C) $(FLAGS) $< -o $@

bin/MG: src/MG.sac RNG Init
	$(C) $(FLAGS) $< -o $@

clean:
	$(RM) bin/*
	$(RM) -rf host tree
