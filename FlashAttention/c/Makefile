CC = gcc
FLAGS = -Ofast -march=native -mtune=native -Wall -Wextra
DBUG_FLAGS = -fsanitize=address -fsanitize=undefined -g
# FLAGS += $(DBUG_FLAGS)
LFLAGS = -lm -lopenblas64

TARGETS = generate compare flash flash_mt attention flash_alg1 blas_alg1 custom_alg1
BINARIES = $(patsubst %,bin/%,$(TARGETS))

.PHONY: all clean seq mt alg1

all: $(BINARIES)

seq: bin/flash
mt: bin/flash_mt
alg1: bin/flash_alg1
blas: bin/blas_alg1

bin/%: src/%.c
	$(CC) $(FLAGS) $< -o $@ $(LFLAGS)

# special case because of the additional flag
bin/flash_mt: src/flash_mt.c
	$(CC) $(FLAGS) $< -o $@ $(LFLAGS) -fopenmp

bin/matmul.o: matmul/src/matmul.c
	$(CC) $(FLAGS) -c $< -o $@

bin/custom_alg1: src/custom_alg1.c bin/matmul.o
	$(CC) $(FLAGS) $^ -o $@ $(LFLAGS)

clean:
	$(RM) bin/*
