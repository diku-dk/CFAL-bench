CC = gcc
FLAGS = -Ofast -march=native -mtune=native -Wall -Wextra
DBUG_FLAGS = -fsanitize=address -fsanitize=undefined -g
# FLAGS += $(DBUG_FLAGS)
# LFLAGS = -lm -lopenblas64
# LFLAGS = -lm -lmkl_rt

TARGETS = kernel kernel_optimal_zen2 kernel_unpacked test
BINARIES = $(patsubst %,bin/%,$(TARGETS))

.PHONY: all clean

all: $(BINARIES)

bin/matmul.o: src/matmul.c
	$(CC) $(FLAGS) -c $< -o $@

bin/%: src/%.c
	$(CC) $(FLAGS) $< -o $@ $(LFLAGS)

bin/test: src/test.c bin/matmul.o
	$(CC) $(FLAGS) $^ -o $@ $(LFLAGS)

clean:
	$(RM) bin/*
