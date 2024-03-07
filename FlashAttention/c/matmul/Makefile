CC = gcc
FLAGS = -Ofast -march=native -mtune=native -Wall -Wextra
DBUG_FLAGS = -fsanitize=address -fsanitize=undefined -g
# FLAGS += $(DBUG_FLAGS)
LFLAGS = -lm -lopenblas64

TARGETS = kernel kernel_optimal_zen2 kernel_unpacked matmul
BINARIES = $(patsubst %,bin/%,$(TARGETS))

.PHONY: all clean

all: $(BINARIES)

bin/%: src/%.c
	$(CC) $(FLAGS) $< -o $@ $(LFLAGS)

clean:
	$(RM) bin/*
