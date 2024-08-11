CC = gcc
FLAGS = -Ofast -march=native -mtune=native -Wall -Wextra -I matmul/include
DBUG_FLAGS = -fsanitize=address -fsanitize=undefined -g
# FLAGS += $(DBUG_FLAGS)
LFLAGS = -lm

TARGETS = flash_attention_cpu
BINARIES = $(patsubst %,bin/%,$(TARGETS))

.PHONY: all clean

all: $(BINARIES)

bin/matmul.o: matmul/src/matmul.c
	$(CC) $(FLAGS) -c $< -o $@

bin/flash_attention_cpu: flash_attention_cpu.c bin/matmul.o
	$(CC) $(FLAGS) $^ -o $@ $(LFLAGS) -fopenmp

clean:
	$(RM) bin/*
