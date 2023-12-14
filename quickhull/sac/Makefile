SAC = ~/sac2c/build/sac2c_d -d cccall
SAC_FLAGS = -L. #-profile m#-check c #-profile m
NVCC = nvcc
NVCC_FLAGS = -O3

.PHONY: all seq mt cuda clean

all: seq mt cuda

seq: bin/graham_cheat_seq
seq: SAC_FLAGS += -Xc -Ofast
mt: bin/graham_cheat_mt
mt: SAC_FLAGS += -t mt_pth -Xc -Ofast
cuda: bin/graham_cheat_cuda
cuda: SAC_FLAGS += -t cuda -Xc -O3 #-profile c

bin/graham_cheat_seq: src/graham_cheat.sac sort_cpu.o
	-$(SAC) $(SAC_FLAGS) $< -o $@
	sed -i '7s/\/usr\/bin\/cc/g++/' ./graham_cheat_seq.sac2c
	./graham_cheat_seq.sac2c

bin/graham_cheat_mt: src/graham_cheat.sac sort_cpu.o
	-$(SAC) $(SAC_FLAGS) $< -o $@
	sed -i '7s/\/usr\/bin\/cc/g++/' ./graham_cheat_mt.sac2c
	./graham_cheat_mt.sac2c

bin/graham_cheat_cuda: src/graham_cheat.sac sort_gpu.o
	-$(SAC) $(SAC_FLAGS) $< -o $@
	sed -i '7s/\/usr\/bin\/cc/g++/' ./graham_cheat_cuda.sac2c
	./graham_cheat_cuda.sac2c

sort_gpu.o: src/sort.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

sort_cpu.o: src/sort.cpp
	g++ -O3 -c $< -o $@

clean:
	$(RM) bin/*
	$(RM) *.sac2c
	$(RM) *.o
