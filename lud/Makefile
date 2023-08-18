CXX        = nvcc -O3

#-arch=sm_60

SRC = main.cu
HELPERS = kernels.cu.h goldenSeq.h
EXECUTABLE = lud

default: compile run

.cu.o: $(SRC) $(HELPERS)
	$(CXX) -c $@ $<

compile: $(EXECUTABLE)

$(EXECUTABLE): $(SRC) $(HELPERS)
	$(CXX) -o $(EXECUTABLE) $(SRC)


run: $(EXECUTABLE)
	./$(EXECUTABLE) 1024
	
#4096

#28544

clean:
	rm -f $(EXECUTABLE)

