CXX=c++
CFLAGS=-O3 -Wall -fopenmp -lm -ffast-math

hull: hull.cpp
	$(CXX) hull.cpp -O3 -o hull -pthread -Wall

.PHONY: run

clean:
	rm -rf hull *.runtimes
