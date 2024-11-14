hull: hull.cpp
	g++ hull.cpp -O3 -o hull -pthread -Wall

.PHONY: run

run: hull
	rm -f quickhull.csv
	numactl --interleave all ./hull ../input/100M_circle.dat circle >> quickhull.csv
	numactl --interleave all ./hull ../input/100M_rectangle.dat rectangle >> quickhull.csv
	numactl --interleave all ./hull ../input/100M_quadratic.dat quadratic >> quickhull.csv
