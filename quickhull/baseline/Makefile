hull: hull.cpp
	g++ hull.cpp -O3 -o hull -pthread -Wall

.PHONY: run

run: hull
	rm -f quickhull.csv
	./hull ../input/100M_circle.dat circle >> quickhull.csv
	./hull ../input/100M_rectangle.dat rectangle >> quickhull.csv
	./hull ../input/100M_quadratic.dat quadratic >> quickhull.csv
