.PHONY: all

all: points2sac input/1M_circle.dat input/1M_rectangle.dat input/1M_quadratic.dat input/100M_circle.dat input/100M_rectangle.dat input/100M_quadratic.dat
	cd input && md5sum -c MD5SUMS # To check that we generated the expected input.

gen_points: gen_points.c
	cc -o gen_points gen_points.c -O -Wall -Wextra -fsanitize=undefined

points2sac: points2sac.c
	cc -o points2sac points2sac.c -O -Wall -Wextra -fsanitize=undefined

input/1M_circle.dat: ./gen_points
	./gen_points 16384 16384 1000000 c > $@

input/1M_rectangle.dat: ./gen_points
	./gen_points 16384 16384 1000000 r > $@

input/1M_quadratic.dat: ./gen_points
	./gen_points 2000000000 2000000000 1000000 q > $@

input/100M_circle.dat: ./gen_points
	./gen_points 2000000000 2000000000 100000000 c > $@

input/100M_rectangle.dat: ./gen_points
	./gen_points 2000000000 2000000000 100000000 r > $@

input/100M_quadratic.dat: ./gen_points
	./gen_points 2000000000 2000000000 100000000 q > $@
