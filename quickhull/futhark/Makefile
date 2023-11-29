all: data/1M_circle_16384.dat data/1M_quadratic_2147483648.dat data/1M_rectangle_16384.dat

data/%:
	@mkdir -p data
	python3 input2fut.py < ../input/$* > $@
