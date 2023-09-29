# ------------------------------------------------------------------------------
# 
# The original NPB 3.4.1 version was written in Fortran and belongs to: 
# 	http://www.nas.nasa.gov/Software/NPB/
# 
# ------------------------------------------------------------------------------
# 
# The serial C++ version is a translation of the original NPB 3.4.1
# Serial C++ version: https://github.com/GMAP/NPB-CPP/tree/master/NPB-SER
# 
# Authors of the C++ code: 
# 	Dalvan Griebler <dalvangriebler@gmail.com>
# 	Gabriell Araujo <hexenoften@gmail.com>
# 	Júnior Löff <loffjh@gmail.com>
#
# ------------------------------------------------------------------------------
#
# The serial Python version is a translation of the NPB serial C++ version
# Serial Python version: https://github.com/PYTHON
# 
# Authors of the Python code:
#	LUPS (Laboratory of Ubiquitous and Parallel Systems)
#	UFPEL (Federal University of Pelotas)
#	Pelotas, Rio Grande do Sul, Brazil
#
# ------------------------------------------------------------------------------

import time
import numpy

# Global variables
start = numpy.repeat(0.0, 64)
elapsed = numpy.repeat(0.0, 64)

#*****************************************************************
#******            T  I  M  E  R  _  C  L  E  A  R          ******
#*****************************************************************
def timer_clear(n):
	global elapsed
	elapsed[n] = 0.0

#*****************************************************************
#******            T  I  M  E  R  _  S  T  A  R  T          ******
#*****************************************************************
def timer_start(n):
	global start
	start[n] = time.time()

#*****************************************************************
#******            T  I  M  E  R  _  S  T  O  P             ******
#*****************************************************************
def timer_stop(n):
	global elapsed
	t = (time.time() - start[n])
	elapsed[n] += t

#*****************************************************************
#******            T  I  M  E  R  _  R  E  A  D             ******
#*****************************************************************
def timer_read(n):
	return elapsed[n]
