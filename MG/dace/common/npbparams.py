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


# This utility configures a NPB to be executed for a specific class
import sys
import math

# this is the master version number for this set of 
# NPB benchmarks. it is in an obscure place so people
# won't accidentally change it. 
VERSION = "4.1"

# Global variables
CLASS = ""

#Commom
NITER_DEFAULT = 0 #BT, FT and SP
DT_DEFAULT = 0.0 #BT, LU and SP
PROBLEM_SIZE = 0 #BT and SP

# BT
# --

# CG
NA = 0
NONZER = 0
NITER = 0
SHIFT = 0.0
RCOND = 1.0e-1

# EP
M = 0 

#FT
NX = 0
NY = 0
NZ = 0
MAXDIM = 0
NXP = 0
NYP = 0
NTOTAL = 0
NTOTALP = 0
DEFAULT_BEHAVIOR = 1

#LU
ISIZ1 = 0 
ISIZ2 = 0
ISIZ3 = 0
ITMAX_DEFAULT = 0
INORM_DEFAULT = 0

# MG
NX_DEFAULT = 0
NY_DEFAULT = 0
NZ_DEFAULT = 0
NIT_DEFAULT = 0
LM = 0
LT_DEFAULT = 0
DEBUG_DEFAULT = 0
NDIM1 = 5
NDIM2 = 5
NDIM3 = 5
ONE = 1


def set_bt_info(class_npb):
	global CLASS
	global PROBLEM_SIZE, NITER_DEFAULT, DT_DEFAULT
	
	CLASS = class_npb
	
	if class_npb == 'S':
		PROBLEM_SIZE, NITER_DEFAULT, DT_DEFAULT = 12, 60, 0.010
	elif class_npb == 'W':
		PROBLEM_SIZE, NITER_DEFAULT, DT_DEFAULT = 24, 200, 0.0008
	elif class_npb == 'A':
		PROBLEM_SIZE, NITER_DEFAULT, DT_DEFAULT = 64, 200, 0.0008
	elif class_npb == 'B':
		PROBLEM_SIZE, NITER_DEFAULT, DT_DEFAULT = 102, 200, 0.0003
	elif class_npb == 'C':
		PROBLEM_SIZE, NITER_DEFAULT, DT_DEFAULT = 162, 200, 0.0001
	elif class_npb == 'D':
		PROBLEM_SIZE, NITER_DEFAULT, DT_DEFAULT = 408, 250, 0.00002
	elif class_npb == 'E':
		PROBLEM_SIZE, NITER_DEFAULT, DT_DEFAULT = 1020, 250, 0.4e-5
	else:
		print("npbparams.py: Internal error: invalid class_npb type", class_npb)
		sys.exit()
#END set_bt_info()

def set_cg_info(class_npb):
	global CLASS
	global NA, NONZER, NITER, SHIFT, RCOND
	
	CLASS = class_npb
	
	if class_npb == 'S':
		NA = 1400
		NONZER = 7
		NITER = 15
		SHIFT = 10.0 
	elif class_npb == 'W':
		NA = 7000
		NONZER = 8
		NITER = 15
		SHIFT = 12.0 
	elif class_npb == 'A':
		NA = 14000
		NONZER = 11
		NITER = 15
		SHIFT = 20.0 
	elif class_npb == 'B':
		NA = 75000
		NONZER = 13
		NITER = 75
		SHIFT = 60.0 
	elif class_npb == 'C':
		NA = 150000
		NONZER = 15
		NITER = 75
		SHIFT = 110.0 
	elif class_npb == 'D':
		NA = 1500000
		NONZER = 21
		NITER = 100
		SHIFT = 500.0 
	elif class_npb == 'E':
		NA = 9000000
		NONZER = 26
		NITER = 100
		SHIFT = 1.5e3
	else:
		print("npbparams.py: Internal error: invalid class_npb type", class_npb)
		sys.exit()
#END set_cg_info()

def set_ep_info(class_npb):
	global CLASS, M
	
	CLASS = class_npb
	
	if class_npb == 'S':
		M = 24
	elif class_npb == 'W':
		M = 25
	elif class_npb == 'A':
		M = 28
	elif class_npb == 'B':
		M = 30
	elif class_npb == 'C':
		M = 32
	elif class_npb == 'D':
		M = 36
	elif class_npb == 'E':
		M = 40
	else:
		print("npbparams.py: Internal error: invalid class_npb type", class_npb)
		sys.exit()
#END set_ep_info()

def set_ft_info(class_npb):
	global CLASS
	global NX, NY, NZ
	global MAXDIM, NITER_DEFAULT
	global NXP, NYP, NTOTAL, NTOTALP
	
	CLASS = class_npb
	
	nx, ny, nz = 0, 0, 0
	niter = 0
	if class_npb == 'S':
		nx, ny, nz = 64, 64, 64
		niter = 6
	elif class_npb == 'W':
		nx, ny, nz = 128, 128, 32
		niter = 6
	elif class_npb == 'A':
		nx, ny, nz = 256, 256, 128
		niter = 6
	elif class_npb == 'B':
		nx, ny, nz = 512, 256, 256
		niter = 20
	elif class_npb == 'C':
		nx, ny, nz = 512, 512, 512
		niter = 20
	elif class_npb == 'D':
		nx, ny, nz = 2048, 1024, 1024
		niter = 25
	elif class_npb == 'E':
		nx, ny, nz = 4096, 2048, 2048
		niter = 25
	else:
		print("npbparams.py: Internal error: invalid class_npb type", class_npb)
		sys.exit()
	
	maxdim = nx
	if ny > maxdim:
		maxdim = ny
	if nx > maxdim:
		maxdim = nx
	
	NX = nx
	NY = ny
	NZ = nz
	MAXDIM = maxdim
	NITER_DEFAULT = niter
	NXP = nx+1
	NYP = ny
	NTOTAL = nx * ny * nz
	NTOTALP = (nx+1) * ny * nz
#END set_ft_info()

def set_is_info(class_npb):
	global CLASS
	
	CLASS = class_npb
	
	if class_npb not in ['S', 'W', 'A', 'B', 'C', 'D']:
		print("npbparams.py: Internal error: invalid class_npb type", class_npb)
		sys.exit()
#END set_is_info()

def set_lu_info(class_npb):
	global CLASS
	global ISIZ1, ISIZ2, ISIZ3
	global ITMAX_DEFAULT, INORM_DEFAULT, DT_DEFAULT
	
	CLASS = class_npb
	
	if class_npb == 'S':
		problem_size, ITMAX_DEFAULT, INORM_DEFAULT, DT_DEFAULT = 12, 50, 50, 0.5
	elif class_npb == 'W':
		problem_size, ITMAX_DEFAULT, INORM_DEFAULT, DT_DEFAULT = 33, 300, 300, 1.5e-3
	elif class_npb == 'A':
		problem_size, ITMAX_DEFAULT, INORM_DEFAULT, DT_DEFAULT = 64, 250, 250, 2.0
	elif class_npb == 'B':
		problem_size, ITMAX_DEFAULT, INORM_DEFAULT, DT_DEFAULT = 102, 250, 250, 2.0
	elif class_npb == 'C':
		problem_size, ITMAX_DEFAULT, INORM_DEFAULT, DT_DEFAULT = 162, 250, 250, 2.0
	elif class_npb == 'D':
		problem_size, ITMAX_DEFAULT, INORM_DEFAULT, DT_DEFAULT = 408, 300, 300, 1.0
	elif class_npb == 'E':
		problem_size, ITMAX_DEFAULT, INORM_DEFAULT, DT_DEFAULT = 1020, 300, 300, 0.5
	else:
		print("npbparams.py: Internal error: invalid class_npb type", class_npb)
		sys.exit()
	
	ISIZ1, ISIZ2, ISIZ3 = problem_size, problem_size, problem_size
#END set_lu_info()

def set_mg_info(class_npb):
	global CLASS
	global NX_DEFAULT, NY_DEFAULT, NZ_DEFAULT
	global NIT_DEFAULT, LM, LT_DEFAULT
	global NDIM1, NDIM2, NDIM3
	
	CLASS = class_npb
	
	problem_size = 0
	nit = 0
	if class_npb == 'S':
		problem_size = 32
		nit = 4
	elif class_npb == 'W':
		problem_size = 128
		nit = 4
	elif class_npb == 'A':
		problem_size = 256
		nit = 4
	elif class_npb == 'B':
		problem_size = 256
		nit = 20
	elif class_npb == 'C':
		problem_size = 512
		nit = 20
	elif class_npb == 'D':
		problem_size = 1024
		nit = 50
	elif class_npb == 'E':
		problem_size = 2048
		nit = 50
	else:
		print("npbparams.py: Internal error: invalid class_npb type", class_npb)
		sys.exit()
	
	log2_size = math.ceil(math.log(problem_size, 2))
	# lt is log of largest total dimension
	lt_default = log2_size
	# log of log of maximum dimension on a node
	lm = log2_size
	ndim1 = lm
	ndim3 = log2_size
	ndim2 = log2_size
	
	NX_DEFAULT = problem_size
	NY_DEFAULT = problem_size
	NZ_DEFAULT = problem_size
	NIT_DEFAULT = nit
	LM = lm
	LT_DEFAULT = lt_default
	NDIM1 = ndim1
	NDIM2 = ndim2
	NDIM3 = ndim3
#END set_mg_info()

def set_sp_info(class_npb):
	global CLASS
	global PROBLEM_SIZE, NITER_DEFAULT, DT_DEFAULT
	
	CLASS = class_npb
	
	if class_npb == 'S':
		PROBLEM_SIZE, DT_DEFAULT, NITER_DEFAULT = 12, 0.015, 100
	elif class_npb == 'W':
		PROBLEM_SIZE, DT_DEFAULT, NITER_DEFAULT = 36, 0.0015, 400
	elif class_npb == 'A':
		PROBLEM_SIZE, DT_DEFAULT, NITER_DEFAULT = 64, 0.0015, 400
	elif class_npb == 'B':
		PROBLEM_SIZE, DT_DEFAULT, NITER_DEFAULT = 102, 0.001, 400
	elif class_npb == 'C':
		PROBLEM_SIZE, DT_DEFAULT, NITER_DEFAULT = 162, 0.00067, 400
	elif class_npb == 'D':
		PROBLEM_SIZE, DT_DEFAULT, NITER_DEFAULT = 408, 0.00030, 500
	elif class_npb == 'E':
		PROBLEM_SIZE, DT_DEFAULT, NITER_DEFAULT = 1020, 0.0001, 500
	else:
		print("npbparams.py: Internal error: invalid class_npb type", class_npb)
		sys.exit()
#END set_bt_info()

