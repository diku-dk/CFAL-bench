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

import npbparams
import sys

#*****************************************************************
#******     C  _  P  R  I  N  T  _  R  E  S  U  L  T  S     ******
#*****************************************************************
def c_print_results(name,
					class_npb,
					n1, 
					n2, 
					n3,
					niter,
					t, 
					mops,
					optype,
					passed_verification):
	print("\n\n %s Benchmark Completed" % (name))
	print(" class_npb       =                        %s" % (class_npb))
	if name == "IS":
		if n3 == 0:
			nn = n1
			if n2 != 0:
				nn *= n2
			print(" Size            =             %12ld" % (nn)) # as in IS
		else:
			print(" Size            =             %4dx%4dx%4d" % (n1, n2, n3))
	else:
		if n2 == 0 and n3 == 0:
			if name =='EP':
				print(" Size            =          %15d" % (pow(2.0, n1)))
			else:
				print(" Size            =             %12d" % (n1))
		else:
			print(" Size            =           %4dx%4dx%4d" % (n1, n2, n3))
	
	print(" Iterations      =             %12d" % (niter)) 
	print(" Time in seconds =             %12.2f" % (t))
	print(" Mop/s total     =             %12.2f" % (mops))
	print(" Operation type  = %24s" % (optype))
	if passed_verification == None:
		print( " Verification    =            NOT PERFORMED")
	elif passed_verification:
		print(" Verification    =               SUCCESSFUL")
	else:
		print(" Verification    =             UNSUCCESSFUL")
	
	print(" NPB Version     =             %12s" % (npbparams.VERSION))
	print(" Python Version  = %s" % (sys.version.replace("\n", "")))

	# print(" Please send the results of this run to:\n")
	# print(" NPB Development Team")
	# print(" Internet: npb@nas.nasa.gov\n")
	# print(" If email is not available, send this to:\n")
	# print(" MS T27A-1")
	# print(" NASA Ames Research Center")
	# print(" Moffett Field, CA  94035-1000\n")
	# print(" Fax: 650-604-3957\n")
	print("\n")
	
	print("----------------------------------------------------------------------")
	print("    NPB-PYTHON is developed by:")
	print("        LUPS (Laboratory of Ubiquitous and Parallel Systems)")
	print("        UFPEL (Federal University of Pelotas)")
	print("        Pelotas, Rio Grande do Sul, Brazil")
	#print("")
	#print("    In case of questions or problems, please send an e-mail to us:")
	#print("        lups@ufpel.edu.br")
	print("----------------------------------------------------------------------\n")
