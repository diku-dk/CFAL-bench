# NAS Parallel Benchmark MG Kernel in APL

## Running the benchmark

To run the benchmark using Dyalog APL, make sure you are in the current working
directory that contains the MG.apln file. 

Linux:

	$ LOAD=.\MG.apln dyalog

Windows:

	$ dyalog.exe LOAD=.\MG.apln
	
For Windows, you can also create a shortcut file that contains the above 
command as long as you ensure that the working directory is set appropriately. 

You can also examine the `RUN.bat` or `RUN.sh` files. 

## Editing and working with the code

If you wish to run the code manually and work with it, you can simply 
open dyalog in the directory and then run the following commands:

      ]load .\MG.apln
      MG.Run ''
