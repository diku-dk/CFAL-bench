MAKE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

include ../setup.mk

ifndef ($(HAVE_GPU))
include ../platform.mk
endif

INCLUDES  += -I../includeC
GPU_OPTS   = -D lgWARP=$(GPU_LG_WARP) -D GPU_DEV_ID=$(GPU_DEVICE_ID) \
             -D GPU_LOC_MEM=$(GPU_LOCAL_MEM) -D GPU_CONST_MEM=$(GPU_CONST_MEM) \
             -D GPU_REG_MEM=$(GPU_REG_MEM) -D GPU_GLB_MEM=$(GPU_DEVICE_MEM) \
             -D GPU_TILE=$(GPU_LOCAL_MEM_PER_TH) -D GPU_CORES=$(GPU_NUM_CORES) \
			 -D CURR_DIR_PATH='"$(MAKE_DIR)"'

SOURCES_CPP =VolCalibAll.cpp
HELPERS     =PrepareKernels.h Vect_CPU.h Vect_GPU.h VolCalibInit.h ../includeC/Constants.h \
              ../includeC/ParseInput.h ../includeC/DataStructConst.h ../includeC/ParPrefixUtil.h
OBJECTS     =VolCalibAll.o  
EXECUTABLE  =VolCalib

ifeq ($(HAVE_GPU),1)
default: gpu
else
# cannot build this one...
no-gpu:	
	@echo "\n*** No GPU configured, cannot build `pwd` ***\n"
endif

#.cpp.o: $(SOURCES_CPP) $(HELPERS)
#	$(CXX) $(CXXFLAGS) $(GPU_OPTS) $(INCLUDES) -c -o $@ $<

#$(OBJECTS)
cpu: 
	$(CXX) $(CXXFLAGS) -D IS_GPU=0 $(GPU_OPTS) $(INCLUDES) -o $(EXECUTABLE) $(SOURCES_CPP) $(LIB)

gpu: 
	$(CXX) $(CXXFLAGS) -D IS_GPU=1 $(GPU_OPTS) $(INCLUDES) -o $(EXECUTABLE) $(SOURCES_CPP) $(LIB)


run_small: 
	cat ../Data/Small/input.data ../Data/Small/output.data | ./$(EXECUTABLE) 2> Debug.txt

run_medium: 
	cat ../Data/Medium/input.data ../Data/Medium/output.data | ./$(EXECUTABLE) 2> Debug.txt

run_large: 
	cat ../Data/Large/input.data ../Data/Large/output.data | ./$(EXECUTABLE) 2> Debug.txt

clean:
	rm -f $(EXECUTABLE) $(OBJECTS) ./*.ptx ./Debug.txt
	@# clean nVidia compiler cache
	rm -rf $(HOME)/.nv/ComputeCache/*

