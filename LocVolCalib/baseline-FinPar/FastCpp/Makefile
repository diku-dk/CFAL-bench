include ../setup.mk

ifndef ($(HAVE_GPU))
include ../platform.mk
endif

INCLUDES    += -I../includeC

SOURCES_CPP =VolCalibOrig.cpp
HELPERS     =../includeC/ParseInput.h
OBJECTS     =VolCalibOrig.o  
EXECUTABLE  =VolCalib

#g++ -I. -I/usr/local/cuda/include  -O3 NordeaVect.cpp -lOpenCL
default: cpu

.cpp.o: $(SOURCES_CPP) $(HELPERS)
	$(CXX) $(CXXFLAGS) -Ofast $(GPU_OPTS) $(INCLUDES) -c -o $@ $<

cpu: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $(EXECUTABLE) $(OBJECTS)

print_implicitx: PrintImplicitX.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< -o $@

print: print_implicitx
	cat ../Data/Small/input.data ../Data/Small/output.data | ./print_implicitx

run_small: $(EXECUTABLE)
	cat ../Data/Small/input.data ../Data/Small/output.data | ./$(EXECUTABLE) 2> Orig.txt

run_medium: $(EXECUTABLE)
	cat ../Data/Medium/input.data ../Data/Medium/output.data | ./$(EXECUTABLE) 2> Orig.txt

run_large: $(EXECUTABLE)
	cat ../Data/Large/input.data ../Data/Large/output.data | ./$(EXECUTABLE) 2> Orig.txt

clean:
	rm -f Orig.txt $(EXECUTABLE) $(OBJECTS)
