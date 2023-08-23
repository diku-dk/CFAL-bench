SRC = kdTree.hs
EXECUTABLE = kdTree

default: compile run

compile: $(EXECUTABLE)
	
$(EXECUTABLE): $(SRC) 
	ghc -O $(SRC)

run: $(EXECUTABLE)
	./$(EXECUTABLE)

clean:
	rm -f $(EXECUTABLE) $(EXECUTABLE).hi $(EXECUTABLE).o

