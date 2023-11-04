clean:
	rm -rf bin/*

exercise ?= default_input.c

build:
	mkdir -p ./bin
	nvcc src/exercise$(exercise)/solution.cu -o bin/exercise$(exercise) -arch=sm_50 -rdc=true

run:
	./bin/exercise$(exercise)
