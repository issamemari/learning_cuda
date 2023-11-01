clean:
	rm -rf bin/*

exercise ?= default_input.c

build:
	nvcc src/exercise$(exercise)/solution.cu -o bin/exercise$(exercise)

run:
	./bin/exercise$(exercise)
