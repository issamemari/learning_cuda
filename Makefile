clean:
	rm -rf bin/*

exercise ?= default_input.c
directory ?= udemy

build:
	mkdir -p ./bin
	nvcc src/$(directory)/exercise$(exercise)/solution.cu -o bin/$(directory)/exercise$(exercise) -arch=sm_50 -rdc=true

run:
	./bin/$(directory)/exercise$(exercise)
