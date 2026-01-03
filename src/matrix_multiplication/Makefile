NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_75

all: matmul_v1_naive

matmul_v1_naive: matmul_v1_naive.cu
	$(NVCC) $(NVCC_FLAGS) -o matmul_v1_naive matmul_v1_naive.cu

clean:
	rm -f matmul_v1_naive *.ncu-rep *.nsys-rep *.sqlite

run: matmul_v1_naive
	./matmul_v1_naive

# Profiling targets

# Nsight Systems (nsys) - System-wide profiling (WORKS!)
profile-nsys: matmul_v1_naive
	@echo "=== Nsight Systems Profile ==="
	nsys profile --stats=true -o matmul_v1_nsys ./matmul_v1_naive

profile-nsys-trace: matmul_v1_naive
	@echo "=== Nsight Systems with CUDA trace ==="
	nsys profile --trace=cuda,nvtx --stats=true -o matmul_v1_trace ./matmul_v1_naive

# Legacy profiling targets (may not work on newer GPUs)
profile-nvprof: matmul_v1_naive
	@echo "=== Basic nvprof Profile ==="
	nvprof ./matmul_v1_naive

profile-ncu: matmul_v1_naive
	@echo "=== Nsight Compute Basic Profile ==="
	sudo /usr/local/cuda/bin/ncu ./matmul_v1_naive

profile-ncu-memory: matmul_v1_naive
	@echo "=== Nsight Compute Memory Analysis ==="
	sudo /usr/local/cuda/bin/ncu --set memory ./matmul_v1_naive

.PHONY: all clean run profile-nsys profile-nsys-trace profile-nvprof profile-ncu profile-ncu-memory
