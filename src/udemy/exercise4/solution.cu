#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>


#define CHECK_CUDA_ERROR(call) { \
    cudaError_t err = call;     \
    if (err != cudaSuccess) {   \
        fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE);     \
    }                           \
}

__global__ void dynamic_parallelism_check(int size, int depth) {
    printf("Depth %d, bid %d, tid %d\n", depth, blockIdx.x, threadIdx.x);

    if (size == 1)
        return;

    if (blockIdx.x == 0 & threadIdx.x == 0)
        dynamic_parallelism_check<<<2, size / 2>>>(size / 2, depth + 1);
}


int main() {
    
    dynamic_parallelism_check<<<2, 8>>>(8, 0);

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR(cudaDeviceReset());
    return 0;
}
