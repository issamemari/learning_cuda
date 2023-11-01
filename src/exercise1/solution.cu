#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void printThreadInfo() {
    printf(
        "threadIdx.x: %d, threadIdx.y: %d, threadIdx.z: %d\n",
        threadIdx.x, threadIdx.y, threadIdx.z
    );
    printf(
        "blockIdx.x: %d, blockIdx.y: %d, blockIdx.z: %d\n",
        blockIdx.x, blockIdx.y, blockIdx.z
    );
    printf(
        "gridDim.x: %d, gridDim.y: %d, gridDim.z: %d\n",
        gridDim.x, gridDim.y, gridDim.z
    );
    printf(
        "blockDim.x: %d, blockDim.y: %d, blockDim.z: %d\n",
        blockDim.x, blockDim.y, blockDim.z
    );
}

int main() {
    dim3 block(2, 2, 2);
    dim3 grid(2, 2, 2);

    printThreadInfo<<<block, grid>>>();
    cudaDeviceSynchronize();

    cudaDeviceReset();
    return 0;
}
