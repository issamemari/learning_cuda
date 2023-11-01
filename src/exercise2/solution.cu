#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void printArray(int* array, unsigned int size) {
    int threadsPerBlock = blockDim.x * blockDim.y * blockDim.z;
    int blockOffset = blockIdx.z * (gridDim.y * gridDim.x) + blockIdx.y * gridDim.x + blockIdx.x;
    int threadOffset = blockOffset * threadsPerBlock;

    int gid = threadOffset + threadIdx.z * (blockDim.y * blockDim.x) + threadIdx.y * blockDim.x + threadIdx.x;

    if (gid < size)
        printf("Thread gid %d array element %d\n", gid, array[gid]);
}

int* createRandomArray(unsigned int size) {
    int* array = (int*)malloc(size * sizeof(int));

    // Check if memory allocation was successful
    if (array == nullptr) {
        printf("Memory allocation failed\n");
        return nullptr;
    }

    // Seed the random number generator
    srand(static_cast<unsigned int>(time(nullptr)));

    // Fill the array with random values
    for (int i = 0; i < size; ++i) {
        array[i] = rand() % 100; // Random numbers between 0 and 99
    }

    return array;
}

int main() {
    const int size = 64;

    int* h_array = createRandomArray(size);

    int* d_array;
    cudaMalloc(&d_array, size * sizeof(int));
    cudaMemcpy(d_array, h_array, size * sizeof(int), cudaMemcpyHostToDevice);

    dim3 block(2, 2, 2);
    dim3 grid(2, 2, 2);

    printArray<<<grid, block>>>(d_array, size);
    cudaDeviceSynchronize();

    cudaDeviceReset();
    return 0;
}
