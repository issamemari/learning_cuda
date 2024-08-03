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


__global__ void gpuRecursiveReduce(int* g_idata, int* g_odata, unsigned int isize) {
    int tid = threadIdx.x;

    int* idata = g_idata + blockIdx.x * blockDim.x;
    int* odata = &g_odata[blockIdx.x];

    if (isize == 2 && tid == 0) {
        g_odata[blockIdx.x] = idata[0] + idata[1];
        return;
    }
    int istride = isize >> 1;

    if (istride > 1 && tid < istride) {
        idata[tid] += idata[tid + istride];
    }

    if (tid == 0)
        gpuRecursiveReduce<<<1, istride>>>(idata, odata, istride);
}

int* createRandomArray(unsigned int size) {
    int* array = (int*)malloc(size * sizeof(int));

    // Check if memory allocation was successful
    if (array == nullptr) {
        printf("Memory allocation failed\n");
        return nullptr;
    }

    // Fill the array with random values
    for (int i = 0; i < size; ++i) {
        array[i] = rand() % 100; // Random numbers between 0 and 99
    }

    return array;
}

int main() {
    int input_size = 128;
    int input_byte_size = input_size * sizeof(int);

    int block_size = 128;
    int grid_size = input_size / block_size;

    int* input_data = createRandomArray(input_size);
    int* output_data = (int*)malloc(grid_size);

    int* device_input_data;
    cudaMalloc(&device_input_data, input_byte_size);

    int* device_output_data;
    cudaMalloc(&device_output_data, input_byte_size);

    cudaMemcpy(device_input_data, input_data, input_byte_size, cudaMemcpyHostToDevice);

    dim3 block(block_size);
    dim3 grid(grid_size);

    gpuRecursiveReduce<<<grid, block>>>(device_input_data, device_output_data, input_size);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    cudaMemcpy(output_data, device_output_data, input_byte_size, cudaMemcpyDeviceToHost);

    int gpu_sum = 0;
    for (int i = 0; i < grid_size; i++) {
        gpu_sum += output_data[i];
    }
    printf("GPU sum is %d\n", gpu_sum);

    int cpu_sum = 0;
    for (int i = 0; i < input_size; i++) {
        cpu_sum += input_data[i];
    }
    printf("CPU sum is %d\n", cpu_sum);

    CHECK_CUDA_ERROR(cudaDeviceReset());
    return 0;
}
