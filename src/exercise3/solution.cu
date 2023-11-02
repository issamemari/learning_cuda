#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>


__device__ int getGlobalIdx3D() {
    int threadsPerBlock = blockDim.x * blockDim.y * blockDim.z;
    int blockOffset = blockIdx.z * gridDim.y * gridDim.x + blockIdx.y * gridDim.x + blockIdx.x;
    int threadOffset = blockOffset * threadsPerBlock;

    int gid = threadOffset + threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;

    return gid;
}

__global__ void sum3Arrays(int* array1, int* array2, int* array3, int* output, unsigned int size) {
    int gid = getGlobalIdx3D();

    if (gid < size)
        output[gid] = array1[gid] + array2[gid] + array3[gid];
}

__host__ void sum3ArraysCPU(int* array1, int* array2, int*array3, int* output, unsigned int size) {
    for (int i = 0; i < size; ++i) {
        output[i] = array1[i] + array2[i] + array3[i];
    }
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

int* moveArrayToDevice(int* h_array, unsigned int size) {
    int* d_array;
    cudaMalloc(&d_array, size * sizeof(int));
    cudaMemcpy(d_array, h_array, size * sizeof(int), cudaMemcpyHostToDevice);

    return d_array;
}


int main() {
    const long size = 1 << 28;

    // Seed the random number generator
    srand(static_cast<unsigned int>(time(nullptr)));

    int* h_array1 = createRandomArray(size);
    int* h_array2 = createRandomArray(size);
    int* h_array3 = createRandomArray(size);
    int* h_output = (int*)malloc(size * sizeof(int));
    int* actual_sum = (int*)malloc(size * sizeof(int));

    clock_t start_htod = clock();
    int* d_array1 = moveArrayToDevice(h_array1, size);
    int* d_array2 = moveArrayToDevice(h_array2, size);
    int* d_array3 = moveArrayToDevice(h_array3, size);
    int* d_output = moveArrayToDevice(h_output, size);
    clock_t end_htod = clock();

    int block_sizes[] = { 64, 128, 256, 512 };

    long* clock_cycles = (long*)malloc(4 * sizeof(long));

    for (int i = 0; i < 4; ++i) {
        dim3 block(block_sizes[i]);
        dim3 grid(size / block_sizes[i] + 1);

        clock_t start = clock();
        sum3Arrays<<<grid, block>>>(d_array1, d_array2, d_array3, d_output, size);
        cudaDeviceSynchronize();
        clock_t end = clock();

        clock_cycles[i] = end - start;
    }

    // Copy the result back to the host
    clock_t start_dtoh = clock();
    cudaMemcpy(h_output, d_output, size * sizeof(int), cudaMemcpyDeviceToHost);
    clock_t end_dtoh = clock();

    clock_t start_cpu = clock();
    sum3ArraysCPU(h_array1, h_array2, h_array3, actual_sum, size);
    clock_t end_cpu = clock();

    // Check if the result is correct
    for (int i = 0; i < size; ++i) {
        if (h_output[i] != actual_sum[i]) {
            printf("Error at index %d\n", i);
            break;
        }
    }

    // print cpu duration
    float cpu_duration = (float)(end_cpu - start_cpu) / CLOCKS_PER_SEC;
    printf("CPU duration: %4.6f\n", cpu_duration);

    // print host to device duration
    float htod_duration = (float)(end_htod - start_htod) / CLOCKS_PER_SEC;
    printf("Host to device duration: %4.6f\n", htod_duration);

    // print durations by dividing by CLOCKS_PER_SEC
    for (int i = 0; i < 4; ++i) {
        float duration = (float)clock_cycles[i] / CLOCKS_PER_SEC;
        printf("Block size: %d, duration: %4.6f\n", block_sizes[i], duration);
    }

    // print device to host duration
    float dtoh_duration = (float)(end_dtoh - start_dtoh) / CLOCKS_PER_SEC;
    printf("Device to host duration: %4.6f\n", dtoh_duration);

    cudaDeviceReset();
    return 0;
}
