#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_VALUE 10000000
#define BUCKET_BITS 8
#define NUM_BUCKETS (1 << BUCKET_BITS)

__device__ uint32_t float_to_sortable(float f) {
    uint32_t bits = __float_as_uint(f);
    uint32_t mask = (bits & 0x80000000u) ? 0xFFFFFFFFu : 0x80000000u;
    return bits ^ mask;
}

__device__ float sortable_to_float(uint32_t bits) {
    uint32_t mask = (bits & 0x80000000u) ? 0x80000000u : 0xFFFFFFFFu;
    bits = bits ^ mask;
    return __uint_as_float(bits);
}

uint32_t host_float_to_sortable(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));
    uint32_t mask = (bits & 0x80000000u) ? 0xFFFFFFFFu : 0x80000000u;
    return bits ^ mask;
}

void verify_bucket_counts(float *h_array, uint32_t N, uint32_t *gpu_counts, uint32_t pass) {
    uint32_t cpu_counts[NUM_BUCKETS] = {0};

    for (uint32_t i = 0; i < N; i++) {
        uint32_t sortable = host_float_to_sortable(h_array[i]);
        uint32_t bucket = (sortable >> (pass * BUCKET_BITS)) & (NUM_BUCKETS - 1);
        cpu_counts[bucket]++;
    }

    bool passed = true;
    for (int i = 0; i < NUM_BUCKETS; i++) {
        if (cpu_counts[i] != gpu_counts[i]) {
            fprintf(stderr, "Verification FAILED! Bucket %d: CPU=%u, GPU=%u\n", i, cpu_counts[i], gpu_counts[i]);
            passed = false;
        }
    }

    if (passed) {
        printf("Verification PASSED!\n");
    }
}

__global__ void count_buckets(float *inputArray, uint32_t N, uint32_t *bucketCounts, uint32_t pass) {
    __shared__ uint32_t sharedBucketCounts[NUM_BUCKETS];

    uint32_t tid = threadIdx.x;
    for (uint32_t i = tid; i < NUM_BUCKETS; i += blockDim.x) {
        sharedBucketCounts[i] = 0;
    }
    __syncthreads();

    uint32_t index = blockIdx.x * blockDim.x + tid;
    if (index < N) {
        uint32_t value = float_to_sortable(inputArray[index]);
        uint32_t bucket = (value >> (pass * BUCKET_BITS)) & (NUM_BUCKETS - 1);
        atomicAdd(&sharedBucketCounts[bucket], 1);
    }
    __syncthreads();

    for (uint32_t i = tid; i < NUM_BUCKETS; i += blockDim.x) {
        if (sharedBucketCounts[i] > 0) {
            atomicAdd(&bucketCounts[i], sharedBucketCounts[i]);
        }
    }
}

int radix_select(uint32_t N, uint32_t K)
{
    printf("=== Radix Select TopK (GPU) ===\n");
    printf("N = %d, K = %d\n\n", N, K);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds;

    // Step 1: Data preparation on host
    cudaEventRecord(start);

    float *h_array = (float *)malloc(N * sizeof(float));
    for (uint32_t i = 0; i < N; i++) {
        h_array[i] = (float)(rand() % MAX_VALUE);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Step 1 - Data preparation: %.3f ms\n", milliseconds);

    // Step 2: Allocate device memory and copy data
    cudaEventRecord(start);

    float *d_input;
    uint32_t *d_bucketCounts;

    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_bucketCounts, NUM_BUCKETS * sizeof(uint32_t));

    cudaMemcpy(d_input, h_array, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_bucketCounts, 0, NUM_BUCKETS * sizeof(uint32_t));

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Step 2 - Memory allocation and copy: %.3f ms\n", milliseconds);

    // Step 3: Call count_buckets kernel
    cudaEventRecord(start);

    int threadsPerBlock = 1024;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    count_buckets<<<blocksPerGrid, threadsPerBlock>>>(d_input, N, d_bucketCounts, 3);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Step 3 - count_buckets kernel: %.3f ms\n", milliseconds);

    // Step 4: Verify results
    uint32_t *h_bucketCounts = (uint32_t *)malloc(NUM_BUCKETS * sizeof(uint32_t));
    cudaMemcpy(h_bucketCounts, d_bucketCounts, NUM_BUCKETS * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    verify_bucket_counts(h_array, N, h_bucketCounts, 3);
    free(h_bucketCounts);

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_bucketCounts);
    free(h_array);

    return 0;
}

int main(int argc, char *argv[]) {
    if (argc != 3)
    {
        fprintf(stderr, "Usage: %s <N> <K>\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    int K = atoi(argv[2]);

    if (N < 0 || K < 0)
    {
        printf("Cannot start radix select with N=%d, K=%d. Both numbers must be positive\n", N, K);
    }

    radix_select(N, K);

    return 0;
}
