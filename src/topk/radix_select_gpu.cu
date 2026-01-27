#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_VALUE 10000.0
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
        printf("Bucket counts verification PASSED!\n");
    }
}

void verify_bucket_pointers(uint32_t *gpu_counts, uint32_t *gpu_starts, uint32_t *gpu_ends) {
    uint32_t cpu_starts[NUM_BUCKETS];
    uint32_t cpu_ends[NUM_BUCKETS];

    // Compute prefix sum on CPU
    cpu_starts[0] = 0;
    cpu_ends[0] = gpu_counts[0];
    for (int i = 1; i < NUM_BUCKETS; i++) {
        cpu_starts[i] = cpu_ends[i - 1];
        cpu_ends[i] = cpu_starts[i] + gpu_counts[i];
    }

    bool passed = true;
    for (int i = 0; i < NUM_BUCKETS; i++) {
        if (cpu_starts[i] != gpu_starts[i]) {
            fprintf(stderr, "Bucket pointers FAILED! bucketStarts[%d]: CPU=%u, GPU=%u\n", i, cpu_starts[i], gpu_starts[i]);
            passed = false;
        }
        if (cpu_ends[i] != gpu_ends[i]) {
            fprintf(stderr, "Bucket pointers FAILED! bucketEnds[%d]: CPU=%u, GPU=%u\n", i, cpu_ends[i], gpu_ends[i]);
            passed = false;
        }
    }

    if (passed) {
        printf("Bucket pointers verification PASSED!\n");
    }
}

void verify_topk(float *h_array, uint32_t N, float *h_result, uint32_t K) {
    // Sort the original array to find true top-K
    float *sorted = (float *)malloc(N * sizeof(float));
    memcpy(sorted, h_array, N * sizeof(float));

    // Simple selection sort to find K largest (inefficient but correct)
    for (uint32_t i = 0; i < K; i++) {
        uint32_t maxIdx = i;
        for (uint32_t j = i + 1; j < N; j++) {
            if (sorted[j] > sorted[maxIdx]) {
                maxIdx = j;
            }
        }
        float tmp = sorted[i];
        sorted[i] = sorted[maxIdx];
        sorted[maxIdx] = tmp;
    }

    // Now sorted[0..K-1] contains the K largest elements
    // Check that every element in h_result is in the top-K
    bool passed = true;
    float kthValue = sorted[K - 1];

    for (uint32_t i = 0; i < K; i++) {
        if (h_result[i] < kthValue) {
            fprintf(stderr, "TopK FAILED! Result[%u]=%.2f is less than Kth value %.2f\n", i, h_result[i], kthValue);
            passed = false;
        }
    }

    // Also verify we have exactly K elements and they all came from original array
    // (count check - each result element should appear in original)

    // Find min in result
    float minResult = h_result[0];
    for (uint32_t i = 1; i < K; i++) {
        if (h_result[i] < minResult) minResult = h_result[i];
    }

    if (passed) {
        printf("TopK verification PASSED! (min result: %.2f, Kth value: %.2f)\n", minResult, kthValue);
    }

    // Print CPU top-K elements (limit output for large K)
    uint32_t printLimit = (K <= 20) ? K : 20;
    printf("\nCPU Top-%u elements", K);
    if (K > printLimit) printf(" (showing first %u)", printLimit);
    printf(":\n");
    for (uint32_t i = 0; i < printLimit; i++) {
        printf("  [%u] %.2f\n", i, sorted[i]);
    }
    if (K > printLimit) {
        printf("  ... (%u more)\n", K - printLimit);
    }

    free(sorted);
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

// Filter elements that belong to a specific bucket into output array
__global__ void filter_bucket(float *inputArray, uint32_t N, float *output, uint32_t *outputCount, uint32_t targetBucket, uint32_t pass) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N) {
        uint32_t value = float_to_sortable(inputArray[index]);
        uint32_t bucket = (value >> (pass * BUCKET_BITS)) & (NUM_BUCKETS - 1);
        if (bucket == targetBucket) {
            uint32_t pos = atomicAdd(outputCount, 1);
            output[pos] = inputArray[index];
        }
    }
}

// Copy elements from buckets >= minBucket to result array
__global__ void copy_topk_buckets(float *inputArray, uint32_t N, float *result, uint32_t *resultCount, uint32_t minBucket, uint32_t pass) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N) {
        uint32_t value = float_to_sortable(inputArray[index]);
        uint32_t bucket = (value >> (pass * BUCKET_BITS)) & (NUM_BUCKETS - 1);
        if (bucket > minBucket) {
            uint32_t pos = atomicAdd(resultCount, 1);
            result[pos] = inputArray[index];
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
    float total_time = 0;

    // Step 1: Data preparation on host
    cudaEventRecord(start);

    float* h_array = (float *)malloc(N * sizeof(float));
    for (uint32_t i = 0; i < N; i++) {
        h_array[i] = (rand() / (double)RAND_MAX) * MAX_VALUE;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Data preparation: %.3f ms\n", milliseconds);

    // Allocate device memory
    cudaEventRecord(start);

    float* d_buffer1;      // First buffer
    float* d_buffer2;      // Second buffer (for ping-pong)
    float* d_current;      // Points to current data
    float* d_filtered;     // Points to output buffer
    float* d_result;       // Final top-K result
    uint32_t* d_bucketCounts;
    uint32_t* d_count;     // Single counter for atomics

    cudaMalloc(&d_buffer1, N * sizeof(float));
    cudaMalloc(&d_buffer2, N * sizeof(float));
    d_current = d_buffer1;
    d_filtered = d_buffer2;
    cudaMalloc(&d_result, K * sizeof(float));
    cudaMalloc(&d_bucketCounts, NUM_BUCKETS * sizeof(uint32_t));
    cudaMalloc(&d_count, sizeof(uint32_t));

    cudaMemcpy(d_buffer1, h_array, N * sizeof(float), cudaMemcpyHostToDevice);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Memory allocation and copy: %.3f ms\n\n", milliseconds);

    int threadsPerBlock = 1024;
    uint32_t currentN = N;
    uint32_t remainingK = K;
    uint32_t resultCount = 0;
    uint32_t h_bucketCounts[NUM_BUCKETS];

    // Process from MSB (pass 3) to LSB (pass 0)
    for (int pass = 3; pass >= 0 && remainingK > 0; pass--) {
        printf("=== Pass %d (N=%u, K=%u) ===\n", pass, currentN, remainingK);

        int blocksPerGrid = (currentN + threadsPerBlock - 1) / threadsPerBlock;

        // Count buckets
        cudaEventRecord(start);
        cudaMemset(d_bucketCounts, 0, NUM_BUCKETS * sizeof(uint32_t));
        count_buckets<<<blocksPerGrid, threadsPerBlock>>>(d_current, currentN, d_bucketCounts, pass);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        total_time += milliseconds;
        printf("  count_buckets: %.3f ms\n", milliseconds);

        // Copy counts to host to find boundary bucket
        cudaMemcpy(h_bucketCounts, d_bucketCounts, NUM_BUCKETS * sizeof(uint32_t), cudaMemcpyDeviceToHost);

        // Scan from highest bucket (255) down to find boundary
        uint32_t accumulated = 0;
        int boundaryBucket = -1;
        for (int b = NUM_BUCKETS - 1; b >= 0; b--) {
            if (accumulated + h_bucketCounts[b] >= remainingK) {
                boundaryBucket = b;
                break;
            }
            accumulated += h_bucketCounts[b];
        }

        printf("  Boundary bucket: %d (accumulated above: %u, bucket size: %u)\n",
               boundaryBucket, accumulated, h_bucketCounts[boundaryBucket]);

        // Copy elements from buckets ABOVE boundary to result (these are definitely top-K)
        if (accumulated > 0) {
            cudaEventRecord(start);
            cudaMemset(d_count, 0, sizeof(uint32_t));
            copy_topk_buckets<<<blocksPerGrid, threadsPerBlock>>>(
                d_current, currentN, d_result + resultCount, d_count, boundaryBucket, pass);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&milliseconds, start, stop);
            total_time += milliseconds;
            printf("  copy_topk_buckets: %.3f ms (%u elements)\n", milliseconds, accumulated);

            resultCount += accumulated;
            remainingK -= accumulated;
        }

        // If we still need more elements, filter the boundary bucket for next pass
        if (remainingK > 0 && boundaryBucket >= 0) {
            cudaEventRecord(start);
            cudaMemset(d_count, 0, sizeof(uint32_t));
            filter_bucket<<<blocksPerGrid, threadsPerBlock>>>(
                d_current, currentN, d_filtered, d_count, boundaryBucket, pass);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&milliseconds, start, stop);
            total_time += milliseconds;

            // Get the new size
            cudaMemcpy(&currentN, d_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);
            printf("  filter_bucket: %.3f ms (filtered to %u elements)\n", milliseconds, currentN);

            // Swap buffers
            float* tmp = d_current;
            d_current = d_filtered;
            d_filtered = tmp;
        }

        // If this is the last pass, copy remaining elements from boundary bucket to result
        if (pass == 0 && remainingK > 0) {
            // Just copy remainingK elements from current buffer
            cudaMemcpy(d_result + resultCount, d_current, remainingK * sizeof(float), cudaMemcpyDeviceToDevice);
            resultCount += remainingK;
            remainingK = 0;
        }

        printf("\n");
    }

    printf("Total kernel time: %.3f ms\n", total_time);
    printf("Result count: %u\n\n", resultCount);

    // Verify result
    // float* h_result = (float *)malloc(K * sizeof(float));
    // cudaMemcpy(h_result, d_result, K * sizeof(float), cudaMemcpyDeviceToHost);
    // verify_topk(h_array, N, h_result, K);

    // Print top-K elements (limit output for large K)
    // uint32_t printLimit = (K <= 20) ? K : 20;
    // printf("\nTop-%u elements", K);
    // if (K > printLimit) printf(" (showing first %u)", printLimit);
    // printf(":\n");
    // for (uint32_t i = 0; i < printLimit; i++) {
    //     printf("  [%u] %.2f\n", i, h_result[i]);
    // }
    // if (K > printLimit) {
    //     printf("  ... (%u more)\n", K - printLimit);
    // }

    // Cleanup
    free(h_array);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_buffer1);
    cudaFree(d_buffer2);
    cudaFree(d_result);
    cudaFree(d_bucketCounts);
    cudaFree(d_count);

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
