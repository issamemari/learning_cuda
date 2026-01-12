#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <float.h>

#define CHECK_CUDA_ERROR(call) { \
    cudaError_t err = call;     \
    if (err != cudaSuccess) {   \
        fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE);     \
    }                           \
}

#define FIND_LOCAL_MAX_BLOCK_SIZE 1024

void verify_result(float *A, float *R, int N, int K) {
    for (int i = 0; i < N; i++) {
        bool is_topk = false;
        for (int j = 0; j < K; j++) {
            if (A[i] == R[j]) {
                is_topk = true;
                break;
            }
        }
        if (!is_topk) {
            for (int j = 0; j < K; j++) {
                if (A[i] > R[j]) {
                    fprintf(stderr, "Verification FAILED! A[%d]=%.2f is greater than R[%d]=%.2f\n", i, A[i], j, R[j]);
                    return;
                }
            }
        }
    }
    printf("Verification PASSED!\n");
}

__global__ void findLocalMax(float* array, float* blockMaxs, int* blockMaxIndices, int offset, int N) {
    __shared__ float sharedArray[FIND_LOCAL_MAX_BLOCK_SIZE];
    __shared__ int sharedIndices[FIND_LOCAL_MAX_BLOCK_SIZE];

    int threadId = threadIdx.x;
    int indexInArray = blockIdx.x * blockDim.x + threadIdx.x + offset;

    if (indexInArray < N) {
        sharedArray[threadId] = array[indexInArray];
        sharedIndices[threadId] = indexInArray;
    } else {
        sharedArray[threadId] = -FLT_MAX;
        sharedIndices[threadId] = -1;
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadId < stride) {
            if (sharedArray[threadId] < sharedArray[threadId + stride]) {
                sharedArray[threadId] = sharedArray[threadId + stride];
                sharedIndices[threadId] = sharedIndices[threadId + stride];
            }
        }
        __syncthreads();
    }

    if (threadId == 0) {
        blockMaxs[blockIdx.x] = sharedArray[0];
        blockMaxIndices[blockIdx.x] = sharedIndices[0];
    }
}

__global__ void reduceBlockMaxs(float* blockMaxs, int* blockMaxIndices, int* globalMaxIdx, int numBlocks) {
    float maxVal = -FLT_MAX;
    int maxIdx = -1;

    for (int i = 0; i < numBlocks; i++) {
        if (blockMaxs[i] > maxVal) {
            maxVal = blockMaxs[i];
            maxIdx = blockMaxIndices[i];
        }
    }

    *globalMaxIdx = maxIdx;
}

__global__ void doSwap(float* array, int* indices, int targetPos, int* globalMaxIdx) {
    int maxIdx = *globalMaxIdx;

    float tempVal = array[targetPos];
    array[targetPos] = array[maxIdx];
    array[maxIdx] = tempVal;

    if (indices != NULL) {
        int tempIdx = indices[targetPos];
        indices[targetPos] = indices[maxIdx];
        indices[maxIdx] = tempIdx;
    }
}

void find_topk(float* deviceArray, int* deviceIndices, int N, int K) {
    float* deviceBlockMaxs;
    int* deviceBlockMaxIndices;
    int* deviceGlobalMaxIdx;

    cudaMalloc(&deviceBlockMaxs, N * sizeof(float));
    cudaMalloc(&deviceBlockMaxIndices, N * sizeof(int));
    cudaMalloc(&deviceGlobalMaxIdx, sizeof(int));

    for (int i = 0; i < K; i++) {
        int remaining = N - i;
        int numBlocks = (remaining + FIND_LOCAL_MAX_BLOCK_SIZE - 1) / FIND_LOCAL_MAX_BLOCK_SIZE;

        findLocalMax<<<numBlocks, FIND_LOCAL_MAX_BLOCK_SIZE>>>(deviceArray, deviceBlockMaxs, deviceBlockMaxIndices, i, N);

        reduceBlockMaxs<<<1, 1>>>(deviceBlockMaxs, deviceBlockMaxIndices, deviceGlobalMaxIdx, numBlocks);

        doSwap<<<1, 1>>>(deviceArray, deviceIndices, i, deviceGlobalMaxIdx);
    }

    cudaFree(deviceBlockMaxs);
    cudaFree(deviceBlockMaxIndices);
    cudaFree(deviceGlobalMaxIdx);
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <N> <K>\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    int K = atoi(argv[2]);

    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    float milliseconds = 0;

    printf("=== Sequential Parallel Max TopK ===\n");
    printf("N = %d, K = %d\n\n", N, K);

    // Step 1: Data preparation & H2D copy
    CHECK_CUDA_ERROR(cudaEventRecord(start));

    size_t bytesArray = N * sizeof(float);
    float *hostArray = (float*)malloc(bytesArray);
    for (int i = 0; i < N; i++) {
        hostArray[i] = (float)(rand() % 10000000);
    }
    float *deviceArray;
    CHECK_CUDA_ERROR(cudaMalloc(&deviceArray, bytesArray));
    CHECK_CUDA_ERROR(cudaMemcpy(deviceArray, hostArray, bytesArray, cudaMemcpyHostToDevice));

    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Step 1 - Data preparation & H2D copy: %.3f ms\n", milliseconds);

    // Step 2: Find top-K using parallel max reduction
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    find_topk(deviceArray, NULL, N, K);
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Step 2 - Find top-K (parallel max, K iterations): %.3f ms\n", milliseconds);

    // Step 3: Copy results back
    size_t bytesTopK = K * sizeof(float);
    float* hostTopK = (float*)malloc(bytesTopK);

    CHECK_CUDA_ERROR(cudaEventRecord(start));
    CHECK_CUDA_ERROR(cudaMemcpy(hostTopK, deviceArray, bytesTopK, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Step 3 - D2H copy: %.3f ms\n\n", milliseconds);

    //verify_result(hostArray, hostTopK, N, K);

    free(hostTopK);
    free(hostArray);
    CHECK_CUDA_ERROR(cudaFree(deviceArray));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    CHECK_CUDA_ERROR(cudaDeviceReset());

    return 0;
}
