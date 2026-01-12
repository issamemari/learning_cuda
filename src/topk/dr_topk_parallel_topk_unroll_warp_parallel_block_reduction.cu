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

# define FIND_LOCAL_MAX_BLOCK_SIZE 1024
# define BLOCK_REDUCTION_BLOCK_SIZE 1024

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
}

__global__ void find_delegates(float *A, float *D, int RW, int N) {
    int subrangeID = blockDim.x * blockIdx.x + threadIdx.x;
    int rangeStart = RW * subrangeID;
    int rangeEnd = min(RW * (subrangeID + 1), N);

    float delegate = A[rangeStart];
    for (int i = rangeStart + 1; i < rangeEnd; i++) {
        if (A[i] > delegate) {
            delegate = A[i];
        }
    }

    D[subrangeID] = delegate;
}

__global__ void findLocalMax(float* array, float* blockMaxs, int* blockMaxIndices, int offset, int N) {
    extern __shared__ char sharedMem[];
    float* sharedArray = (float*)sharedMem;
    int* sharedIndices = (int*)(sharedMem + blockDim.x * sizeof(float));

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

    // Standard reduction until we hit one warp
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (threadId < stride) {
            if (sharedArray[threadId] < sharedArray[threadId + stride]) {
                sharedArray[threadId] = sharedArray[threadId + stride];
                sharedIndices[threadId] = sharedIndices[threadId + stride];
            }
        }
        __syncthreads();
    }

    if (threadId < 32) {
        if (blockDim.x >= 64 && sharedArray[threadId] < sharedArray[threadId + 32]) {
            sharedArray[threadId] = sharedArray[threadId + 32];
            sharedIndices[threadId] = sharedIndices[threadId + 32];
        }
        if (sharedArray[threadId] < sharedArray[threadId + 16]) {
            sharedArray[threadId] = sharedArray[threadId + 16];
            sharedIndices[threadId] = sharedIndices[threadId + 16];
        }
        if (sharedArray[threadId] < sharedArray[threadId + 8]) {
            sharedArray[threadId] = sharedArray[threadId + 8];
            sharedIndices[threadId] = sharedIndices[threadId + 8];
        }
        if (sharedArray[threadId] < sharedArray[threadId + 4]) {
            sharedArray[threadId] = sharedArray[threadId + 4];
            sharedIndices[threadId] = sharedIndices[threadId + 4];
        }
        if (sharedArray[threadId] < sharedArray[threadId + 2]) {
            sharedArray[threadId] = sharedArray[threadId + 2];
            sharedIndices[threadId] = sharedIndices[threadId + 2];
        }
        if (sharedArray[threadId] < sharedArray[threadId + 1]) {
            sharedArray[threadId] = sharedArray[threadId + 1];
            sharedIndices[threadId] = sharedIndices[threadId + 1];
        }
    }

    if (threadId == 0) {
        blockMaxs[blockIdx.x] = sharedArray[0];
        blockMaxIndices[blockIdx.x] = sharedIndices[0];
    }
}

__global__ void reduceBlockMaxs(float* blockMaxs, int* blockMaxIndices, 
                                 int* globalMaxIdx, int numBlocks) {
    __shared__ float sMaxVals[BLOCK_REDUCTION_BLOCK_SIZE];
    __shared__ int sMaxIndices[BLOCK_REDUCTION_BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int stride = blockDim.x;
    
    // Each thread finds max over its portion
    float maxVal = -FLT_MAX;
    int maxIdx = -1;
    
    for (int i = tid; i < numBlocks; i += stride) {
        if (blockMaxs[i] > maxVal) {
            maxVal = blockMaxs[i];
            maxIdx = blockMaxIndices[i];
        }
    }
    
    sMaxVals[tid] = maxVal;
    sMaxIndices[tid] = maxIdx;
    __syncthreads();
    
    // Tree reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sMaxVals[tid + s] > sMaxVals[tid]) {
                sMaxVals[tid] = sMaxVals[tid + s];
                sMaxIndices[tid] = sMaxIndices[tid + s];
            }
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        *globalMaxIdx = sMaxIndices[0];
    }
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

    cudaMalloc(&deviceBlockMaxs, N * sizeof(float));  // Allocate max possible size upfront
    cudaMalloc(&deviceBlockMaxIndices, N * sizeof(int));
    cudaMalloc(&deviceGlobalMaxIdx, sizeof(int));

    for (int i = 0; i < K; i++) {
        int remaining = N - i;
        int numBlocks = (remaining + FIND_LOCAL_MAX_BLOCK_SIZE - 1) / FIND_LOCAL_MAX_BLOCK_SIZE;

        size_t sharedMemSize = FIND_LOCAL_MAX_BLOCK_SIZE * (sizeof(float) + sizeof(int));
        findLocalMax<<<numBlocks, FIND_LOCAL_MAX_BLOCK_SIZE, sharedMemSize>>>(deviceArray, deviceBlockMaxs, deviceBlockMaxIndices, i, N);

        reduceBlockMaxs<<<1, BLOCK_REDUCTION_BLOCK_SIZE>>>(deviceBlockMaxs, deviceBlockMaxIndices, deviceGlobalMaxIdx, numBlocks);

        doSwap<<<1, 1>>>(deviceArray, deviceIndices, i, deviceGlobalMaxIdx);
    }

    cudaFree(deviceBlockMaxs);
    cudaFree(deviceBlockMaxIndices);
    cudaFree(deviceGlobalMaxIdx);
}

__global__ void concatenate_ranges(float* deviceArray, int* deviceDelegatesTopKRangeIDs, int K, int RW, int N, float* deviceConcatenated) {
    int outputIndex = blockDim.x * blockIdx.x + threadIdx.x;
    if (outputIndex >= RW * K) return;

    int rangeIndexInOutputArray = outputIndex / RW;
    int offsetInRange = outputIndex % RW;

    deviceConcatenated[outputIndex] = deviceArray[deviceDelegatesTopKRangeIDs[rangeIndexInOutputArray] * RW + offsetInRange];
}

int dr_topk(int N, int K) {
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    float milliseconds = 0;

    printf("=== Algorithm Performance Report ===\n");
    printf("N = %d, K = %d\n\n", N, K);

    // Step 1: Data preparation
    CHECK_CUDA_ERROR(cudaEventRecord(start));

    size_t bytesArray = N * sizeof(float);
    float *hostArray = (float*)malloc(bytesArray);
    for (int i = 0; i < N; i++) {
        hostArray[i] = (float)(rand() % 10000000);
    }
    float *deviceArray;
    CHECK_CUDA_ERROR(cudaMalloc(&deviceArray, bytesArray));
    CHECK_CUDA_ERROR(cudaMemcpy(deviceArray, hostArray, bytesArray, cudaMemcpyHostToDevice));

    size_t bytest_delegatesTopK = K * sizeof(float);
    float* hostDelegatesTopK = (float*)malloc(bytest_delegatesTopK);
    float* deviceDelegatesTopK;
    CHECK_CUDA_ERROR(cudaMalloc(&deviceDelegatesTopK, bytest_delegatesTopK));

    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Step 1 - Data preparation & H2D copy: %.3f ms\n", milliseconds);

    // Step 2: Find delegates
    int RW = 32;
    int delegateCount = (N + RW - 1) / RW;

    dim3 blockDim(1024);
    dim3 gridDim((delegateCount + blockDim.x - 1) / blockDim.x);

    size_t bytesDelegates = delegateCount * sizeof(float);
    float *deviceDelegates;
    CHECK_CUDA_ERROR(cudaMalloc(&deviceDelegates, bytesDelegates));

    CHECK_CUDA_ERROR(cudaEventRecord(start));
    find_delegates<<<gridDim, blockDim>>>(deviceArray, deviceDelegates, RW, N);
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Step 2 - Find delegates (RW=%d, threads=%d): %.3f ms\n", RW, delegateCount, milliseconds);

    // Step 3: Prepare delegate indices
    CHECK_CUDA_ERROR(cudaEventRecord(start));

    size_t bytesDelegatesIndices = delegateCount * sizeof(int);
    int* hostDelegatesIndices = (int*)malloc(bytesDelegatesIndices);
    // initialize indices
    for (int i = 0; i < delegateCount; i++) {
        hostDelegatesIndices[i] = i;
    }
    int* deviceDelegatesIndices;
    CHECK_CUDA_ERROR(cudaMalloc(&deviceDelegatesIndices, bytesDelegatesIndices));
    CHECK_CUDA_ERROR(cudaMemcpy(deviceDelegatesIndices, hostDelegatesIndices, bytesDelegatesIndices, cudaMemcpyHostToDevice));

    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Step 3 - Prepare delegate indices & H2D copy: %.3f ms\n", milliseconds);

    // Step 4: Find top-K from delegates

    dim3 blockDimFirstFindTopK(1024);
    dim3 gridDimFirstFindTopK((delegateCount * delegateCount + blockDimFirstFindTopK.x - 1) / blockDimFirstFindTopK.x);

    CHECK_CUDA_ERROR(cudaEventRecord(start));
    find_topk(deviceDelegates, deviceDelegatesIndices, delegateCount, K);
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Step 4 - Find top-K from delegates: %.3f ms\n", milliseconds);

    // Step 5: Concatenate corresponding ranges to run top-K on them
    int concatenatedSize = K * RW;
    size_t bytesConcatenated = concatenatedSize * sizeof(float);
    float* deviceConcatenated;
    CHECK_CUDA_ERROR(cudaMalloc(&deviceConcatenated, bytesConcatenated));

    dim3 blockDimConcat(1024);
    dim3 gridDimConcat((concatenatedSize + blockDimConcat.x - 1) / blockDimConcat.x);

    CHECK_CUDA_ERROR(cudaEventRecord(start));
    concatenate_ranges<<<gridDimConcat, blockDimConcat>>>(deviceArray, deviceDelegatesIndices, K, RW, N, deviceConcatenated);
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Step 5 - Concatenate top-K ranges: %.3f ms\n", milliseconds);

    // Step 6: Perform top-K on concatenated ranges
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    find_topk(deviceConcatenated, NULL, RW * K, K);
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Step 6 - Final top-K on concatenated ranges: %.3f ms\n", milliseconds);

    // Step 7: Copy results back
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    CHECK_CUDA_ERROR(cudaMemcpy(hostDelegatesTopK, deviceConcatenated, bytest_delegatesTopK, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Step 7 - D2H copy: %.3f ms\n\n", milliseconds);

    verify_result(hostArray, hostDelegatesTopK, N, K);

    free(hostDelegatesTopK);
    free(hostArray);

    CHECK_CUDA_ERROR(cudaFree(deviceDelegates));
    CHECK_CUDA_ERROR(cudaFree(deviceArray));
    CHECK_CUDA_ERROR(cudaFree(deviceDelegatesTopK));
    CHECK_CUDA_ERROR(cudaFree(deviceConcatenated));

    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));

    CHECK_CUDA_ERROR(cudaDeviceReset());

    return 0;
}


int main(int argc, char* argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <N> <K>\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    int K = atoi(argv[2]);

    printf("Running dr_topk with N=%d, K=%d\n", N, K);
    dr_topk(N, K);

    return 0;
}