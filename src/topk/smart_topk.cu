#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR(call) { \
    cudaError_t err = call;     \
    if (err != cudaSuccess) {   \
        fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE);     \
    }                           \
}

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

__global__ void find_topk(float* array, int* indices, int N, int K) {
    for (int i = 0; i < K; i++) {
        int maxIdx = i;
        for (int j = i + 1; j < N; j++) {
            if (array[j] > array[maxIdx]) {
                maxIdx = j;
            }
        }

        float temp = array[i];
        array[i] = array[maxIdx];
        array[maxIdx] = temp;

        if (indices != NULL) {
            int tempIdx = indices[i];
            indices[i] = indices[maxIdx];
            indices[maxIdx] = tempIdx;
        }
    }
}

__global__ void concatenate_ranges(float* deviceArray, int* deviceDelegatesTopKRangeIDs, int K, int RW, int N, float* deviceConcatenated) {
    int outputIndex = blockDim.x * blockIdx.x + threadIdx.x;
    if (outputIndex >= RW * K) return;

    int rangeIndexInOutputArray = outputIndex / RW;
    int offsetInRange = outputIndex % RW;

    deviceConcatenated[outputIndex] = deviceArray[deviceDelegatesTopKRangeIDs[rangeIndexInOutputArray] * RW + offsetInRange];
}

int main() {
    int N = 1000000;
    int K = 10;

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

    dim3 blockDim(32);
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

    // Step 3: Find top-K delegates
    size_t bytesDelegatesIndices = delegateCount * sizeof(int);
    int* hostDelegatesIndices = (int*)malloc(bytesDelegatesIndices);
    // initialize indices
    for (int i = 0; i < delegateCount; i++) {
        hostDelegatesIndices[i] = i;
    }
    int* deviceDelegatesIndices;
    CHECK_CUDA_ERROR(cudaMalloc(&deviceDelegatesIndices, bytesDelegatesIndices));
    CHECK_CUDA_ERROR(cudaMemcpy(deviceDelegatesIndices, hostDelegatesIndices, bytesDelegatesIndices, cudaMemcpyHostToDevice));

    CHECK_CUDA_ERROR(cudaEventRecord(start));
    find_topk<<<1, 1>>>(deviceDelegates, deviceDelegatesIndices, delegateCount, K);
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Step 3 - Find top-K from delegates: %.3f ms\n", milliseconds);

    // Step 4: Concatenate corresponding ranges to run top-K on them
    int concatenatedSize = K * RW;
    size_t bytesConcatenated = concatenatedSize * sizeof(float);
    float* deviceConcatenated;
    CHECK_CUDA_ERROR(cudaMalloc(&deviceConcatenated, bytesConcatenated));

    dim3 blockDimConcat(256);
    dim3 gridDimConcat((concatenatedSize + blockDimConcat.x - 1) / blockDimConcat.x);

    CHECK_CUDA_ERROR(cudaEventRecord(start));
    concatenate_ranges<<<gridDimConcat, blockDimConcat>>>(deviceArray, deviceDelegatesIndices, K, RW, N, deviceConcatenated);
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Step 4 - Concatenate top-K ranges: %.3f ms\n", milliseconds);

    // Step 5: perform topk on concatenated ranges and put that in result
    find_topk<<<1, 1>>>(deviceConcatenated, NULL, RW * K, K);

    // Step 5: Copy results back
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    CHECK_CUDA_ERROR(cudaMemcpy(hostDelegatesTopK, deviceConcatenated, bytest_delegatesTopK, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Step 5 - D2H copy: %.3f ms\n\n", milliseconds);

    // print top K results
    printf("Top %d results:\n", K);
    for (int i = 0; i < K; i++) {
        printf("%.2f ", hostDelegatesTopK[i]);
    }
    printf("\n");

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
