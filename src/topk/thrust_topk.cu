#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/functional.h>

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
    printf("Verification PASSED!\n");
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

    printf("=== Thrust Sort TopK ===\n");
    printf("N = %d, K = %d\n\n", N, K);

    // Step 1: Data preparation & H2D copy
    CHECK_CUDA_ERROR(cudaEventRecord(start));

    size_t bytesArray = N * sizeof(float);
    float *hostArray = (float*)malloc(bytesArray);
    for (int i = 0; i < N; i++) {
        hostArray[i] = (float)(rand() % 10000000);
    }

    float* deviceArray;
    CHECK_CUDA_ERROR(cudaMalloc(&deviceArray, bytesArray));
    CHECK_CUDA_ERROR(cudaMemcpy(deviceArray, hostArray, bytesArray, cudaMemcpyHostToDevice));

    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Step 1 - Data preparation & H2D copy: %.3f ms\n", milliseconds);

    // Step 2: Sort in descending order using thrust
    CHECK_CUDA_ERROR(cudaEventRecord(start));

    thrust::device_ptr<float> devicePtr(deviceArray);
    thrust::sort(devicePtr, devicePtr + N, thrust::greater<float>());

    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Step 2 - Thrust sort (descending): %.3f ms\n", milliseconds);

    // Step 3: Copy top-K results back
    size_t bytesTopK = K * sizeof(float);
    float* hostTopK = (float*)malloc(bytesTopK);

    CHECK_CUDA_ERROR(cudaEventRecord(start));

    CHECK_CUDA_ERROR(cudaMemcpy(hostTopK, deviceArray, bytesTopK, cudaMemcpyDeviceToHost));

    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Step 3 - D2H copy: %.3f ms\n\n", milliseconds);

    verify_result(hostArray, hostTopK, N, K);

    free(hostTopK);
    free(hostArray);
    CHECK_CUDA_ERROR(cudaFree(deviceArray));

    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    CHECK_CUDA_ERROR(cudaDeviceReset());

    return 0;
}
