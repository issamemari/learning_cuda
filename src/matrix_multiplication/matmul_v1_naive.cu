#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>

#define CHECK_CUDA_ERROR(call) { \
    cudaError_t err = call;     \
    if (err != cudaSuccess) {   \
        fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE);     \
    }                           \
}

// Naive matrix multiplication kernel
// Each thread computes one element of the output matrix C
// C = A * B
// A is MxK, B is KxN, C is MxN
__global__ void matmul_naive(float *A, float *B, float *C, int M, int N, int K) {
    // Calculate the row and column index of the C element this thread will compute
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check
    if (row < M && col < N) {
        float sum = 0.0f;

        // Compute dot product of row from A and column from B
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }

        // Write result to C
        C[row * N + col] = sum;
    }
}

void initialize_matrix(float *mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)(rand() % 10);
    }
}

void verify_result(float *A, float *B, float *C, int M, int N, int K) {
    // Check a few elements
    int errors = 0;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float expected = 0.0f;
            for (int k = 0; k < K; k++) {
                expected += A[i * K + k] * B[k * N + j];
            }
            float diff = fabs(expected - C[i * N + j]);
            if (diff > 0.01f) {
                errors++;
                if (errors < 5) {
                    printf("Error at C[%d][%d]: expected %.2f, got %.2f\n",
                           i, j, expected, C[i * N + j]);
                }
            }
        }
    }
    if (errors == 0) {
        //printf("Verification PASSED!\n");
    } else {
        printf("Verification FAILED with %d errors!\n", errors);
    }
}

int main() {
    // Matrix dimensions: C(M x N) = A(M x K) * B(K x N)
    int M = 500;  // rows of A and C
    int K = 500;   // cols of A, rows of B
    int N = 1024;  // cols of B and C

    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(bytes_A);
    float *h_B = (float*)malloc(bytes_B);
    float *h_C = (float*)malloc(bytes_C);

    // Initialize matrices
    initialize_matrix(h_A, M, K);
    initialize_matrix(h_B, K, N);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK_CUDA_ERROR(cudaMalloc(&d_A, bytes_A));
    CHECK_CUDA_ERROR(cudaMalloc(&d_B, bytes_B));
    CHECK_CUDA_ERROR(cudaMalloc(&d_C, bytes_C));

    // Copy data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));

    // Configure kernel launch parameters
    dim3 blockDim(16, 16);  // 16x16 = 256 threads per block
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (M + blockDim.y - 1) / blockDim.y);

    // Launch kernel and measure time
    matmul_naive<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);

    // Copy result back to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));

    verify_result(h_A, h_B, h_C, M, N, K);

    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);

    CHECK_CUDA_ERROR(cudaDeviceReset());

    return 0;
}
