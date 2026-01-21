#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <chrono>

#define MAX_VALUE 10000000
#define BUCKET_BITS 8


uint32_t float_to_sortable(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));
    // If sign bit is set (negative), flip all bits
    // If sign bit is clear (positive), flip only sign bit
    uint32_t mask = (bits & 0x80000000u) ? 0xFFFFFFFFu : 0x80000000u;
    return bits ^ mask;
}

float sortable_to_float(uint32_t bits) {
    // Reverse the transformation
    uint32_t mask = (bits & 0x80000000u) ? 0x80000000u : 0xFFFFFFFFu;
    bits = bits ^ mask;
    float f;
    memcpy(&f, &bits, sizeof(float));
    return f;
}

typedef struct {
    uint32_t* data;
    int length;
} UintArray;


UintArray compute_topk_recursive(
    uint32_t* array,
    uint32_t* bucketsArray,
    uint32_t* bucketStarts,
    uint32_t* bucketEnds,
    uint32_t N,
    uint32_t K,
    uint32_t pass
) {
    uint32_t bucketCount = 1 << BUCKET_BITS;

    // using bucketStarts as bucketCounts here
    for (uint32_t i = 0; i < N; i++) {
        uint32_t number = array[i];
        uint32_t bucket = (number >> (pass * BUCKET_BITS)) & (bucketCount - 1);
        bucketStarts[bucket] += 1;
    }

    uint32_t cumSum = 0;
    for (uint32_t bucket = 0; bucket < bucketCount; bucket++) {
        cumSum += bucketStarts[bucket];
        bucketEnds[bucket] = cumSum;
    }

    // Now bucket ends are actually bucket ends
    // we need to adjust the bucket starts
    // Currently bucket starts are [countOfBucket0, countOfBucket1, countOfBucket2, ..., etc.]
    // We basically just need to subtract bucketEnds[i] - bucketStarts[i] to get real bucketStarts
    for (uint32_t bucket = 0; bucket < bucketCount; bucket++) {
        bucketStarts[bucket] = bucketEnds[bucket] - bucketStarts[bucket];
        bucketEnds[bucket] = bucketStarts[bucket];
    }

    // Now we can pass through the numbers and bucket them
    for (uint32_t i = 0; i < N; i++) {
        uint32_t number = array[i];
        uint32_t bucket = (number >> (pass * BUCKET_BITS)) & (bucketCount - 1);
        bucketsArray[bucketEnds[bucket]] = number;
        bucketEnds[bucket] += 1;
    }

    UintArray result;
    result.length = 0;
    result.data = (uint32_t*)malloc(K * sizeof(uint32_t)); // allocate K items, might be shorter

    // Now we scan from the highest bucket downward
    for (uint32_t bucket = bucketCount; bucket-- > 0; ) {
        uint32_t countOfThisBucket = bucketEnds[bucket] - bucketStarts[bucket];
        if (countOfThisBucket == 0) continue;

        // Now we have a non-empty bucket
        if (result.length + countOfThisBucket <= K) {
            for (uint32_t i = 0; i < countOfThisBucket; i++) {
                result.data[result.length + i] = bucketsArray[bucketStarts[bucket] + i];
            }
            result.length += countOfThisBucket;

            if (result.length == K) return result;
        } else {
            uint32_t needed = K - result.length;

            if (pass == 0) {
                for (uint32_t i = 0; i < needed; i++) {
                    result.data[result.length + i] = bucketsArray[bucketStarts[bucket] + i];
                }
                result.length += needed;
                return result;
            }

            uint32_t bucketStart = bucketStarts[bucket];
            memset(bucketStarts, 0, bucketCount * sizeof(uint32_t));
            memset(bucketEnds, 0, bucketCount * sizeof(uint32_t));
            UintArray subResult = compute_topk_recursive(&bucketsArray[bucketStart], array, bucketStarts, bucketEnds, countOfThisBucket, needed, pass - 1);
            for (uint32_t i = 0; i < subResult.length; i++) {
                result.data[result.length + i] = subResult.data[i];
            }
            result.length += subResult.length;
            free(subResult.data);
            return result;
        }
    }

    return result;
}

typedef struct {
    float* data;
    int length;
} FloatArray;

FloatArray compute_topk(float* array, uint32_t N, uint32_t K) {
    FloatArray finalResult;
    if (N <= K) {
        finalResult.length = N;
        finalResult.data = (float*)malloc(N * sizeof(float));
        for (uint32_t i = 0; i < N; i++) finalResult.data[i] = array[i];
        return finalResult;
    }

    uint32_t* arrayTransformed = (uint32_t*)malloc(N * sizeof(uint32_t));
    for (uint32_t i = 0; i < N; i++) arrayTransformed[i] = float_to_sortable(array[i]);

    uint32_t* bucketsArray = (uint32_t*)malloc(N * sizeof(uint32_t));

    uint32_t* bucketStarts = (uint32_t*)calloc(1 << BUCKET_BITS, sizeof(uint32_t));
    uint32_t* bucketEnds = (uint32_t*)calloc(1 << BUCKET_BITS, sizeof(uint32_t));

    UintArray result = compute_topk_recursive(arrayTransformed, bucketsArray, bucketStarts, bucketEnds, N, K, 3);

    finalResult.length = result.length;
    finalResult.data = (float*)malloc(result.length * sizeof(float));

    for (uint32_t i = 0; i < result.length; i++) finalResult.data[i] = sortable_to_float(result.data[i]);

    free(result.data);
    free(bucketEnds);
    free(bucketStarts);
    free(bucketsArray);
    free(arrayTransformed);

    return finalResult;
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


int radix_select(uint32_t N, uint32_t K) {
    printf("=== Radix Select TopK (CPU) ===\n");
    printf("N = %d, K = %d\n\n", N, K);

    // Step 1: Data preparation
    auto start = std::chrono::high_resolution_clock::now();

    float* array = (float*)malloc(N * sizeof(float));
    for (uint32_t i = 0; i < N; i++) {
        array[i] = (float)(rand() % MAX_VALUE);
    }

    auto end = std::chrono::high_resolution_clock::now();
    double milliseconds = std::chrono::duration<double, std::milli>(end - start).count();
    printf("Step 1 - Data preparation: %.3f ms\n", milliseconds);

    // Step 2: Compute top-K using radix select
    start = std::chrono::high_resolution_clock::now();

    FloatArray result = compute_topk(array, N, K);

    end = std::chrono::high_resolution_clock::now();
    milliseconds = std::chrono::duration<double, std::milli>(end - start).count();
    printf("Step 2 - Radix select: %.3f ms\n\n", milliseconds);

    verify_result(array, result.data, N, K);

    free(result.data);
    free(array);

    return 0;
}


int main(int argc, char* argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <N> <K>\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    int K = atoi(argv[2]);

    if (N < 0 || K < 0) {
        printf("Cannot start radix select with N=%d, K=%d. Both numbers must be positive\n", N, K);    
    }

    radix_select(N, K);

    return 0;
}
