#include <stdint.h>
#include <stdio.h>


#define MAX_VALUE 10000000
#define BUCKET_BITS 8

uint32_t flip_msb_int2uint(int32_t x) {
    return x ^ 0x80000000u;
}

int32_t flib_msb_uint2int(uint32_t x) {
    return x ^ 0x80000000u;
}

typedef struct {
    int* data;
    int length;
} IntArray;


IntArray compute_topk_recursive(
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

    IntArray result;
    result.length = 0;
    result.data = (int32_t*)malloc(K * sizeof(uint32_t)); // allocate K items, might be shorter

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
            IntArray subResult = compute_topk_recursive(&bucketsArray[bucketStart], array, bucketStarts, bucketEnds, countOfThisBucket, needed, pass - 1);
            for (uint32_t i = 0; i < subResult.length; i++) {
                result.data[result.length + i] = subResult.data[i];
            }
            result.length += subResult.length;
            return result;
        }
    }

    return result;
}

IntArray compute_topk(int32_t* array, uint32_t N, uint32_t K) {
    IntArray finalResult;
    if (N <= K) {
        finalResult.length = N;
        finalResult.data = (int32_t*)malloc(N * sizeof(int32_t));
        for (int i = 0; i < N; i++) finalResult.data[i] = array[i];
        return finalResult;
    }

    uint32_t* arrayTransformed = (uint32_t*)malloc(N * sizeof(uint32_t));
    for (uint32_t i = 0; i < N; i++) arrayTransformed[i] = flip_msb_int2uint(array[i]);

    uint32_t* bucketsArray = (uint32_t*)malloc(N * sizeof(uint32_t));

    uint32_t* bucketStarts = (uint32_t*)calloc(1 << BUCKET_BITS, sizeof(uint32_t));
    uint32_t* bucketEnds = (uint32_t*)calloc(1 << BUCKET_BITS, sizeof(uint32_t));

    IntArray result = compute_topk_recursive(arrayTransformed, bucketsArray, bucketStarts, bucketEnds, N, K, 3);

    finalResult.length = result.length;
    finalResult.data = (int32_t*)malloc(result.length * sizeof(int32_t));

    for (uint32_t i = 0; i < result.length; i++) finalResult.data[i] = flib_msb_uint2int(result.data[i]);

    free(bucketsArray);
    free(arrayTransformed);

    return finalResult;
}


void verify_result(int32_t *A, int32_t *R, int N, int K) {
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
                    fprintf(stderr, "Verification FAILED! A[%d]=%d is greater than R[%d]=%d\n", i, A[i], j, R[j]);
                    return;
                }
            }
        }
    }
    printf("Verification PASSED!\n");
}


int radix_select(uint32_t N, uint32_t K) {
    int32_t* array = (int32_t*)malloc(N * sizeof(int32_t));
    for (uint32_t i = 0; i < N; i++) {
        array[i] = (int32_t)(rand() % MAX_VALUE) - MAX_VALUE / 2;
    }

    IntArray result = compute_topk(array, N, K);

    //verify_result(array, result.data, N, K);

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

    printf("Running radix select with N=%d, K=%d\n", N, K);
    radix_select(N, K);

    return 0;
}
