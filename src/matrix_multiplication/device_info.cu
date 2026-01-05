#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    printf("Found %d CUDA device(s)\n\n", deviceCount);
    
    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);
        
        printf("========================================\n");
        printf("Device %d: %s\n", dev, prop.name);
        printf("========================================\n\n");
        
        // Shared Memory Information (your main interest)
        printf("SHARED MEMORY:\n");
        printf("  Shared memory per block:        %zu bytes (%.2f KB)\n", 
               prop.sharedMemPerBlock, prop.sharedMemPerBlock / 1024.0);
        printf("  Shared memory per SM:           %zu bytes (%.2f KB)\n", 
               prop.sharedMemPerMultiprocessor, prop.sharedMemPerMultiprocessor / 1024.0);
        printf("  Shared memory per block (opt-in): %zu bytes (%.2f KB)\n\n",
               prop.sharedMemPerBlockOptin, prop.sharedMemPerBlockOptin / 1024.0);
        
        // Compute Capability
        printf("COMPUTE CAPABILITY:\n");
        printf("  Compute capability:             %d.%d\n\n", 
               prop.major, prop.minor);
        
        // Memory Information
        printf("GLOBAL MEMORY:\n");
        printf("  Total global memory:            %zu bytes (%.2f GB)\n", 
               prop.totalGlobalMem, prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  Memory clock rate:              %.2f GHz\n", 
               prop.memoryClockRate / 1e6);
        printf("  Memory bus width:               %d bits\n", 
               prop.memoryBusWidth);
        printf("  L2 cache size:                  %d bytes (%.2f KB)\n\n", 
               prop.l2CacheSize, prop.l2CacheSize / 1024.0);
        
        // Multiprocessor Information
        printf("MULTIPROCESSORS:\n");
        printf("  Number of SMs:                  %d\n", 
               prop.multiProcessorCount);
        printf("  Max threads per SM:             %d\n", 
               prop.maxThreadsPerMultiProcessor);
        printf("  Max threads per block:          %d\n", 
               prop.maxThreadsPerBlock);
        printf("  Max thread dimensions:          (%d, %d, %d)\n", 
               prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  Max grid dimensions:            (%d, %d, %d)\n\n", 
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        
        // Warp Information
        printf("WARP & REGISTERS:\n");
        printf("  Warp size:                      %d\n", 
               prop.warpSize);
        printf("  Registers per block:            %d\n", 
               prop.regsPerBlock);
        printf("  Registers per SM:               %d\n\n", 
               prop.regsPerMultiprocessor);
        
        // Constant Memory
        printf("CONSTANT MEMORY:\n");
        printf("  Total constant memory:          %zu bytes (%.2f KB)\n\n", 
               prop.totalConstMem, prop.totalConstMem / 1024.0);
        
        // Memory Access Features
        printf("MEMORY FEATURES:\n");
        printf("  Global memory coalescing:       %s\n", 
               (prop.major >= 2) ? "Yes" : "Limited");
        printf("  ECC enabled:                    %s\n", 
               prop.ECCEnabled ? "Yes" : "No");
        printf("  Unified addressing:             %s\n", 
               prop.unifiedAddressing ? "Yes" : "No");
        printf("  Managed memory:                 %s\n\n", 
               prop.managedMemory ? "Yes" : "No");
        
        // Concurrent Execution
        printf("CONCURRENCY:\n");
        printf("  Concurrent kernels:             %s\n", 
               prop.concurrentKernels ? "Yes" : "No");
        printf("  Async engine count:             %d\n", 
               prop.asyncEngineCount);
        printf("  Stream priorities:              %s\n\n", 
               prop.streamPrioritiesSupported ? "Yes" : "No");
        
        // Clock Speeds
        printf("CLOCK SPEEDS:\n");
        printf("  Clock rate:                     %.2f GHz\n\n", 
               prop.clockRate / 1e6);
    }
    
    return 0;
}