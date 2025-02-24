#include "cuda_kernels.h"
#include "address_matcher.h"
#include "key_generator.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

// Configuration
constexpr int THREADS_PER_BLOCK = 1024;
constexpr int KEYS_PER_THREAD = 16;
constexpr int UNROLL_FACTOR = 4;

// Global counters (managed memory for host-device access)
__device__ __managed__ uint64_t global_key_counter = 0;
__device__ __managed__ bool key_found_flag = false;
__device__ __managed__ uint8_t found_private_key[32];

// CUDA-optimized Secp256k1 implementation
#include "secp256k1.cuh"

//-------------------------------------------------------------
// Device Function: Generate Address from Private Key
//-------------------------------------------------------------
__device__ void generate_address(const uint8_t* priv_key, char* address) {
    secp256k1_pubkey pubkey;
    secp256k1_ec_pubkey_create(priv_key, &pubkey);

    uint8_t pub_serialized[33];
    secp256k1_pubkey_serialize(&pubkey, pub_serialized);

    uint8_t hash160[20];
    hash160_rmdsha(pub_serialized, 33, hash160);

    base58check_encode(0x00, hash160, 20, address);
}

//-------------------------------------------------------------
// Device Function: Hash160 (SHA256 + RIPEMD160)
//-------------------------------------------------------------
__device__ void hash160_rmdsha(const uint8_t* data, size_t len, uint8_t* out) {
    // Optimized CUDA implementation of SHA256 + RIPEMD160
    // TODO: Implement fast CUDA-based hash function
}

//-------------------------------------------------------------
// CUDA Kernel: Brute Force Key Search
//-------------------------------------------------------------
__global__ void crack_kernel(uint64_t start_key, uint64_t total_keys) {
    const uint64_t thread_start = start_key + 
        (blockIdx.x * blockDim.x + threadIdx.x) * KEYS_PER_THREAD;
    
    uint64_t local_count = 0;
    uint8_t priv_key[32] = {0};
    
    __shared__ char shared_address[THREADS_PER_BLOCK][ADDRESS_LENGTH];
    char* address = shared_address[threadIdx.x];

    for (int i = 0; i < KEYS_PER_THREAD; i += UNROLL_FACTOR) {
        #pragma unroll
        for (int u = 0; u < UNROLL_FACTOR; u++) {
            const uint64_t current_key = thread_start + i * blockDim.x + u;
            
            memcpy(priv_key, &current_key, 8);
            memset(priv_key + 8, 0, 24);

            generate_address(priv_key, address);

            if (AddressMatcher::bloomCheck(address)) {
                if (AddressMatcher::compare(address)) {
                    atomicExch(&key_found_flag, true);
                    memcpy(found_private_key, priv_key, 32);
                }
            }

            local_count++;
        }
    }

    atomicAdd(&global_key_counter, local_count);
}

//-------------------------------------------------------------
// Host Functions
//-------------------------------------------------------------

// Launches the CUDA kernel
void launch_cuda_kernel(uint64_t start_key, uint64_t batch_size) {
    dim3 blocks((batch_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    dim3 threads(THREADS_PER_BLOCK);

    crack_kernel<<<blocks, threads>>>(start_key, batch_size);
    cudaDeviceSynchronize();
}

// Checks if a matching private key was found
bool check_results(uint8_t* private_key) {
    if (key_found_flag) {
        cudaMemcpy(private_key, found_private_key, 32, cudaMemcpyDeviceToHost);
        return true;
    }
    return false;
}

// Returns the progress of key searches
uint64_t get_progress() {
    return global_key_counter;
}

// Resets counters before a new batch
void reset_counters() {
    global_key_counter = 0;
    key_found_flag = false;
    cudaMemset(found_private_key, 0, 32);
}
