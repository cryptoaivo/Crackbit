#include "address_matcher.h"
#include <thrust/device_vector.h>
#include <cuda_runtime.h>
#include <iostream>

//-------------------------------------------------------------
// Device Constants
//-------------------------------------------------------------
__constant__ uint32_t AddressMatcher::bloom_filter[BLOOM_FILTER_BITS / 32];
__constant__ char AddressMatcher::target_address[ADDRESS_LENGTH];

//-------------------------------------------------------------
// Host Initialization
//-------------------------------------------------------------
void AddressMatcher::init(const char* target) {
    if (!target) {
        throw std::invalid_argument("Null target address in AddressMatcher::init().");
    }

    // Copy target address to device constant memory
    cudaMemcpyToSymbol(target_address, target, ADDRESS_LENGTH, 0, cudaMemcpyHostToDevice);

    // Initialize host Bloom filter
    uint32_t host_bloom[BLOOM_FILTER_BITS / 32] = {0};

    for (int seed : BLOOM_HASH_SEEDS) {
        uint32_t h = seed;
        for (int i = 0; i < ADDRESS_LENGTH; i++) {
            h = (h * 0x01000193) ^ static_cast<uint8_t>(target[i]);  // FNV-1a Hashing
        }
        h = h % BLOOM_FILTER_BITS;
        host_bloom[h / 32] |= (1 << (h % 32));
    }

    // Copy Bloom filter to device constant memory
    cudaMemcpyToSymbol(bloom_filter, host_bloom, sizeof(host_bloom), 0, cudaMemcpyHostToDevice);
}

//-------------------------------------------------------------
// Device Functions
//-------------------------------------------------------------

// Fast SIMD-based address comparison
__device__ bool AddressMatcher::compare(const char* candidate) {
    return simdCompare(candidate, target_address);
}

// Bloom filter check to quickly rule out non-matching addresses
__device__ bool AddressMatcher::bloomCheck(const char* candidate) {
    #pragma unroll
    for (int seed : BLOOM_HASH_SEEDS) {
        uint32_t h = hash(candidate, seed);
        if (!(bloom_filter[h / 32] & (1 << (h % 32)))) return false;
    }
    return true;
}

// Fast SIMD-based address comparison using uint4 loads
__device__ bool AddressMatcher::simdCompare(const char* a, const char* b) {
    const uint4* a4 = reinterpret_cast<const uint4*>(a);
    const uint4* b4 = reinterpret_cast<const uint4*>(b);

    // Compare first 32 bytes in parallel
    bool match = (a4[0].x == b4[0].x) && (a4[0].y == b4[0].y) &&
                 (a4[0].z == b4[0].z) && (a4[0].w == b4[0].w) &&
                 (a4[1].x == b4[1].x) && (a4[1].y == b4[1].y) &&
                 (a4[1].z == b4[1].z) && (a4[1].w == b4[1].w);

    // Compare remaining 3 bytes without branching
    return match && (__ldg(&a[32]) == __ldg(&b[32])) &&
                   (__ldg(&a[33]) == __ldg(&b[33])) &&
                   (__ldg(&a[34]) == __ldg(&b[34]));
}

// Fast hash function using FNV-1a for Bloom filter
__device__ uint32_t AddressMatcher::hash(const char* data, int seed) {
    uint32_t h = seed;
    #pragma unroll
    for (int i = 0; i < ADDRESS_LENGTH; i++) {
        h = (h ^ static_cast<uint8_t>(data[i])) * 0x01000193;
    }
    return h % BLOOM_FILTER_BITS;
}
