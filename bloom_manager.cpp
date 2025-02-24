#include "bloom_manager.h"
#include "address_matcher.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <array>
#include <omp.h>

// Internal constants matching device configuration
constexpr int BLOOM_FILTER_BITS = 4096;
constexpr std::array<uint32_t, 2> BLOOM_HASH_SEEDS = {0x15a4db35, 0x76a54d32};
constexpr int ADDRESS_LENGTH = 35;

// Cache for filter reuse
static std::string last_target;
static std::array<uint32_t, BLOOM_FILTER_BITS / 32> bloom_cache;

void BloomManager::init(const std::string& target) {
    // Validate input
    if (target.length() != ADDRESS_LENGTH) {
        throw std::invalid_argument("Invalid target address length");
    }

    // Avoid redundant updates
    if (target == last_target) return;
    last_target = target;

    // Reinitialize Bloom filter
    std::array<uint32_t, BLOOM_FILTER_BITS / 32> bloom_filter = {};

    // Parallel hash computation with OpenMP
    #pragma omp parallel for
    for (size_t i = 0; i < BLOOM_HASH_SEEDS.size(); i++) {
        uint32_t h = BLOOM_HASH_SEEDS[i];

        for (char c : target) {
            h = (h ^ static_cast<uint8_t>(c)) * 0x01000193;
        }

        const uint32_t pos = h % BLOOM_FILTER_BITS;
        const uint32_t mask = 1U << (pos % 32);

        // Atomic OR operation for thread safety
        #pragma omp atomic
        bloom_filter[pos / 32] |= mask;
    }

    // Copy to GPU memory
    cudaError_t cudaStatus = cudaMemcpyToSymbol(
        AddressMatcher::bloom_filter,
        bloom_filter.data(),
        BLOOM_FILTER_BITS / 8
    );

    if (cudaStatus != cudaSuccess) {
        throw std::runtime_error("Bloom filter copy failed: " +
                                 std::string(cudaGetErrorString(cudaStatus)));
    }

    // Update cache
    bloom_cache = bloom_filter;
}

void BloomManager::precompute(const std::vector<std::string>& targets) {
    std::array<uint32_t, BLOOM_FILTER_BITS / 32> combined_filter = {};

    // Batch process multiple targets
    for (const auto& target : targets) {
        if (target.length() != ADDRESS_LENGTH) continue;

        for (uint32_t seed : BLOOM_HASH_SEEDS) {
            uint32_t h = seed;
            for (char c : target) {
                h = (h ^ static_cast<uint8_t>(c)) * 0x01000193;
            }
            const uint32_t pos = h % BLOOM_FILTER_BITS;
            combined_filter[pos / 32] |= (1U << (pos % 32));
        }
    }

    // Asynchronous copy to GPU
    cudaMemcpyToSymbolAsync(
        AddressMatcher::bloom_filter,
        combined_filter.data(),
        BLOOM_FILTER_BITS / 8,
        cudaMemcpyHostToDevice
    );
}

void BloomManager::optimized_update(const std::string& target) {
    // Incremental update with cached filter
    std::array<uint32_t, BLOOM_FILTER_BITS / 32> new_filter = bloom_cache;

    for (uint32_t seed : BLOOM_HASH_SEEDS) {
        uint32_t h = seed;
        for (char c : target) {
            h = (h ^ static_cast<uint8_t>(c)) * 0x01000193;
        }
        const uint32_t pos = h % BLOOM_FILTER_BITS;
        new_filter[pos / 32] |= (1U << (pos % 32));
    }

    // Asynchronous GPU update
    cudaMemcpyToSymbolAsync(
        AddressMatcher::bloom_filter,
        new_filter.data(),
        BLOOM_FILTER_BITS / 8,
        cudaMemcpyHostToDevice
    );

    bloom_cache = new_filter;
    last_target = target;
}

void BloomManager::reset() {
    std::array<uint32_t, BLOOM_FILTER_BITS / 32> empty_filter = {};
    cudaMemcpyToSymbol(
        AddressMatcher::bloom_filter,
        empty_filter.data(),
        BLOOM_FILTER_BITS / 8
    );

    bloom_cache.fill(0);
    last_target.clear();
}
