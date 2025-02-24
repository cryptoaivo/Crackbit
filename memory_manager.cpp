#include "memory_manager.h"
#include <cuda_runtime.h>
#include <list>
#include <mutex>
#include <algorithm>

struct MemoryBlock {
    void* ptr;
    size_t size;
    bool in_use;
};

static std::list<MemoryBlock> memory_pool;
static std::mutex pool_mutex;
static size_t total_allocated = 0;

void* gpu_malloc(size_t size) {
    constexpr size_t ALIGNMENT = 256;
    size = (size + ALIGNMENT - 1) & ~(ALIGNMENT - 1);

    std::lock_guard<std::mutex> lock(pool_mutex);

    // Search for reusable block
    auto it = std::find_if(memory_pool.begin(), memory_pool.end(),
        [size](const MemoryBlock& block) {
            return !block.in_use && block.size >= size;
        });

    if (it != memory_pool.end()) {
        it->in_use = true;
        return it->ptr;
    }

    // Allocate new block
    void* ptr = nullptr;
    cudaError_t err = cudaMallocManaged(&ptr, size);
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA malloc failed: " + std::string(cudaGetErrorString(err)));
    }

    memory_pool.push_back({ptr, size, true});
    total_allocated += size;
    
    return ptr;
}

void gpu_free(void* ptr) {
    if (!ptr) return;

    std::lock_guard<std::mutex> lock(pool_mutex);

    auto it = std::find_if(memory_pool.begin(), memory_pool.end(),
        [ptr](const MemoryBlock& block) { return block.ptr == ptr; });

    if (it != memory_pool.end()) {
        it->in_use = false;
    }
}

void memory_cleanup() {
    std::lock_guard<std::mutex> lock(pool_mutex);

    for (auto& block : memory_pool) {
        if (block.ptr) {
            cudaFree(block.ptr);
            block.ptr = nullptr;
        }
    }

    memory_pool.clear();
    total_allocated = 0;
}

size_t gpu_memory_usage() {
    std::lock_guard<std::mutex> lock(pool_mutex);
    return total_allocated;
}
