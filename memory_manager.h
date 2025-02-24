#pragma once

#include <cstddef>

class MemoryManager {
public:
    // Allocate GPU memory with specified size
    static void* gpu_malloc(size_t size);

    // Free allocated GPU memory
    static void gpu_free(void* ptr);

    // Cleanup all allocated memory
    static void memory_cleanup();

    // Get total GPU memory usage
    static size_t gpu_memory_usage();

private:
    // Helper function to align memory size to 256 bytes
    static size_t align_size(size_t size);

    // Internal list and mutex management for memory blocks
    struct MemoryBlock {
        void* ptr;
        size_t size;
        bool in_use;
    };

    static std::list<MemoryBlock> memory_pool;
    static std::mutex pool_mutex;
    static size_t total_allocated;
};
