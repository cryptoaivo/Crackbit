#pragma once
#include <string>
#include <vector>

class BloomManager {
public:
    // Initializes the Bloom filter with a single target
    static void init(const std::string& target);
    
    // Precomputes Bloom filter entries for a batch of targets
    static void precompute(const std::vector<std::string>& targets);
    
    // Efficiently updates the Bloom filter with a new target
    static void optimized_update(const std::string& target);

    // Clears the Bloom filter (if necessary)
    static void reset();
};
