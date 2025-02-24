#pragma once

#include <vector>
#include <thread>
#include <atomic>

class BenchmarkSuite {
public:
    // Run a series of benchmark tests on the kernel
    static void run_tests();

    // Perform a stress test with a high number of iterations
    static void stress_test();

private:
    // Helper function to print the status of the test
    static void print_status(const std::atomic<size_t>& total_keys, const std::chrono::high_resolution_clock::time_point& start);

    // Helper function to generate a heatmap from the results
    static void generate_heatmap(const std::string& filename);
};
