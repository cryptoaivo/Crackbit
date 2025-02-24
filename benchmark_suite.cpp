#include "benchmark_suite.h"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <thread>
#include <atomic>
#include <algorithm>

using namespace std::chrono;

void BenchmarkSuite::run_tests() {
    constexpr size_t TEST_ITERATIONS = 10;
    constexpr size_t BATCH_SIZES[] = {1e6, 1e7, 1e8};
    
    std::ofstream report("benchmark.csv");
    if (!report.is_open()) {
        throw std::runtime_error("Failed to open benchmark report file.");
    }
    report << "Batch Size,Keys/s,Memory Usage,GPU Utilization\n";
    
    for (auto batch : BATCH_SIZES) {
        double total_time = 0;
        size_t max_memory = 0;
        
        for (int i = 0; i < TEST_ITERATIONS; i++) {
            auto start = high_resolution_clock::now();
            
            // Run kernel
            launch_cuda_kernel(0, batch);
            cudaDeviceSynchronize();
            
            auto end = high_resolution_clock::now();
            total_time += duration_cast<nanoseconds>(end - start).count();
            
            max_memory = std::max(max_memory, gpu_memory_usage());
        }
        
        double avg_time = total_time / TEST_ITERATIONS / 1e9;
        double keys_per_sec = batch / avg_time;
        
        report << batch << ","
               << keys_per_sec << ","
               << max_memory << ","
               << get_gpu_utilization() << "\n";
    }
}

void BenchmarkSuite::stress_test() {
    constexpr int STRESS_ITERATIONS = 100;
    constexpr size_t MAX_BATCH = 1e9;
    
    std::vector<std::thread> workers;
    std::atomic<size_t> total_keys{0};
    
    auto worker = [&](int seed) {
        for (int i = 0; i < STRESS_ITERATIONS; i++) {
            size_t batch = MAX_BATCH * (seed + i) / (STRESS_ITERATIONS * 4);
            launch_cuda_kernel(0, batch);
            total_keys += batch;
        }
    };
    
    // Launch parallel workers
    for (int i = 0; i < 4; i++) {
        workers.emplace_back(worker, i);
    }
    
    // Monitor resources
    auto start = high_resolution_clock::now();
    while (std::any_of(workers.begin(), workers.end(),
               [](auto& t) { return t.joinable(); })) {
        print_status(total_keys, start);
        std::this_thread::sleep_for(1s);
    }
    
    // Generate final report
    generate_heatmap("stress_test.csv");
}

void BenchmarkSuite::print_status(std::atomic<size_t>& total_keys, const time_point<high_resolution_clock>& start) {
    auto elapsed = duration_cast<seconds>(high_resolution_clock::now() - start);
    size_t keys_processed = total_keys.load();
    std::cout << "Elapsed time: " << elapsed.count() << "s, Keys processed: " 
              << keys_processed << ", Keys per second: " 
              << (elapsed.count() > 0 ? keys_processed / elapsed.count() : 0) << std::endl;
}
