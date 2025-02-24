#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <thread>
#include <cstdlib>
#include <cuda_runtime.h>
#include <openssl/bn.h>
#include <atomic>
#include <mutex>
#include "puzzle_config.h"

//----------- File Names -----------
const std::string CHECKPOINT_FILE = "checkpoint.dat";
const std::string LOG_FILE = "crackbit.log";
const std::string RESULT_FILE = "found_keys.txt";

//----------- Global State -----------
std::atomic<bool> key_found(false);
std::atomic<uint64_t> keys_processed(0);
std::mutex log_mutex;
std::string current_range;

//----------- CUDA Kernel -----------
__global__ void crackKernel(uint64_t start, uint64_t end, const char* target, 
                           bool* found, char* key, uint64_t* counter) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t current = start + idx;

    if (current >= end || *found) return;

    // Simulated Key Checking (Replace with actual logic)
    if (current == 123456789) {  // Example condition for finding the key
        *found = true;
        sprintf(key, "%llu", current);
    }
    atomicAdd(counter, 1);
}

//----------- Progress Manager -----------
class ProgressManager {
public:
    void load_checkpoint() {
        std::ifstream file(CHECKPOINT_FILE, std::ios::binary);
        if (file) {
            file.read(reinterpret_cast<char*>(&start_key), sizeof(start_key));
            file.read(reinterpret_cast<char*>(&total_keys), sizeof(total_keys));
            file.close();
        }
    }

    void save_checkpoint() {
        std::ofstream file(CHECKPOINT_FILE, std::ios::binary);
        uint64_t processed = start_key + keys_processed.load();
        file.write(reinterpret_cast<const char*>(&processed), sizeof(processed));
        file.write(reinterpret_cast<const char*>(&total_keys), sizeof(total_keys));
        file.close();
    }

    void get_user_range() {
        std::cout << "Enter starting 2-digit hex range (e.g. FF): ";
        std::cin >> current_range;
        start_key = convert_range_to_decimal(current_range);
        total_keys = calculate_total_keys();
    }

    void show_progress() {
        while (!key_found) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            float progress = (keys_processed * 100.0f) / total_keys;
            std::lock_guard<std::mutex> lock(log_mutex);
            std::cout << "\r[" << current_range << "] Progress: "
                      << std::fixed << std::setprecision(2) << progress << "%";
            std::flush(std::cout);
        }
    }

private:
    uint64_t start_key, total_keys;

    uint64_t convert_range_to_decimal(const std::string& range) {
        return std::stoull(range, nullptr, 16);
    }

    uint64_t calculate_total_keys() {
        return 0xFFFFFFFFFFFFFFFF / 256; // Example calculation
    }
};

//----------- Logger Class -----------
class Logger {
public:
    static void log(const std::string& message) {
        std::lock_guard<std::mutex> lock(log_mutex);
        std::ofstream file(LOG_FILE, std::ios::app);
        auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        file << std::put_time(std::localtime(&now), "%F %T") << " - " << message << "\n";
    }

    static void save_result(const std::string& key) {
        std::ofstream file(RESULT_FILE);
        file << "Private Key Found: " << key << "\n";
    }
};

//----------- Main Application -----------
class CrackBit {
public:
    void run() {
        Logger::log("Application started");

        ProgressManager progress;
        progress.load_checkpoint();
        if (current_range.empty()) progress.get_user_range();

        std::thread progress_thread(&ProgressManager::show_progress, &progress);

        cudaStream_t stream;
        cudaStreamCreate(&stream);

        uint64_t* d_counter;
        bool* d_found;
        char* d_key;
        cudaMalloc(&d_counter, sizeof(uint64_t));
        cudaMalloc(&d_found, sizeof(bool));
        cudaMalloc(&d_key, 64);
        cudaMemset(d_counter, 0, sizeof(uint64_t));
        cudaMemset(d_found, 0, sizeof(bool));

        uint64_t keys_per_batch = 1000000;
        uint64_t current_key = 0;
        while (!key_found && current_key < keys_per_batch) {
            crackKernel<<<256, 256, 0, stream>>>(current_key, current_key + 65536, nullptr, d_found, d_key, d_counter);
            cudaMemcpyAsync(&key_found, d_found, sizeof(bool), cudaMemcpyDeviceToHost, stream);
            keys_processed += 65536;
            current_key += 65536;

            if (keys_processed.load() % 100000 == 0) progress.save_checkpoint();
        }

        cudaStreamDestroy(stream);
        cudaFree(d_counter);
        cudaFree(d_found);
        cudaFree(d_key);
        progress_thread.join();

        if (key_found) {
            char host_key[64];
            cudaMemcpy(host_key, d_key, 64, cudaMemcpyDeviceToHost);
            Logger::save_result(host_key);
            Logger::log("KEY FOUND! Application stopping");
            std::remove(CHECKPOINT_FILE.c_str());
        } else {
            Logger::log("Application stopped normally");
        }
    }
};

int main() {
    try {
        CrackBit app;
        app.run();
    } catch (const std::exception& e) {
        Logger::log("CRITICAL ERROR: " + std::string(e.what()));
    }
    return 0;
}