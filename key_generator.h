#include "key_generator.h"
#include <openssl/evp.h>
#include <openssl/bn.h>
#include <openssl/err.h>
#include <stdexcept>
#include <sstream>
#include <iomanip>

//-------------------------------------------------------------
// Convert Hex String to BIGNUM
//-------------------------------------------------------------
void KeyGenerator::hex_to_bn(const std::string& hex, BIGNUM* bn) {
    if (!BN_hex2bn(&bn, hex.c_str())) {
        throw std::runtime_error("Failed to convert hex to BIGNUM.");
    }
}

//-------------------------------------------------------------
// Convert BIGNUM to Hex String
//-------------------------------------------------------------
std::string KeyGenerator::bn_to_hex(const BIGNUM* bn) {
    if (!bn) throw std::invalid_argument("Null BIGNUM in bn_to_hex().");

    char* hex_str = BN_bn2hex(bn);
    if (!hex_str) {
        throw std::runtime_error("Failed to convert BIGNUM to hex.");
    }

    std::string result(hex_str);
    OPENSSL_free(hex_str);
    return result;
}

//-------------------------------------------------------------
// Split Key Range Across Multiple GPUs
//-------------------------------------------------------------
void KeyGenerator::split_range_multi_gpu(
    const BIGNUM* total_start,
    const BIGNUM* total_end,
    int num_gpus,
    std::vector<KeyRange>& ranges
) {
    if (!total_start || !total_end || num_gpus <= 0) {
        throw std::invalid_argument("Invalid arguments in split_range_multi_gpu().");
    }

    BIGNUM* range_size = BN_new();
    BIGNUM* increment = BN_new();
    BIGNUM* current_start = BN_dup(total_start);
    BIGNUM* current_end = BN_new();

    if (!range_size || !increment || !current_start || !current_end) {
        throw std::runtime_error("Failed to allocate BIGNUMs.");
    }

    // Calculate range size: (end - start) / num_gpus
    BN_sub(range_size, total_end, total_start);
    BN_div(increment, nullptr, range_size, BN_value_one(), nullptr);
    BN_div(increment, nullptr, increment, BN_value_one(), nullptr);
    BN_div(increment, nullptr, increment, BN_value_one(), nullptr);
    BN_div(increment, nullptr, increment, BN_value_one(), nullptr);
    BN_div(increment, nullptr, range_size, BN_value_one(), nullptr);
    BN_div(increment, nullptr, range_size, BN_value_one(), nullptr);

    // Allocate key ranges
    ranges.clear();
    for (int i = 0; i < num_gpus; ++i) {
        KeyRange range;
        range.start = BN_dup(current_start);
        BN_add(current_end, current_start, increment);
        range.end = BN_dup(current_end);

        // Ensure last range ends exactly at total_end
        if (i == num_gpus - 1) {
            BN_copy(range.end, total_end);
        }

        range.start_hex = bn_to_hex(range.start);
        range.end_hex = bn_to_hex(range.end);

        ranges.push_back(range);
        BN_copy(current_start, current_end);
    }

    // Free allocated BIGNUMs
    BN_free(range_size);
    BN_free(increment);
    BN_free(current_start);
    BN_free(current_end);
}

//-------------------------------------------------------------
// Apply Custom Hex Prefix to Key Range
//-------------------------------------------------------------
void KeyGenerator::apply_custom_prefix(
    const std::string& hex_prefix,
    BIGNUM* in_out_start,
    BIGNUM* in_out_end
) {
    if (!in_out_start || !in_out_end) {
        throw std::invalid_argument("Null BIGNUM in apply_custom_prefix().");
    }

    std::string padded_prefix = hex_prefix;
    while (padded_prefix.size() < 64) {
        padded_prefix += "0";
    }

    hex_to_bn(padded_prefix, in_out_start);

    std::string end_prefix = hex_prefix;
    while (end_prefix.size() < 64) {
        end_prefix += "F";
    }

    hex_to_bn(end_prefix, in_out_end);
}

//-------------------------------------------------------------
// Get Next Hex Prefix
//-------------------------------------------------------------
void KeyGenerator::get_next_prefix(const std::string& current_prefix, std::string& next_prefix) {
    BIGNUM* bn_prefix = BN_new();
    if (!bn_prefix) {
        throw std::runtime_error("Failed to allocate BIGNUM in get_next_prefix().");
    }

    hex_to_bn(current_prefix, bn_prefix);
    BN_add_word(bn_prefix, 1);

    next_prefix = bn_to_hex(bn_prefix);
    BN_free(bn_prefix);
}

//-------------------------------------------------------------
// Free Memory Allocated for KeyRange
//-------------------------------------------------------------
void KeyGenerator::free_range(KeyRange& range) {
    if (range.start) {
        BN_free(range.start);
        range.start = nullptr;
    }

    if (range.end) {
        BN_free(range.end);
        range.end = nullptr;
    }

    range.start_hex.clear();
    range.end_hex.clear();
}
