#include "key_generator.h"
#include <openssl/bn.h>
#include <stdexcept>
#include <algorithm>

// Thread-local BN_CTX for better efficiency
thread_local BN_CTX* bn_ctx = BN_CTX_new();

//-------------------------------------------------------------
// Convert Hex String to BIGNUM
//-------------------------------------------------------------
void KeyGenerator::hex_to_bn(const std::string& hex, BIGNUM* bn) {
    if (!BN_hex2bn(&bn, hex.c_str())) {
        throw std::invalid_argument("Invalid hex string in hex_to_bn()");
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
        throw std::invalid_argument("Invalid parameters in split_range_multi_gpu().");
    }

    BIGNUM* total_keys = BN_new();
    BIGNUM* keys_per_gpu = BN_new();
    BIGNUM* remainder = BN_new();
    BIGNUM* current = BN_dup(total_start);
    BIGNUM* one = BN_new();

    if (!total_keys || !keys_per_gpu || !remainder || !current || !one) {
        throw std::runtime_error("Failed to allocate BIGNUMs.");
    }

    BN_one(one);
    BN_sub(total_keys, total_end, total_start);
    BN_add(total_keys, total_keys, one);  // Ensure inclusive range
    BN_div(keys_per_gpu, remainder, total_keys, BN_value_one(), bn_ctx);

    ranges.resize(num_gpus);

    for (int i = 0; i < num_gpus; i++) {
        ranges[i].start = BN_dup(current);

        if (i < BN_get_word(remainder)) {
            BN_add_word(keys_per_gpu, 1);
        }

        BN_add(current, current, keys_per_gpu);
        BN_sub_word(current, 1);

        ranges[i].end = BN_dup(current);
        ranges[i].start_hex = bn_to_hex(ranges[i].start);
        ranges[i].end_hex = bn_to_hex(ranges[i].end);

        BN_add_word(current, 1);

        if (i < BN_get_word(remainder)) {
            BN_sub_word(keys_per_gpu, 1);
        }
    }

    // Free allocated BIGNUMs
    BN_free(total_keys);
    BN_free(keys_per_gpu);
    BN_free(remainder);
    BN_free(current);
    BN_free(one);
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

    int prefix_bits = hex_prefix.length() * 4;
    BIGNUM* mask = BN_new();
    BIGNUM* prefix = BN_new();

    if (!mask || !prefix) {
        throw std::runtime_error("Failed to allocate BIGNUMs.");
    }

    // Create a mask for clearing prefix bits
    BN_set_bit(mask, prefix_bits);
    BN_sub_word(mask, 1);
    BN_lshift(mask, mask, 256 - prefix_bits);

    // Convert hex prefix to BIGNUM
    hex_to_bn(hex_prefix, prefix);
    BN_lshift(prefix, prefix, 256 - prefix_bits);

    // Apply prefix mask
    BN_and(in_out_start, in_out_start, mask);
    BN_or(in_out_start, in_out_start, prefix);
    
    BN_and(in_out_end, in_out_end, mask);
    BN_or(in_out_end, in_out_end, prefix);

    BN_free(mask);
    BN_free(prefix);
}

//-------------------------------------------------------------
// Get Next Hex Prefix
//-------------------------------------------------------------
void KeyGenerator::get_next_prefix(const std::string& current_prefix, std::string& next_prefix) {
    if (current_prefix.empty()) {
        throw std::invalid_argument("Empty prefix in get_next_prefix().");
    }

    int prefix_len = current_prefix.length();
    BIGNUM* bn_prefix = BN_new();

    if (!bn_prefix) {
        throw std::runtime_error("Failed to allocate BIGNUM in get_next_prefix().");
    }

    hex_to_bn(current_prefix, bn_prefix);
    BN_add_word(bn_prefix, 1);

    next_prefix = bn_to_hex(bn_prefix);
    BN_free(bn_prefix);

    // Ensure next prefix remains the same length
    if (next_prefix.length() > prefix_len) {
        next_prefix = std::string(prefix_len, '0');
    } else if (next_prefix.length() < prefix_len) {
        next_prefix = std::string(prefix_len - next_prefix.length(), '0') + next_prefix;
    }
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
