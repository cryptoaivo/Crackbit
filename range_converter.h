#pragma once

#include <string>

// Forward declaration of BIGNUM from OpenSSL to avoid unnecessary includes
struct bignum_st;
typedef struct bignum_st BIGNUM;

struct KeyRange {
    std::string start_hex;
    std::string end_hex;
    std::string total_keys;
};

// Converts a 2-digit hex prefix into a range within a puzzle range
KeyRange convert_hex_range(
    const std::string& two_digit_hex,
    const std::string& puzzle_start_hex,
    const std::string& puzzle_end_hex
);

// Computes progress percentage of the current key in the given range
std::string get_progress_percentage(
    const std::string& current_key_hex,
    const std::string& range_start_hex,
    const std::string& range_end_hex
);

// Generates the next range from a given 2-digit hex value
std::string get_next_range(const std::string& current_range);

// Utility functions for converting BIGNUM to hex/decimal strings
std::string bn2hex(const BIGNUM* bn);
std::string bn2dec(const BIGNUM* bn);
