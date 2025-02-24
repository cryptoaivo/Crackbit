#include "range_converter.h"
#include <openssl/bn.h>
#include <stdexcept>
#include <iomanip>
#include <sstream>

//--------------------------------------------------------------------------------------------------
// Internal Helper Functions
//--------------------------------------------------------------------------------------------------

static BIGNUM* create_mask(int prefix_bits) {
    BIGNUM* mask = BN_new();
    BN_set_bit(mask, prefix_bits);  // mask = 2^prefix_bits
    BN_sub_word(mask, 1);           // mask = 2^prefix_bits - 1
    return mask;
}

static void clamp_to_range(BIGNUM* value, const BIGNUM* range_start, const BIGNUM* range_end) {
    if (BN_cmp(value, range_start) < 0) {
        BN_copy(value, range_start);
    } else if (BN_cmp(value, range_end) > 0) {
        BN_copy(value, range_end);
    }
}

//--------------------------------------------------------------------------------------------------
// Public Interface Implementation
//--------------------------------------------------------------------------------------------------

KeyRange convert_hex_range(
    const std::string& two_digit_hex,
    const std::string& puzzle_start_hex,
    const std::string& puzzle_end_hex
) {
    if (two_digit_hex.length() != 2 || !isxdigit(two_digit_hex[0]) || !isxdigit(two_digit_hex[1])) {
        throw std::invalid_argument("Invalid 2-digit hex prefix");
    }

    BIGNUM* puzzle_start = nullptr;
    BIGNUM* puzzle_end = nullptr;
    BN_hex2bn(&puzzle_start, puzzle_start_hex.c_str());
    BN_hex2bn(&puzzle_end, puzzle_end_hex.c_str());

    int prefix_bits = 8;  // 2 hex digits = 8 bits
    BIGNUM* prefix_mask = create_mask(prefix_bits);
    BIGNUM* prefix_value = nullptr;
    BN_hex2bn(&prefix_value, two_digit_hex.c_str());

    int total_bits = BN_num_bits(puzzle_end);
    BN_lshift(prefix_value, prefix_value, total_bits - prefix_bits);

    BIGNUM* range_start = BN_dup(prefix_value);
    BIGNUM* range_end = BN_new();
    BN_add(range_end, prefix_value, prefix_mask);

    clamp_to_range(range_start, puzzle_start, puzzle_end);
    clamp_to_range(range_end, puzzle_start, puzzle_end);

    BIGNUM* total_keys = BN_new();
    BN_sub(total_keys, range_end, range_start);
    BN_add_word(total_keys, 1);

    KeyRange result;
    result.start_hex = bn2hex(range_start);
    result.end_hex = bn2hex(range_end);
    result.total_keys = bn2dec(total_keys);

    BN_free(puzzle_start);
    BN_free(puzzle_end);
    BN_free(prefix_mask);
    BN_free(prefix_value);
    BN_free(range_start);
    BN_free(range_end);
    BN_free(total_keys);

    return result;
}

std::string get_progress_percentage(
    const std::string& current_key_hex,
    const std::string& range_start_hex,
    const std::string& range_end_hex
) {
    BIGNUM* current = nullptr;
    BIGNUM* start = nullptr;
    BIGNUM* end = nullptr;
    
    BN_hex2bn(&current, current_key_hex.c_str());
    BN_hex2bn(&start, range_start_hex.c_str());
    BN_hex2bn(&end, ra
