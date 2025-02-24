#pragma once

#include <cstdint>
#include <string>
#include <array>

class AddressUtils {
public:
    // Optimized hash160 using AVX2 instruction set
    static void hash160_avx2(const uint8_t* input, size_t len, uint8_t* output);

    // Base58Check encoding with version byte and checksum
    static std::string base58check_encode(uint8_t version, const uint8_t* payload, size_t len);
    
private:
    // Base58 character table
    static constexpr std::array<char, 58> BASE58_TABLE = {
        '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G',
        'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y',
        'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'm', 'n', 'o', 'p',
        'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
    };
};
