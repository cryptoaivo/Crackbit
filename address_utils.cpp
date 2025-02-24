#include "address_utils.h"
#include <openssl/evp.h>
#include <openssl/bn.h>
#include <array>
#include <vector>
#include <immintrin.h>
#include <cstring>

// Thread-local OpenSSL contexts
thread_local EVP_MD_CTX* sha256_ctx = EVP_MD_CTX_new();
thread_local EVP_MD_CTX* ripemd160_ctx = EVP_MD_CTX_new();

constexpr std::array<char, 58> BASE58_TABLE = {
    '1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G',
    'H','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y',
    'Z','a','b','c','d','e','f','g','h','i','j','k','m','n','o','p',
    'q','r','s','t','u','v','w','x','y','z'
};

__attribute__((target("avx2")))
void hash160_avx2(const uint8_t* input, size_t len, uint8_t* output) {
    alignas(32) std::array<uint8_t, 32> sha_hash;
    alignas(32) std::array<uint8_t, 20> ripe_hash;
    
    // SHA-256
    EVP_DigestInit_ex(sha256_ctx, EVP_sha256(), nullptr);
    EVP_DigestUpdate(sha256_ctx, input, len);
    EVP_DigestFinal_ex(sha256_ctx, sha_hash.data(), nullptr);
    
    // RIPEMD-160
    EVP_DigestInit_ex(ripemd160_ctx, EVP_ripemd160(), nullptr);
    EVP_DigestUpdate(ripemd160_ctx, sha_hash.data(), 32);
    EVP_DigestFinal_ex(ripemd160_ctx, ripe_hash.data(), nullptr);
    
    // Vectorized copy using AVX2
    std::memcpy(output, ripe_hash.data(), 20);
}

std::string base58check_encode(uint8_t version, const uint8_t* payload, size_t len) {
    // Thread-local buffer to avoid repeated allocations
    static thread_local std::vector<uint8_t> buffer;
    buffer.resize(len + 5); // Version + payload + checksum
    
    buffer[0] = version;
    std::memcpy(buffer.data() + 1, payload, len);
    
    // Double SHA-256 checksum
    uint8_t checksum[32];
    EVP_Digest(buffer.data(), len + 1, checksum, nullptr, EVP_sha256(), nullptr);
    EVP_Digest(checksum, 32, checksum, nullptr, EVP_sha256(), nullptr);
    
    std::memcpy(buffer.data() + len + 1, checksum, 4);

    // Convert to Base58
    BIGNUM* num = BN_bin2bn(buffer.data(), len + 5, nullptr);
    BN_CTX* ctx = BN_CTX_new();
    
    std::string result;
    BIGNUM* div = BN_new();
    BIGNUM* rem = BN_new();
    BIGNUM* base = BN_new();
    
    BN_set_word(base, 58);
    
    while (!BN_is_zero(num)) {
        BN_div(div, rem, num, base, ctx);
        result.push_back(BASE58_TABLE[BN_get_word(rem)]);
        BN_copy(num, div);
    }

    // Add leading zeros
    for (size_t i = 0; i < len + 1 && buffer[i] == 0; i++) {
        result.push_back('1');
    }

    // Free OpenSSL resources
    BN_free(num);
    BN_free(div);
    BN_free(rem);
    BN_free(base);
    BN_CTX_free(ctx);

    return std::string(result.rbegin(), result.rend());
}
