#include "bitcoin_utils.h"
#include <secp256k1.h>
#include <openssl/evp.h>
#include <openssl/bn.h>
#include <algorithm>
#include <stdexcept>

// Secp256k1 context (initialized once)
secp256k1_context* BitcoinUtils::secp256k1_ctx = nullptr;

// Base58 character map
const std::string BASE58_CHARS = 
    "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

//-------------------------------------------------------------
// Secp256k1 Initialization
//-------------------------------------------------------------
void BitcoinUtils::initContext() {
    if (!secp256k1_ctx) {
        secp256k1_ctx = secp256k1_context_create(
            SECP256K1_CONTEXT_SIGN | SECP256K1_CONTEXT_VERIFY);
    }
}

//-------------------------------------------------------------
// Public Key Generation
//-------------------------------------------------------------
bool BitcoinUtils::generatePublicKey(const std::vector<uint8_t>& privateKey,
                                     std::vector<uint8_t>& publicKey) {
    initContext();
    publicKey.clear();

    if (privateKey.size() != 32) return false;  // Private key must be 32 bytes

    secp256k1_pubkey pubkey;
    if (!secp256k1_ec_pubkey_create(secp256k1_ctx, &pubkey, privateKey.data()))
        return false;

    // Serialize the public key (compressed format: 33 bytes)
    size_t output_len = 33;
    publicKey.resize(output_len);
    secp256k1_ec_pubkey_serialize(
        secp256k1_ctx, publicKey.data(), &output_len,
        &pubkey, SECP256K1_EC_COMPRESSED
    );

    return output_len == 33;
}

//-------------------------------------------------------------
// HASH160 (SHA-256 + RIPEMD-160)
//-------------------------------------------------------------
std::vector<uint8_t> BitcoinUtils::computeHash160(const std::vector<uint8_t>& data) {
    return ripemd160(sha256(data.data(), data.size()).data(), 32);
}

//-------------------------------------------------------------
// Bitcoin Address Generation
//-------------------------------------------------------------
std::string BitcoinUtils::generateAddress(const std::vector<uint8_t>& publicKey) {
    if (publicKey.size() != 33) {
        throw std::invalid_argument("Public key must be 33 bytes (compressed format).");
    }
    
    // 1. Compute HASH160
    auto hash160 = computeHash160(publicKey);
    
    // 2. Base58Check encode with version byte 0x00 (Bitcoin mainnet)
    return base58CheckEncode(0x00, hash160);
}

//-------------------------------------------------------------
// Internal Cryptographic Helpers
//-------------------------------------------------------------

std::vector<uint8_t> BitcoinUtils::sha256(const uint8_t* data, size_t len) {
    std::vector<uint8_t> digest(EVP_MD_size(EVP_sha256()));

    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    if (!ctx) throw std::runtime_error("Failed to create OpenSSL EVP_MD_CTX.");

    EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr);
    EVP_DigestUpdate(ctx, data, len);
    EVP_DigestFinal_ex(ctx, digest.data(), nullptr);

    EVP_MD_CTX_free(ctx);
    return digest;
}

std::vector<uint8_t> BitcoinUtils::ripemd160(const uint8_t* data, size_t len) {
    std::vector<uint8_t> digest(EVP_MD_size(EVP_ripemd160()));

    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    if (!ctx) throw std::runtime_error("Failed to create OpenSSL EVP_MD_CTX.");

    EVP_DigestInit_ex(ctx, EVP_ripemd160(), nullptr);
    EVP_DigestUpdate(ctx, data, len);
    EVP_DigestFinal_ex(ctx, digest.data(), nullptr);

    EVP_MD_CTX_free(ctx);
    return digest;
}

//-------------------------------------------------------------
// Base58Check Encoding
//-------------------------------------------------------------
std::string BitcoinUtils::base58CheckEncode(uint8_t version, 
                                            const std::vector<uint8_t>& payload) {
    // Create versioned payload
    std::vector<uint8_t> vpayload(1 + payload.size() + 4);
    vpayload[0] = version;
    std::copy(payload.begin(), payload.end(), vpayload.begin() + 1);

    // Compute checksum (first 4 bytes of double SHA-256)
    auto hash1 = sha256(vpayload.data(), payload.size() + 1);
    auto hash2 = sha256(hash1.data(), hash1.size());
    std::copy(hash2.begin(), hash2.begin() + 4, vpayload.begin() + 1 + payload.size());

    // Convert to base58
    BIGNUM* bn = BN_bin2bn(vpayload.data(), vpayload.size(), nullptr);
    if (!bn) throw std::runtime_error("Failed to create BIGNUM.");

    BN_CTX* ctx = BN_CTX_new();
    if (!ctx) {
        BN_free(bn);
        throw std::runtime_error("Failed to create BN_CTX.");
    }

    BIGNUM* base = BN_new();
    BIGNUM* rem = BN_new();
    BN_set_word(base, 58);

    std::string result;
    while (!BN_is_zero(bn)) {
        BN_div(bn, rem, bn, base, ctx);
        result += BASE58_CHARS[BN_get_word(rem)];
    }

    // Count leading zero bytes
    size_t leading_zeros = 0;
    for (uint8_t byte : vpayload) {
        if (byte == 0) leading_zeros++;
        else break;
    }

    // Add leading '1's for each leading zero
    result.insert(result.end(), leading_zeros, '1');

    // Reverse the string
    std::reverse(result.begin(), result.end());

    // Cleanup OpenSSL objects
    BN_free(base);
    BN_free(rem);
    BN_free(bn);
    BN_CTX_free(ctx);

    return result;
}
