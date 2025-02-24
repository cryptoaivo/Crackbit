#ifndef BITCOIN_UTILS_H
#define BITCOIN_UTILS_H

#include <vector>
#include <string>
#include <secp256k1.h>

class BitcoinUtils {
public:
    /**
     * @brief Generates a public key from a given private key.
     * @param privateKey A 32-byte private key.
     * @param publicKey Output vector for the generated public key (compressed, 33 bytes).
     * @return True if successful, false on failure.
     */
    static bool generatePublicKey(const std::vector<uint8_t>& privateKey, 
                                  std::vector<uint8_t>& publicKey);

    /**
     * @brief Generates a Bitcoin address from a public key.
     * @param publicKey A compressed public key (33 bytes).
     * @return A valid Bitcoin address.
     */
    static std::string generateAddress(const std::vector<uint8_t>& publicKey);

    /**
     * @brief Computes HASH160 (SHA-256 followed by RIPEMD-160).
     * @param data Input data.
     * @return 20-byte HASH160 result.
     */
    static std::vector<uint8_t> computeHash160(const std::vector<uint8_t>& data);

private:
    /**
     * @brief secp256k1 context used for cryptographic operations.
     */
    static secp256k1_context* secp256k1_ctx;

    /**
     * @brief Initializes the secp256k1 context (called once).
     */
    static void initContext();

    /**
     * @brief Computes the SHA-256 hash of the input data.
     * @param data Pointer to input data.
     * @param len Length of input data.
     * @return 32-byte SHA-256 hash.
     */
    static std::vector<uint8_t> sha256(const uint8_t* data, size_t len);

    /**
     * @brief Computes the RIPEMD-160 hash of the input data.
     * @param data Pointer to input data.
     * @param len Length of input data.
     * @return 20-byte RIPEMD-160 hash.
     */
    static std::vector<uint8_t> ripemd160(const uint8_t* data, size_t len);

    /**
     * @brief Encodes data using Base58Check encoding.
     * @param version Version byte (e.g., 0x00 for mainnet addresses).
     * @param payload The payload data (typically HASH160 output).
     * @return A Base58Check-encoded Bitcoin address.
     */
    static std::string base58CheckEncode(uint8_t version, 
                                         const std::vector<uint8_t>& payload);
};

#endif // BITCOIN_UTILS_H
