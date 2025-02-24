cmake_minimum_required(VERSION 3.15)
project(CrackBit)

# Enable C++17 for better performance & compatibility
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(CUDA REQUIRED)
find_package(OpenSSL REQUIRED)

# Source files
set(SOURCES
    crackbit.cpp
    bitcoin_utils.cpp
    key_generator.cpp
    address_matcher.cpp
)

# Create the executable
add_executable(CrackBit ${SOURCES})

# Include directories
target_include_directories(CrackBit PRIVATE ${OPENSSL_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS})

# Link necessary libraries
target_link_libraries(CrackBit PRIVATE ${CUDA_LIBRARIES} ${OPENSSL_LIBRARIES})

# Optimize build flags
if (CMAKE_BUILD_TYPE MATCHES Release)
    message(STATUS "Building in Release mode: Enabling optimizations.")
    target_compile_options(CrackBit PRIVATE -O3 -march=native -flto)
endif
