// puzzle_config.h
#pragma once
#include <string>
#include <unordered_map>

// Struct to store puzzle configuration
struct PuzzleConfig {
    std::string startHex;
    std::string endHex;
    std::string targetAddress;
};

// Puzzle Database (Predefined Ranges & Addresses)
inline const std::unordered_map<int, PuzzleConfig> PUZZLE_DB = {
    {68, {"80000000000000000",  "fffffffffffffffff",  "1MVDYgVaSN6iKKEsbzRUAYFrYJadLYZvvZ"}},
    {69, {"100000000000000000", "1fffffffffffffffff", "19vkiEajfhuZ8bs8Zu2jgmC6o"}},
    {71, {"400000000000000000", "7fffffffffffffffff", "1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU"}}
};
