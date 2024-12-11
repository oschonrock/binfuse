#pragma once

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <random>
#include <string>
#include <vector>

inline std::vector<std::uint64_t> load_sample() {
  std::ifstream              sample("data/sample.txt");
  std::vector<std::uint64_t> keys;
  for (std::string line; std::getline(sample, line);) {
    std::uint64_t key{};
    std::from_chars(line.data(), line.data() + line.size(), key, 16);
    keys.emplace_back(key);
  }
  return keys;
}

template <typename F>
[[nodiscard]] static double estimate_false_positive_rate(const F& fil) {
  auto         gen         = std::mt19937_64(std::random_device{}());
  std::size_t  matches     = 0;
  const size_t sample_size = 1'000'000;
  for (size_t t = 0; t < sample_size; t++) {
    if (fil.contains(gen())) { // no distribution needed
      matches++;
    }
  }
  return static_cast<double>(matches) / static_cast<double>(sample_size);
}
