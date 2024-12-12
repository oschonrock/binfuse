#include "binfuse/sharded_filter.hpp"
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <random>
#include <ratio>
#include <stdexcept>
#include <vector>

using clk   = std::chrono::high_resolution_clock;
using nanos = std::chrono::duration<double, std::nano>;

std::vector<std::uint64_t> gen_shard(std::uint64_t prefix, std::uint8_t shard_bits,
                                     std::size_t size) {

  std::mt19937_64 gen{std::random_device{}()};

  std::uint8_t shift = 64 - shard_bits;
  const auto   min   = prefix << shift;
  const auto   max   = ((prefix + 1) << shift) - 1UL;

  std::uniform_int_distribution<std::uint64_t> dist(min, max);

  std::vector<std::uint64_t> shard;
  shard.reserve(size);
  for (std::size_t i = 0; i != size; ++i) {
    shard.push_back(dist(gen));
  }
  return shard;
}

template <typename T>
void populate(T& filter, std::uint32_t shards, std::uint8_t shard_bits, std::size_t shard_size) {
  clk::duration gen_shard_total{};
  clk::duration popluate_filter_total{};
  clk::duration verify_filter_total{};
  clk::duration add_shard_total{};
  for (std::uint32_t prefix = 0; prefix != shards; ++prefix) {
    auto       start      = clk::now();
    const auto shard_keys = gen_shard(prefix, shard_bits, shard_size);
    gen_shard_total += clk::now() - start;

    start = clk::now();
    typename T::shard_filter_t shard(shard_keys);
    popluate_filter_total += clk::now() - start;

    start = clk::now();
    if (!shard.verify(shard_keys)) {
      throw std::runtime_error("verify failed!!");
    }
    verify_filter_total += clk::now() - start;

    start = clk::now();
    filter.add_shard(shard, prefix);
    add_shard_total += clk::now() - start;
  }
  auto size = static_cast<double>(shards * shard_size);
  std::cout << std::format("f{:<2d} {:8.1f}ns {:8.1f}ns {:8.1f}ns {:8.1f}ns", T::nbits,
                           duration_cast<nanos>(gen_shard_total).count() / size,
                           duration_cast<nanos>(popluate_filter_total).count() / size,
                           duration_cast<nanos>(verify_filter_total).count() / size,
                           duration_cast<nanos>(add_shard_total).count() / size);
}

template <typename T>
void query(const T& filter, std::size_t size) {
  std::mt19937_64 gen{std::random_device{}()};

  auto        start       = clk::now();
  std::size_t found_count = 0;
  for (std::size_t i = 0; i != size; ++i) {
    found_count += filter.contains(gen());
  }
  auto end = clk::now();
  std::cout << std::format(" {:8.1f}ns  {:.6f}%\n",
                           duration_cast<nanos>(end - start).count() / static_cast<double>(size),
                           100.0 * static_cast<double>(found_count) / static_cast<double>(size));
}

int main() {

  try {
    constexpr std::size_t size = 100'000'000;

    for (std::uint8_t shard_bits = 1; shard_bits <= 8; ++shard_bits) {

      const std::uint32_t shards     = 1U << shard_bits;
      const std::size_t   shard_size = size / shards;

      std::cout << std::format("\n\nShard Size: {}  Shards: {}  Keys: {}\n\n", shard_size, shards,
                               size);

      std::cout << std::format("      {:>8s}   {:>8s}   {:>8s}   {:>8s}   {:>8s}   {:>8s}\n", "gen",
                               "populate", "verify", "add", "query", "f+ve");

      {
        binfuse::sharded_filter8_sink sink8("filter8.bin", shard_bits);
        populate(sink8, shards, shard_bits, shard_size);
      }
      {
        binfuse::sharded_filter8_source source8("filter8.bin", shard_bits);
        query(source8, size);
      }
      {
        binfuse::sharded_filter16_sink sink16("filter16.bin", shard_bits);
        populate(sink16, shards, shard_bits, shard_size);
      }
      {
        binfuse::sharded_filter16_source source16("filter16.bin", shard_bits);
        query(source16, size);
      }
      std::filesystem::remove("filter8.bin");
      std::filesystem::remove("filter16.bin");
    }
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return EXIT_FAILURE;
  }
}
