#include "binaryfusefilter.h"
#include "binfuse/filter.hpp"
#include "binfuse/sharded_filter.hpp"
#include "gtest/gtest.h"
#include <charconv>
#include <cstdint>
#include <filesystem>
#include <random>
#include <stdexcept>
#include <vector>

std::vector<std::uint64_t> load_sample() {
  std::ifstream              sample("data/sample.txt");
  std::vector<std::uint64_t> keys;
  for (std::string line; std::getline(sample, line);) {
    std::uint64_t key{};
    std::from_chars(line.data(), line.data() + line.size(), key, 16);
    keys.emplace_back(key);
  }
  return keys;
}

template <binfuse::filter_type FilterType>
void test_filter(std::span<const std::uint64_t> keys, double max_false_positive_rate) {
  auto filter = binfuse::filter<FilterType>(keys);
  EXPECT_TRUE(filter.verify(keys));
  EXPECT_LE(estimate_false_positive_rate(filter), max_false_positive_rate);
}

template <binfuse::filter_type FilterType>
void test_sharded_filter(std::span<const std::uint64_t> keys, double max_false_positive_rate,
                         uint8_t sharded_bits = 8) {
  std::filesystem::path filter_filename;
  filter_filename = "tmp/sharded_filter.bin";
  {
    binfuse::sharded_filter<FilterType, mio::access_mode::write> sharded_sink(filter_filename,
                                                                              sharded_bits);

    sharded_sink.stream_prepare();
    for (auto key: keys) {
      sharded_sink.stream_add(key);
    }
    sharded_sink.stream_finalize();

    const binfuse::sharded_filter<FilterType, mio::access_mode::read> sharded_source(
        filter_filename, sharded_bits);

    // full verify across all shards
    for (auto needle: keys) {
      EXPECT_TRUE(sharded_source.contains(needle));
    }

    EXPECT_LE(estimate_false_positive_rate(sharded_source), max_false_positive_rate);
  } // allow mmap to destroy before removing file (required on windows)

  std::filesystem::remove(filter_filename);
}

template <typename F>
[[nodiscard]] static double estimate_false_positive_rate(const F& fil) {
  auto         gen         = std::mt19937_64(std::random_device{}());
  size_t       matches     = 0;
  const size_t sample_size = 1'000'000;
  for (size_t t = 0; t < sample_size; t++) {
    if (fil.contains(gen())) { // no distribution needed
      matches++;
    }
  }
  return static_cast<double>(matches) / static_cast<double>(sample_size) -
         static_cast<double>(fil.size()) /
             static_cast<double>(std::numeric_limits<std::uint64_t>::max());
}

TEST(binfuse, filter8) { // NOLINT
  test_filter<binary_fuse8_t>(load_sample(), 0.005);
}

TEST(binfuse, sharded_filter8) { // NOLINT
  test_sharded_filter<binary_fuse8_t>(load_sample(), 0.005);
}

TEST(binfuse, filter16) { // NOLINT
  test_filter<binary_fuse16_t>(load_sample(), 0.00005);
}

TEST(binfuse, sharded_filter16) { // NOLINT
  test_sharded_filter<binary_fuse16_t>(load_sample(), 0.00005);
}

TEST(binfuse, sharded_filter8_32) {                             // NOLINT
  test_sharded_filter<binary_fuse8_t>(load_sample(), 0.005, 5); // 5 sharded_bits
}

TEST(binfuse, sharded_filter16_32) {                               // NOLINT
  test_sharded_filter<binary_fuse16_t>(load_sample(), 0.00005, 5); // 5 sharded_bits
}

TEST(binfuse_sharded_filter, default_construct) { // NOLINT
  binfuse::sharded_filter8_source sharded_source;
}

TEST(binfuse_sharded_filter, build_tiny) { // NOLINT
  binfuse::filter8 tiny_low(std::vector<std::uint64_t>{
      0x0000000000000000,
      0x0000000000000001,
      0x0000000000000002,
  });

  binfuse::filter8 tiny_high(std::vector<std::uint64_t>{
      0x8000000000000000,
      0x8000000000000001,
      0x8000000000000002,
  });

  binfuse::sharded_filter<binary_fuse8_t, mio::access_mode::write> sharded_tiny_sink(
      "tmp/sharded_filter8_tiny.bin", 1);

  sharded_tiny_sink.add(tiny_low, 0);
  sharded_tiny_sink.add(tiny_high, 1);

  binfuse::sharded_filter<binary_fuse8_t, mio::access_mode::read> sharded_tiny_source(
      "tmp/sharded_filter8_tiny.bin", 1);

  EXPECT_TRUE(sharded_tiny_source.contains(0x0000000000000000));
  EXPECT_TRUE(sharded_tiny_source.contains(0x0000000000000001));
  EXPECT_TRUE(sharded_tiny_source.contains(0x0000000000000002));
  EXPECT_TRUE(sharded_tiny_source.contains(0x8000000000000000));
  EXPECT_TRUE(sharded_tiny_source.contains(0x8000000000000001));
  EXPECT_TRUE(sharded_tiny_source.contains(0x8000000000000002));
  std::filesystem::remove("tmp/sharded_filter8_tiny.bin");
}

TEST(binfuse_sharded_filter, stream_tiny) { // NOLINT
  binfuse::sharded_filter<binary_fuse8_t, mio::access_mode::write> sharded_tiny_sink(
      "tmp/sharded_filter8_tiny.bin", 1);

  sharded_tiny_sink.stream_prepare();
  sharded_tiny_sink.stream_add(0x0000000000000000);
  sharded_tiny_sink.stream_add(0x0000000000000001);
  sharded_tiny_sink.stream_add(0x0000000000000002);
  sharded_tiny_sink.stream_add(0x8000000000000000);
  sharded_tiny_sink.stream_add(0x8000000000000001);
  sharded_tiny_sink.stream_add(0x8000000000000002);
  sharded_tiny_sink.stream_finalize();

  binfuse::sharded_filter<binary_fuse8_t, mio::access_mode::read> sharded_tiny_source(
      "tmp/sharded_filter8_tiny.bin", 1);

  EXPECT_TRUE(sharded_tiny_source.contains(0x0000000000000000));
  EXPECT_TRUE(sharded_tiny_source.contains(0x0000000000000001));
  EXPECT_TRUE(sharded_tiny_source.contains(0x0000000000000002));
  EXPECT_TRUE(sharded_tiny_source.contains(0x8000000000000000));
  EXPECT_TRUE(sharded_tiny_source.contains(0x8000000000000001));
  EXPECT_TRUE(sharded_tiny_source.contains(0x8000000000000002));
  std::filesystem::remove("tmp/sharded_filter8_tiny.bin");
}

TEST(binfuse_sharded_filter, load_tiny) { // NOLINT
  binfuse::sharded_filter8_source sharded_tiny_source;
  EXPECT_THROW(sharded_tiny_source.set_filename("non_existant.bin"), std::runtime_error);

  // wrong capacity
  EXPECT_THROW(sharded_tiny_source.set_filename("data/sharded_filter8_tiny.bin"),
               std::runtime_error);

  sharded_tiny_source.set_filename("data/sharded_filter8_tiny.bin", 1); // correct capacity

  EXPECT_TRUE(sharded_tiny_source.contains(0x0000000000000002));
  EXPECT_TRUE(sharded_tiny_source.contains(0x8000000000000000));
}
