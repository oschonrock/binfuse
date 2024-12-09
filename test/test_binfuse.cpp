#include "binfuse/filter.hpp"
#include "binfuse/sharded_filter.hpp"
#include "gtest/gtest.h"
#include <charconv>
#include <cstdint>
#include <filesystem>
#include <random>
#include <vector>

class binfuse_test : public testing::Test {
protected:
  binfuse_test()
      : testtmpdir{std::filesystem::canonical(std::filesystem::current_path() / "tmp")},
        testdatadir{std::filesystem::canonical(std::filesystem::current_path() / "data")},
        filter_path{testtmpdir / "filter.bin"} {}

  template <binfuse::filter_type FilterType>
  void test_filter(double max_false_positive_rate) {
    std::ifstream sample(testdatadir / "sample.txt");

    std::vector<std::uint64_t> keys;
    for (std::string line; std::getline(sample, line);) {
      std::uint64_t key{};
      std::from_chars(line.data(), line.data() + line.size(), key, 16);
      keys.emplace_back(key);
    }
    auto filter = binfuse::filter<FilterType>(keys);
    EXPECT_TRUE(filter.verify(keys));
    EXPECT_LE(estimate_false_positive_rate(filter), max_false_positive_rate);
  }

  template <binfuse::filter_type FilterType>
  void test_sharded_filter(double max_false_positive_rate) {
    std::filesystem::path filter_filename;
    if constexpr (std::same_as<FilterType, binary_fuse8_t>) {
      filter_filename = "sharded_filter8.bin";
    } else {
      filter_filename = "sharded_filter16.bin";
    }
    {
      binfuse::sharded_filter<FilterType, mio::access_mode::write> sharded_sink(testtmpdir /
                                                                                filter_filename);

      std::ifstream sample(testdatadir / "sample.txt");
      sharded_sink.stream_prepare();
      for (std::string line; std::getline(sample, line);) {
        std::uint64_t key{};
        std::from_chars(line.data(), line.data() + line.size(), key, 16);
        sharded_sink.stream_add(key);
      }
      sharded_sink.stream_finalize();

      const binfuse::sharded_filter<FilterType, mio::access_mode::read> sharded_source(
          testtmpdir / filter_filename);

      // full verify across all shards
      sample.seekg(0);
      for (std::string line; std::getline(sample, line);) {
        std::uint64_t needle{};
        std::from_chars(line.data(), line.data() + line.size(), needle, 16);
        EXPECT_TRUE(sharded_source.contains(needle));
      }

      EXPECT_LE(estimate_false_positive_rate(sharded_source), max_false_positive_rate);
    } // allow mmap to destroy before removing file (required on windows)

    std::filesystem::remove(testtmpdir / filter_filename);
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

  std::filesystem::path testtmpdir;
  std::filesystem::path testdatadir;

  std::filesystem::path filter_path;
};

TEST_F(binfuse_test, test_filter8) { // NOLINT
  test_filter<binary_fuse8_t>(0.005);
}

TEST_F(binfuse_test, test_sharded_filter8) { // NOLINT
  test_sharded_filter<binary_fuse8_t>(0.005);
}

TEST_F(binfuse_test, test_filter16) { // NOLINT
  test_filter<binary_fuse16_t>(0.00005);
}

TEST_F(binfuse_test, test_sharded_filter16) { // NOLINT
  test_sharded_filter<binary_fuse16_t>(0.00005);
}
