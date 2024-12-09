#include "binfuse/filter.hpp"
#include "binfuse/sharded_filter.hpp"
#include "gtest/gtest.h"
#include <charconv>
#include <cstdint>
#include <filesystem>
#include <vector>

class binfuse_test : public testing::Test {
protected:
  binfuse_test()
      : testtmpdir{std::filesystem::canonical(std::filesystem::current_path() / "../../test/tmp")},
        testdatadir{std::filesystem::canonical(testtmpdir / "../data")},
        filter_path{testtmpdir / "filter.bin"} {}

  template <binfuse::filter_type FilterType>
  void build_filter(double max_false_positive_rate) {
    std::ifstream sample(testdatadir / "sample.txt");

    std::vector<std::uint64_t> keys;
    for (std::string line; std::getline(sample, line);) {
      std::uint64_t key{};
      std::from_chars(line.data(), line.data() + line.size(), key, 16);
      keys.emplace_back(key);
    }
    auto filter = binfuse::filter<FilterType>(keys);
    EXPECT_TRUE(filter.verify(keys));
    EXPECT_LE(filter.estimate_false_positive_rate(), max_false_positive_rate);
  }

  template <binfuse::filter_type FilterType>
  void build_sharded_filter(double max_false_positive_rate) {
    std::filesystem::path filter_filename;
    if constexpr (std::same_as<FilterType, binary_fuse8_t>) {
      filter_filename = "sharded_filter8.bin";
    } else {
      filter_filename = "sharded_filter16.bin";
    }
    {
      binfuse::sharded_filter<FilterType, mio::access_mode::write> sharded_filter_sink(
          testtmpdir / filter_filename);

      std::ifstream sample(testdatadir / "sample.txt");
      sharded_filter_sink.stream_prepare();
      for (std::string line; std::getline(sample, line);) {
        std::uint64_t key{};
        std::from_chars(line.data(), line.data() + line.size(), key, 16);
        sharded_filter_sink.stream_add(key);
      }
      sharded_filter_sink.stream_finalize();

      const binfuse::sharded_filter<FilterType, mio::access_mode::read> sharded_filter_source(
          testtmpdir / filter_filename);

      // full verify across all shards
      sample.seekg(0);
      for (std::string line; std::getline(sample, line);) {
        std::uint64_t needle{};
        std::from_chars(line.data(), line.data() + line.size(), needle, 16);
        EXPECT_TRUE(sharded_filter_source.contains(needle));
      }

      EXPECT_LE(sharded_filter_source.estimate_false_positive_rate(), max_false_positive_rate);
    } // allow mmap to destroy before removing file (required on windows)

    std::filesystem::remove(testtmpdir / filter_filename);
  }

  std::filesystem::path testtmpdir;
  std::filesystem::path testdatadir;

  std::filesystem::path filter_path;
};

TEST_F(binfuse_test, build_filter8) { // NOLINT
  build_filter<binary_fuse8_s>(0.005);
}

TEST_F(binfuse_test, build_sharded_filter8) { // NOLINT
  build_sharded_filter<binary_fuse8_s>(0.005);
}

TEST_F(binfuse_test, build_filter16) { // NOLINT
  build_filter<binary_fuse16_s>(0.00005);
}

TEST_F(binfuse_test, build_sharded_filter16) { // NOLINT
  build_sharded_filter<binary_fuse16_s>(0.00005);
}
