#include "binfuse/filter.hpp"
#include "binaryfusefilter.h"
#include "helpers.hpp"
#include "gtest/gtest.h"
#include <cstdint>
#include <filesystem>
#include <span>
#include <utility>
#include <vector>

TEST(binfuse_filter, default_construct) { // NOLINT
  binfuse::filter8 filter;
  EXPECT_FALSE(filter.is_populated());
}

TEST(binfuse_filter, default_construct_persistent) { // NOLINT
  binfuse::filter8_sink filter_sink;
  EXPECT_FALSE(filter_sink.is_populated());

  binfuse::filter8_source filter_source;
  EXPECT_FALSE(filter_source.is_populated());
}

TEST(binfuse_filter, in_memory) { // NOLINT
  binfuse::filter8 filter(std::vector<std::uint64_t>{
      0x0000000000000000,
      0x0000000000000001, // order is not important
      0x0000000000000002,
  });
  EXPECT_TRUE(filter.is_populated());

  EXPECT_TRUE(filter.contains(0x0000000000000000));
  EXPECT_TRUE(filter.contains(0x0000000000000001));
  EXPECT_TRUE(filter.contains(0x0000000000000002));
}

TEST(binfuse_filter, save_load8) { // NOLINT
  {
    binfuse::filter8_sink filter_sink(std::vector<std::uint64_t>{
        0x0000000000000000,
        0x0000000000000001, // order is not important
        0x0000000000000002,
    });
    filter_sink.save("tmp/filter8.bin");

    binfuse::filter8_source filter_source;
    filter_source.load("tmp/filter8.bin");

    EXPECT_TRUE(filter_source.contains(0x0000000000000000));
    EXPECT_TRUE(filter_source.contains(0x0000000000000001));
    EXPECT_TRUE(filter_source.contains(0x0000000000000002));
  }
  std::filesystem::remove("tmp/filter8.bin");
}

TEST(binfuse_filter, save_load16) { // NOLINT
  {
    binfuse::filter16_sink filter_sink(std::vector<std::uint64_t>{
        0x0000000000000000,
        0x0000000000000001, // order is not important
        0x0000000000000002,
    });
    filter_sink.save("tmp/filter16.bin");

    binfuse::filter16_source filter_source;
    filter_source.load("tmp/filter16.bin");

    EXPECT_TRUE(filter_source.contains(0x0000000000000000));
    EXPECT_TRUE(filter_source.contains(0x0000000000000001));
    EXPECT_TRUE(filter_source.contains(0x0000000000000002));
  }
  std::filesystem::remove("tmp/filter16.bin");
}

TEST(binfuse_filter, move) { // NOLINT
  {
    binfuse::filter8_sink filter_sink(std::vector<std::uint64_t>{
        0x0000000000000000,
        0x0000000000000001, // order is not important
        0x0000000000000002,
    });
    filter_sink.save("tmp/filter8.bin");

    binfuse::filter8_source filter_source;
    filter_source.load("tmp/filter8.bin");

    binfuse::filter8_source filter_source2 = std::move(filter_source);

    EXPECT_TRUE(filter_source2.contains(0x0000000000000000));
    EXPECT_TRUE(filter_source2.contains(0x0000000000000001));
    EXPECT_TRUE(filter_source2.contains(0x0000000000000002));
  }
  std::filesystem::remove("tmp/filter8.bin");
}

// larger data tests

TEST(binfuse_filter, large8) { // NOLINT
  auto keys   = load_sample();
  auto filter = binfuse::filter<binary_fuse8_t>(keys);
  EXPECT_TRUE(filter.verify(keys));
  EXPECT_LE(estimate_false_positive_rate(filter), 0.005);
}

TEST(binfuse_filter, large16) { // NOLINT
  auto keys   = load_sample();
  auto filter = binfuse::filter<binary_fuse16_t>(keys);
  EXPECT_TRUE(filter.verify(keys));
  EXPECT_LE(estimate_false_positive_rate(filter), 0.00005);
}

TEST(binfuse_filter, large8_persistent) { // NOLINT
  auto                  keys = load_sample();
  std::filesystem::path filter_path("tmp/filter.bin");
  {
    auto filter_sink = binfuse::filter8_sink(keys);
    filter_sink.save(filter_path);
    auto filter_source = binfuse::filter8_source();
    filter_source.load(filter_path);
    EXPECT_TRUE(filter_source.verify(keys));
    EXPECT_LE(estimate_false_positive_rate(filter_source), 0.005);
  }
  std::filesystem::remove(filter_path);
}

TEST(binfuse_filter, large16_persistent) { // NOLINT
  auto                  keys = load_sample();
  std::filesystem::path filter_path("tmp/filter.bin");
  {
    auto filter_sink = binfuse::filter16_sink(keys);
    filter_sink.save(filter_path);
    auto filter_source = binfuse::filter16_source();
    filter_source.load(filter_path);
    EXPECT_TRUE(filter_source.verify(keys));
    EXPECT_LE(estimate_false_positive_rate(filter_source), 0.005);
  }
  std::filesystem::remove(filter_path);
}
