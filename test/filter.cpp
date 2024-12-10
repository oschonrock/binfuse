#include "binaryfusefilter.h"
#include "binfuse/filter.hpp"
#include "helpers.hpp"
#include "mio/page.hpp"
#include "gtest/gtest.h"
#include <cstdint>

TEST(binfuse_filter, default_construct) { // NOLINT
  binfuse::filter8 fil;
  EXPECT_EQ(fil.size(), 0);
  EXPECT_FALSE(fil.is_populated());
}

TEST(binfuse_filter, default_construct_persistent) { // NOLINT
  binfuse::filter8_sink filter_sink;
  EXPECT_EQ(filter_sink.size(), 0);
  EXPECT_FALSE(filter_sink.is_populated());

  binfuse::filter8_source filter_source;
  EXPECT_EQ(filter_source.size(), 0);
  EXPECT_FALSE(filter_source.is_populated());
}

TEST(binfuse_filter, save_load8) { // NOLINT
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
  
  std::filesystem::remove("tmp/filter8.bin");
}

TEST(binfuse_filter, save_load16) { // NOLINT
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
  
  std::filesystem::remove("tmp/filter16.bin");
}

TEST(binfuse_filter, move) { // NOLINT
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
  
  std::filesystem::remove("tmp/filter8.bin");
}


// larger data tests

template <binfuse::filter_type FilterType>
void test_filter(std::span<const std::uint64_t> keys, double max_false_positive_rate) {
  auto filter = binfuse::filter<FilterType>(keys);
  EXPECT_TRUE(filter.verify(keys));
  EXPECT_LE(estimate_false_positive_rate(filter), max_false_positive_rate);
}

TEST(binfuse_filter, large8) { // NOLINT
  test_filter<binary_fuse8_t>(load_sample(), 0.005);
}

TEST(binfuse_filter, large16) { // NOLINT
  test_filter<binary_fuse16_t>(load_sample(), 0.00005);
}

