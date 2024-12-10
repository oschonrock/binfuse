#include "binaryfusefilter.h"
#include "binfuse/filter.hpp"
#include "helpers.hpp"
#include "gtest/gtest.h"
#include <cstdint>

TEST(binfuse_filter, default_construct) { // NOLINT
  binfuse::filter8 fil;
  EXPECT_EQ(fil.size(), 0);
  EXPECT_FALSE(fil.is_populated());
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

