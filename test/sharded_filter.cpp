#include "binfuse/sharded_filter.hpp"
#include "binaryfusefilter.h"
#include "binfuse/filter.hpp"
#include "helpers.hpp"
#include "mio/page.hpp"
#include "gtest/gtest.h"
#include <cstdint>
#include <filesystem>
#include <span>
#include <stdexcept>
#include <vector>

TEST(binfuse_sfilter, default_construct) { // NOLINT
  binfuse::sharded_filter8_source sharded_source;
  EXPECT_EQ(sharded_source.shards(), 0);
}

TEST(binfuse_sfilter, add_tiny) { // NOLINT
  {
    binfuse::filter8 tiny_low(std::vector<std::uint64_t>{
        // note the MSB is clear on all below
        0x0000000000000000,
        0x0000000000000001, // order is not important
        0x0000000000000002,
    });

    binfuse::filter8 tiny_high(std::vector<std::uint64_t>{
        // note the MSB is set on all below
        0x8000000000000000,
        0x8000000000000001, // order is not important
        0x8000000000000002,
    });

    binfuse::sharded_filter8_sink sink("tmp/sharded_filter8_tiny.bin",
                                                    1); // one bit sharding, ie 2 shards

    sink.add(tiny_low, 0);  // specify the prefix for each shard
    sink.add(tiny_high, 1); // order of adding is not important

    EXPECT_EQ(sink.shards(), 2);

    // now reopen the filter as a "source"
    binfuse::sharded_filter8_source source("tmp/sharded_filter8_tiny.bin", 1);

    // verify all entries
    EXPECT_TRUE(source.contains(0x0000000000000000));
    EXPECT_TRUE(source.contains(0x0000000000000001));
    EXPECT_TRUE(source.contains(0x0000000000000002));
    EXPECT_TRUE(source.contains(0x8000000000000000));
    EXPECT_TRUE(source.contains(0x8000000000000001));
    EXPECT_TRUE(source.contains(0x8000000000000002));

    EXPECT_EQ(source.shards(), 2);
  }
  // cleanup
  std::filesystem::remove("tmp/sharded_filter8_tiny.bin");
}

TEST(binfuse_sfilter, add_ooo) { // NOLINT
  {
    binfuse::filter8 tiny_low( // out of order elements are permissible
        std::vector<std::uint64_t>{0x0000000000000002, 0x0000000000000000, 0x0000000000000001});

    binfuse::filter8 tiny_high( // out of order elements are permissible
        std::vector<std::uint64_t>{0x8000000000000001, 0x8000000000000002, 0x8000000000000000});

    binfuse::sharded_filter<binary_fuse8_t, mio::access_mode::write> sink(
        "tmp/sharded_filter8_tiny.bin", 1);

    // adding shards out of order is also permissible, alhtough may result in
    // very slightly suboptimal disk layout of the filter
    sink.add(tiny_high, 1);
    sink.add(tiny_low, 0);

    binfuse::sharded_filter<binary_fuse8_t, mio::access_mode::read> source(
        "tmp/sharded_filter8_tiny.bin", 1);

    EXPECT_TRUE(source.contains(0x0000000000000000));
    EXPECT_TRUE(source.contains(0x0000000000000001));
    EXPECT_TRUE(source.contains(0x0000000000000002));
    EXPECT_TRUE(source.contains(0x8000000000000000));
    EXPECT_TRUE(source.contains(0x8000000000000001));
    EXPECT_TRUE(source.contains(0x8000000000000002));
  }
  std::filesystem::remove("tmp/sharded_filter8_tiny.bin");
}

TEST(binfuse_sfilter, missing_shard) { // NOLINT
  {
    binfuse::filter8 tiny_high(
        std::vector<std::uint64_t>{0x8000000000000000, 0x8000000000000001, 0x8000000000000002});

    binfuse::sharded_filter<binary_fuse8_t, mio::access_mode::write> sink(
        "tmp/sharded_filter8_tiny.bin", 1);

    // only add a `high` shard with prefix = 1, omit prefix = 0
    sink.add(tiny_high, 1);
    EXPECT_EQ(sink.shards(), 1);

    binfuse::sharded_filter<binary_fuse8_t, mio::access_mode::read> source(
        "tmp/sharded_filter8_tiny.bin", 1);

    // try to find an element in the missing low shard => always false
    EXPECT_FALSE(source.contains(0x0000000000000000));
  }
  std::filesystem::remove("tmp/sharded_filter8_tiny.bin");
}

TEST(binfuse_sfilter, empty_shard) { // NOLINT
  {
    binfuse::filter8 tiny_high(std::vector<std::uint64_t>{});

    binfuse::sharded_filter<binary_fuse8_t, mio::access_mode::write> sink(
        "tmp/sharded_filter8_tiny.bin", 1);

    // only add a `high` shard with prefix = 1, omit prefix = 0
    sink.add(tiny_high, 1);

    binfuse::sharded_filter<binary_fuse8_t, mio::access_mode::read> source(
        "tmp/sharded_filter8_tiny.bin", 1);

    // try to find an element in the missing low shard
    EXPECT_FALSE(source.contains(0x8000000000000000));
  }
  std::filesystem::remove("tmp/sharded_filter8_tiny.bin");
}

TEST(binfuse_sfilter, read_sink_directly) { // NOLINT
  {
    binfuse::filter8 tiny_low(std::vector<std::uint64_t>{
        // note the MSB is clear on all below
        0x0000000000000000,
        0x0000000000000001, // order is not important
        0x0000000000000002,
    });

    binfuse::filter8 tiny_high(std::vector<std::uint64_t>{
        // note the MSB is set on all below
        0x8000000000000000,
        0x8000000000000001, // order is not important
        0x8000000000000002,
    });

    binfuse::sharded_filter8_sink sink("tmp/sharded_filter8_tiny.bin",
                                                    1); // one bit sharding, ie 2 shards

    sink.add(tiny_low, 0);  // specify the prefix for each shard
    sink.add(tiny_high, 1); // order of adding is not important

    // verify all entries directly in sink
    EXPECT_TRUE(sink.contains(0x0000000000000000));
    EXPECT_TRUE(sink.contains(0x0000000000000001));
    EXPECT_TRUE(sink.contains(0x0000000000000002));
    EXPECT_TRUE(sink.contains(0x8000000000000000));
    EXPECT_TRUE(sink.contains(0x8000000000000001));
    EXPECT_TRUE(sink.contains(0x8000000000000002));

    EXPECT_EQ(sink.shards(), 2);
  }
  // cleanup
  std::filesystem::remove("tmp/sharded_filter8_tiny.bin");
}

TEST(binfuse_sfilter, read_sink_after_load) { // NOLINT
  {
    binfuse::filter8 tiny_low(std::vector<std::uint64_t>{
        // note the MSB is clear on all below
        0x0000000000000000,
        0x0000000000000001, // order is not important
        0x0000000000000002,
    });

    binfuse::filter8 tiny_high(std::vector<std::uint64_t>{
        // note the MSB is set on all below
        0x8000000000000000,
        0x8000000000000001, // order is not important
        0x8000000000000002,
    });

    binfuse::sharded_filter8_sink sink("tmp/sharded_filter8_tiny.bin",
                                                    1); 

    sink.add(tiny_low, 0);
    sink.add(tiny_high, 1);

    binfuse::sharded_filter8_sink sink2("tmp/sharded_filter8_tiny.bin",
                                                     1);
    // verify all entries directly in sink
    EXPECT_TRUE(sink2.contains(0x0000000000000000));
    EXPECT_TRUE(sink2.contains(0x0000000000000001));
    EXPECT_TRUE(sink2.contains(0x0000000000000002));
    EXPECT_TRUE(sink2.contains(0x8000000000000000));
    EXPECT_TRUE(sink2.contains(0x8000000000000001));
    EXPECT_TRUE(sink2.contains(0x8000000000000002));

    EXPECT_EQ(sink.shards(), 2);
  }
  // cleanup
  std::filesystem::remove("tmp/sharded_filter8_tiny.bin");
}

TEST(binfuse_sfilter, stream_tiny) { // NOLINT
  {
    binfuse::sharded_filter<binary_fuse8_t, mio::access_mode::write> sink(
        "tmp/sharded_filter8_tiny.bin", 1);

    // alternative "streaming" API for bullding the filter
    // the entries below must be strictly in order
    sink.stream_prepare();
    sink.stream_add(0x0000000000000000);
    sink.stream_add(0x0000000000000001);
    sink.stream_add(0x0000000000000002);
    sink.stream_add(0x8000000000000000);
    sink.stream_add(0x8000000000000001);
    sink.stream_add(0x8000000000000002);
    sink.stream_finalize();

    binfuse::sharded_filter<binary_fuse8_t, mio::access_mode::read> source(
        "tmp/sharded_filter8_tiny.bin", 1);

    EXPECT_TRUE(source.contains(0x0000000000000000));
    EXPECT_TRUE(source.contains(0x0000000000000001));
    EXPECT_TRUE(source.contains(0x0000000000000002));
    EXPECT_TRUE(source.contains(0x8000000000000000));
    EXPECT_TRUE(source.contains(0x8000000000000001));
    EXPECT_TRUE(source.contains(0x8000000000000002));
  }
  std::filesystem::remove("tmp/sharded_filter8_tiny.bin");
}

TEST(binfuse_sfilter, stream_ooo) { // NOLINT
  binfuse::sharded_filter8_sink sink("tmp/sharded_filter8_tiny.bin", 1);

  // alternative "streaming" API for bullding the filter
  // the entries below must be strictly in order
  sink.stream_prepare();
  sink.stream_add(0x0000000000000001);
  // out of order add
  EXPECT_THROW(sink.stream_add(0x0000000000000000), std::runtime_error);
}

TEST(binfuse_sfilter, load_tiny) { // NOLINT
  binfuse::sharded_filter8_source source;
  EXPECT_THROW(source.set_filename("non_existant.bin"), std::runtime_error);

  // wrong `shard_bits`.. which defaults to `8`, but file ws created with `2`
  EXPECT_THROW(source.set_filename("data/sharded_filter8_tiny.bin"),
               std::runtime_error);

  source.set_filename("data/sharded_filter8_tiny.bin", 1); // correct capacity

  EXPECT_TRUE(source.contains(0x0000000000000002));
  EXPECT_TRUE(source.contains(0x8000000000000000));
}

// larger data tests

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

TEST(binfuse_sfilter, large8) { // NOLINT
  test_sharded_filter<binary_fuse8_t>(load_sample(), 0.005);
}

TEST(binfuse_sfilter, large16) { // NOLINT
  test_sharded_filter<binary_fuse16_t>(load_sample(), 0.00005);
}

TEST(binfuse_sfilter, large8_32) {                              // NOLINT
  test_sharded_filter<binary_fuse8_t>(load_sample(), 0.005, 5); // 5 sharded_bits
}

TEST(binfuse_sfilter, large16_32) {                                // NOLINT
  test_sharded_filter<binary_fuse16_t>(load_sample(), 0.00005, 5); // 5 sharded_bits
}
