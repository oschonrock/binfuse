# `binfuse` C++ Library for Binary Fuse Filters

Binary fuse filters are a recent (2022) development in the group of 
[Approximate Membership Query filters](https://en.wikipedia.org/wiki/Approximate_membership_query_filter)

> Approximate membership query filters (hereafter, AMQ filters)
> comprise a group of space-efficient probabilistic data structures
> that support approximate membership queries. An approximate
> membership query answers whether an element is in a set or not with
> a false positive rate of ϵ.

Binary fuse filters are a further development of XOR filters. Both
are more space efficient, and faster to build/query than traditional
options like Bloom and Cuckoo filters.

This [`binfuse` C++ library](https://github.com/oschonrock/binfuse)
builds on the
[C-libary](https://github.com/FastFilter/xor_singleheader) by the
authors of the [relevant research
paper](http://arxiv.org/abs/2201.01174).

As well as adding a convenient C++ interface, `binfuse::filter` also
facilitates (de-)serializing the populated filter to/from disk as well
as querying it directly from disk via an `mmap`, with cross platform
support from [mio](https://github.com/vimpunk/mio). Both in memory and
"off disk" operation is supported.

One of the challenges with binary fuse filters, is that they are
immutable once populated, so data cannot be added incrementally, and
they consume a significant amount of memory during the populate
process - 64GB of memory is recommended for populating with 500
million `uint64_t` keys/hashes. This has, until now, placed an upward
bound on the practical application of these filters to very large
datasets.

`binfuse::sharded_filter` allows convenient slicing of the dataset
into an arbitrary number of shards which are written to
disk and indexed by the `N` most significant bits of the `uint64_t`
keys/hashes. Sharding is transparent to the user during queries is and
still very fast with just 3 `mmap` accesses per query.

`binfuse::sharded_filter` easily controls RAM requirements during the
"populate filter" process and enables datasets of 10s of billions of
records with common hardware. Query speeds depend on disk hardware and
cache conditions, but can be in the 50ns range.

## Usage examples from [tests](https://github.com/oschonrock/binfuse/tree/main/test)

Singular `binuse::filter` for in memory use:

```C++
binfuse::filter8 filter(std::vector<std::uint64_t>{
    0x0000000000000000,
    0x0000000000000001, // order is not important
    0x0000000000000002,
});
EXPECT_TRUE(filter.is_populated());

EXPECT_TRUE(filter.contains(0x0000000000000000));
EXPECT_TRUE(filter.contains(0x0000000000000001));
EXPECT_TRUE(filter.contains(0x0000000000000002));
```

Singular `binuse::filter` save and load:

```C++
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
```

Sharded filter, bulding one shard at the time:

```C++
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

sink.add_shard(tiny_low, 0);  // specify the prefix for each shard
sink.add_shard(tiny_high, 1); // order of adding is not important

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
```

or via "streaming" API:

```C++
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
```

The main classes are templated as follows to select underlying 8 or 16
bit filters, giving a 1/256 and 1/65536 chance of a false positive
respectively. The persistent versions are also templated by the
`mio::mmap::access_mode` to select read or write (AKA source or sink):

```C++
namespace binfuse {

template <typename T>
concept filter_type = std::same_as<T, binary_fuse8_t> || std::same_as<T, binary_fuse16_t>;


template <filter_type FilterType>
class filter;

template <filter_type FilterType, mio::access_mode AccessMode>
class persistent_filter;


template <filter_type FilterType, mio::access_mode AccessMode>
class sharded_filter;

} // namespace binfuse
```

the following convenience aliases are provided:

```C++
namespace binfuse {

// binfuse::filter

using filter8  = filter<binary_fuse8_t>;
using filter16 = filter<binary_fuse16_t>;

using filter8_sink   = persistent_filter<binary_fuse8_t, mio::access_mode::write>;
using filter8_source = persistent_filter<binary_fuse8_t, mio::access_mode::read>;

using filter16_sink   = persistent_filter<binary_fuse16_t, mio::access_mode::write>;
using filter16_source = persistent_filter<binary_fuse16_t, mio::access_mode::read>;

// binfuse::sharded_filter

using sharded_filter8_sink = sharded_filter<binary_fuse8_t, mio::access_mode::write>;
using sharded_filter8_source = sharded_filter<binary_fuse8_t, mio::access_mode::read>;

using sharded_filter16_sink = sharded_filter<binary_fuse16_t, mio::access_mode::write>;
using sharded_filter16_source = sharded_filter<binary_fuse16_t, mio::access_mode::read>;

} // namespace binfuse

```

### Requirements and building

POSIX systems and Windows(mingw) are supported, with a recent C++20 compiler (eg gcc 13.2, clang 18.1)

```bash
git clone https://github.com/oschonrock/binfuse.git
git submodule update --init --recursive

cmake -S . -B  build -DCMAKE_BUILD_TYPE=release 
cmake --build build
```

Tests are run automatically at the end of build. They can be disabled with `-DBINFUSE_TEST=OFF`.

### Including in your project

The library, and its included upstream C-library are header only, but
there is a `cmake` interface target available:

```cmake
add_subdirectory(ext/binfuse)

add_executable(my_exe main.cpp)
target_link_libraries(my_exe PRIVATE binfuse)

```


```c++
// main.cpp

#include "binfuse/sharded_filter.hpp"
#include "binfuse/filter.hpp"

...

```

### A note on file formats and tags

Two different binary formats are uses for `filter` and
`sharded_filter`. They each have differnt build parameters, which
further affect the structure. These are `fingerprint` size (8 or
16bit) and, in the case of the sharded filter, the number of
`shards`. 

The file format parameter are recorded in the first 16 bytes of each
file, so that files are not accidentally opened with the wrong
parameters, giving bogus results. These tags are human readable and
can be inspected with `hexdump`.

```bash
$ hd filter8.bin | head -n2
00000000  73 62 69 6e 66 75 73 65  30 38 2d 30 30 36 34 00  |sbinfuse08-0064.|
00000010  10 02 00 00 00 00 00 00  28 02 0d 01 00 00 00 00  |........(.......|

$ hd filter16.bin | head -n2
00000000  73 62 69 6e 66 75 73 65  31 36 2d 30 30 36 34 00  |sbinfuse16-0064.|
00000010  10 02 00 00 00 00 00 00  2c 02 1a 02 00 00 00 00  |........,.......|
```

`binfuse` will throw exceptions, if files of wrong type or format are opened. 

### Benchmarks

There is a benchmark for the `sharded_filter`, which includes figures
for the `filter` internally. This can be built with
`-DBINFUSE_BENCH=ON`. Some results below for 100m keys. 

It is difficult to run due to memory pressure for 1billion or more,
unless the first runs, with low shard count, are pruned out,
demonstrating the precise motivation for the sharded_filter.

Edit for your needs.

```
$ ./build/binfuse_bench_large

Shard Size: 50000000  Shards: 2  Keys: 100000000

           gen   populate     verify        add      query       f+ve
f8      16.3ns     87.2ns     33.9ns      5.3ns     48.3ns  0.390792%
f16     16.3ns     93.5ns     39.3ns     10.5ns     50.8ns  0.001542%


Shard Size: 25000000  Shards: 4  Keys: 100000000

           gen   populate     verify        add      query       f+ve
f8      12.4ns     87.1ns     30.7ns      5.4ns     47.7ns  0.390143%
f16     12.6ns     87.6ns     33.6ns     10.6ns     50.1ns  0.001487%


Shard Size: 12500000  Shards: 8  Keys: 100000000

           gen   populate     verify        add      query       f+ve
f8       9.8ns     70.6ns     28.1ns      5.6ns     48.2ns  0.390789%
f16     11.0ns     73.2ns     31.3ns     11.0ns     51.9ns  0.001483%


Shard Size: 6250000  Shards: 16  Keys: 100000000

           gen   populate     verify        add      query       f+ve
f8       5.6ns     69.6ns     23.7ns      6.0ns     47.6ns  0.390566%
f16      5.7ns     70.6ns     28.4ns     11.6ns     50.6ns  0.001511%


Shard Size: 3125000  Shards: 32  Keys: 100000000

           gen   populate     verify        add      query       f+ve
f8       5.1ns     60.4ns     11.7ns      6.8ns     46.7ns  0.390106%
f16      5.0ns     64.8ns     23.3ns     12.1ns     50.6ns  0.001576%


Shard Size: 1562500  Shards: 64  Keys: 100000000

           gen   populate     verify        add      query       f+ve
f8       4.7ns     59.2ns      6.2ns      8.2ns     46.1ns  0.390199%
f16      4.7ns     59.8ns     11.9ns     14.0ns     52.6ns  0.001496%


Shard Size: 781250  Shards: 128  Keys: 100000000

           gen   populate     verify        add      query       f+ve
f8       4.7ns     56.6ns      6.2ns     10.2ns     46.4ns  0.390504%
f16      4.7ns     54.6ns      6.4ns     17.1ns     48.4ns  0.001524%


Shard Size: 390625  Shards: 256  Keys: 100000000

           gen   populate     verify        add      query       f+ve
f8       4.6ns     48.7ns      5.7ns     15.1ns     47.5ns  0.390732%
f16      4.6ns     48.9ns      6.3ns     22.2ns     46.7ns  0.001484%
```

#### A note on memory consumption

The first few runs of the above benchmark, with low shard count, will
consume large, transient amounts of memory (1-2GB range) during
`filter::populate`. The latter runs show the benefit of the
`sharded_filter`, and the max memory consumption will settle at the
size of the filter file (108MB/215MB for 8/16bit).

**Note** that even this latter memory consumption is almost all in the
`mmap`. The OS is likely deciding to just let your process cache the
mmap fully in memory. This obviously benefits performance, but it is
flexible disk cache. If the machine comes under memory pressure, these
pages will be evicted.
