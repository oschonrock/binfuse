# binfuse: c++ library for binary fuse filters

Binary fuse filters are a recent (2022) development in the group of 
[Approximate Membership Query filters](https://en.wikipedia.org/wiki/Approximate_membership_query_filter)

> Approximate membership query filters (hereafter, AMQ filters)
> comprise a group of space-efficient probabilistic data structures
> that support approximate membership queries. An approximate
> membership query answers whether an element is in a set or not with
> a false positive rate of ϵ.

Binary fuse filters are a further development on XOR filters, which
are more space efficient, and faster to build and query than traditional
options like Bloom and Cookoo filters.

The `binfuse` C++ library builds on the
[C-libary](https://github.com/FastFilter/xor_singleheader) by the
authors of the [relevant research
paper](http://arxiv.org/abs/2201.01174).

As well as adding a convenient C++ interface, `binfuse::filter` also
includes serializing the populated filter to disk and querying it
directly from disk via an `mmap`, with cross platform support from
[mio](https://github.com/vimpunk/mio). Both in memory and "off disk"
operation is supported.

One of the challenges with binary fuse filters, is that they are
immutable once populated, so data cannot be added incrementally, and
they consume a significant amount of memory during the populate
process - 64GB of memory is recommended for population with 500
million `uint64_t` keys/hashes. This has, until now, placed an upward
bound on the practical application of these filters to very large
datasets.

`binfuse::sharded_filter` allows convenient slicing of the dataset
into an arbitrary number of `binfuse::filter`s which are written to
disk and indexed by the `N` most significant bits of the `uint64_t`
keys/hashes. Querying is then easily accomplished and still very fast
with just 3 `mmap` accesses per query.

`binfuse::sharded_filter` easily controls RAM requirements during the
filter populate process and enables datasets of 10s of billions of
records with common hardware. Query speeds depend on disk hardware and
cache conditions, but can be in the sub microsecond range.

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

Singular `binuse::filter` load and save:

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

Sharded filter, one shard at the time:

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
bit filters, giving 1/256 and 1/65535 chance of false positive, and
for the persisted version they are also templated by the
`mio::mmap::access_mode` to select source and sink:

```C++
namespace binfuse {

template <typename T>
concept filter_type = std::same_as<T, binary_fuse8_t> || std::same_as<T, binary_fuse16_t>;

template <filter_type FilterType>
class filter;




```

the following convenience aliases are provided:

```C++

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
```

=== Requirements and building

POSIX systems and Windows are supported, with a recent C++20 compiler (eg gcc 13.2, clang 18.1)

```bash
git clone https://github.com/oschonrock/binfuse.git
git submodule update --init --recursive

cmake -S . -B  build -DCMAKE_BUILD_TYPE=release 
cmake --build build
```

Tests are run automatically at the end of build.

=== Including in your project

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
