#pragma once

#include "binaryfusefilter.h"
#include "binfuse/filter.hpp"
#include "mio/mmap.hpp"
#include "mio/page.hpp"
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace binfuse {

// selecting the appropriate map for the access mode
template <mio::access_mode AccessMode>
struct sharded_mmap_base {
  mio::basic_mmap<AccessMode, char> mmap;
};

/* sharded_bin_fuse_filter.
 *
 * Wraps a set of `binfuse::filter`s.
 * Saves/loads them to/from an mmap'd file via mio::mmap.
 * Directs `contains` queries to the apropriate sub-filter.
 *
 */
template <filter_type FilterType, mio::access_mode AccessMode>
class sharded_filter : private sharded_mmap_base<AccessMode> {
public:
  sharded_filter() = default;
  explicit sharded_filter(std::filesystem::path path, std::uint8_t shard_bits = 8)
      : filepath_(std::move(path)), shard_bits_(shard_bits) {
    load();
  }

  // make a default constructor possible
  void set_filename(std::filesystem::path path, std::uint8_t shard_bits = 8) {
    filepath_   = std::move(path);
    shard_bits_ = shard_bits;
    load();
  }

  [[nodiscard]] bool contains(std::uint64_t needle) const {
    auto prefix = extract_prefix(needle);
    // we know prefix is always < max_shards() by definition
    auto& filter = filters[prefix];
    if (!filter.is_populated()) {
      // this filter has not been populated. no fingerprint pointer
      // has been set and an upstream `contain` call will throw
      return false;
    }
    return filter.contains(needle);
  }

  [[nodiscard]] std::uint32_t extract_prefix(std::uint64_t key) const {
    return static_cast<std::uint32_t>(key >> (sizeof(key) * 8 - shard_bits_));
  }

  void stream_prepare()
    requires(AccessMode == mio::access_mode::write)
  {
    stream_keys_.clear();
    stream_last_prefix_ = 0;
    stream_last_key_    = 0;
  }

  void stream_add(std::uint64_t key)
    requires(AccessMode == mio::access_mode::write)
  {
    if (key < stream_last_key_) {
      throw std::runtime_error("sharded_filter: stream_add: key out of order");
    }
    stream_last_key_ = key;
    auto prefix      = extract_prefix(key);
    if (prefix != stream_last_prefix_) {

      add(filter<FilterType>(stream_keys_), stream_last_prefix_);
      stream_keys_.clear();
      stream_last_prefix_ = prefix;
    }
    stream_keys_.emplace_back(key);
  }

  void stream_finalize()
    requires(AccessMode == mio::access_mode::write)
  {
    if (!stream_keys_.empty()) {
      add(filter<FilterType>(stream_keys_), stream_last_prefix_);
    }
  }

  void add(const filter<FilterType>& new_filter, std::uint32_t prefix)
    requires(AccessMode == mio::access_mode::write)
  {
    if (shards_ == max_shards()) {
      throw std::runtime_error("sharded filter has reached max_shards of " +
                               std::to_string(max_shards()));
    }

    std::size_t new_size = ensure_header();

    const std::size_t size_req          = new_filter.serialization_bytes();
    const offset_t    new_filter_offset = new_size; // place new filter at end
    new_size += size_req;

    sync();
    std::filesystem::resize_file(filepath_, new_size);
    map_whole_file();

    auto old_filter_offset = get_from_map<offset_t>(filter_index_offset(prefix));

    if (old_filter_offset != empty_offset) {
      throw std::runtime_error("there is already a filter in this file for prefix = " +
                               std::to_string(prefix));
    }
    copy_to_map(new_filter_offset, filter_index_offset(prefix)); // set up the index ptr
    new_filter.serialize(&this->mmap[new_filter_offset]);        // insert the data
    index[prefix] = new_filter_offset;
    ++shards_;
    sync(); // save all changes
    // because has been re-mapped above, need to reload all filters as ptrs to
    // Fingerprints will likely have changed
    load_filters();
  }

  [[nodiscard]] std::size_t shards() const { return shards_; }
  [[nodiscard]] std::size_t size() const { return size_; }

private:
  std::vector<std::size_t>        index;
  std::vector<filter<FilterType>> filters;
  std::filesystem::path           filepath_;
  std::uint8_t                    shard_bits_ = 8;
  std::uint32_t                   shards_     = 0;
  std::uint64_t                   size_       = 0;

  std::vector<std::uint64_t> stream_keys_;
  std::uint32_t              stream_last_prefix_ = 0;
  std::uint64_t              stream_last_key_    = 0;

  using offset_t                     = typename decltype(index)::value_type;
  static constexpr auto empty_offset = static_cast<offset_t>(-1);

  /*
   * `binfuse::sharded_filter` has to be file backed, because data is
   * presumably large enough to not fit in memory at least during
   * filter build (otherwise use a binfuse::filter)
   *
   * file structure is as follows:
   *
   * header [0 -> 16) : small number of bytes identifying the type of file, the
   * type of filters contained and how many shards are contained.
   *
   * index [16 -> 16 + 8 * max_shards() ): table of offsets to each
   * filter in the body. The offsets in the table are relative to the
   * start of the file.
   *
   * body [16 + 8 * max_shards() -> end ): the filters: each one has
   * the filter_struct_fields (ie the "header") followed by the large
   * array of (8 or 16bit) fingerprints. The offsets in the index will
   * point the start of the filter_heade, so that deserialize can be
   * called directly on that.
   *
   */

  static constexpr std::size_t header_start  = 0;
  static constexpr std::size_t header_length = 16;

  static constexpr std::size_t index_start = header_start + header_length;

  template <typename T>
  void copy_to_map(T value, offset_t offset)
    requires(AccessMode == mio::access_mode::write)
  {
    memcpy(&this->mmap[offset], &value, sizeof(T));
  }

  void copy_str_to_map(std::string value, offset_t offset)
    requires(AccessMode == mio::access_mode::write)
  {
    memcpy(&this->mmap[offset], value.data(), value.size());
  }

  template <typename T>
  [[nodiscard]] T get_from_map(offset_t offset) const {
    T value;
    memcpy(&value, &this->mmap[offset], sizeof(T));
    return value;
  }

  [[nodiscard]] std::string get_str_from_map(offset_t offset, std::size_t strsize) const {
    std::string value;
    value.resize(strsize);
    memcpy(value.data(), &this->mmap[offset], strsize);
    return value;
  }

  [[nodiscard]] std::uint32_t max_shards() const { return 1U << shard_bits_; }

  [[nodiscard]] std::size_t index_length() const { return sizeof(std::size_t) * max_shards(); }

  [[nodiscard]] std::size_t filter_index_offset(std::uint32_t prefix) const {
    return index_start + sizeof(std::size_t) * prefix;
  }

  [[nodiscard]] offset_t filter_offset(std::uint32_t prefix) const {
    return get_from_map<offset_t>(filter_index_offset(prefix));
  }

  [[nodiscard]] std::string type_id() const {
    std::string       type_id;
    std::stringstream type_id_stream(type_id);
    type_id_stream << "sbinfuse" << std::setfill('0') << std::setw(2)
                   << sizeof(typename ftype<FilterType>::fingerprint_t) * 8;
    return type_id_stream.str();
  }

  void sync()
    requires(AccessMode == mio::access_mode::write)
  {
    std::error_code err;
    this->mmap.sync(err); // ensure any existing map is sync'd
    if (err) {
      throw std::runtime_error("sharded_bin_fuse_filter:: mmap.map(): " + err.message());
    }
  }

  void map_whole_file() {
    std::error_code err;
    this->mmap.map(filepath_.string(), err); // does unmap then remap
    if (err) {
      throw std::runtime_error("sharded_bin_fuse_filter:: mmap.map(): " + err.message());
    }
  }

  void check_type_id() const {
    auto tid       = type_id();
    auto check_tid = get_str_from_map(0, tid.size());
    if (check_tid != tid) {
      throw std::runtime_error("incorrect type_id: expected: " + tid + ", found: " + check_tid);
    }
  }

  void check_max_shards() const {
    std::uint32_t check_max_shards = 0;
    std::from_chars(&this->mmap[11], &this->mmap[15], check_max_shards);
    if (check_max_shards != max_shards()) {
      throw std::runtime_error("wrong capacity: expected: " + std::to_string(max_shards()) +
                               ", found: " + std::to_string(check_max_shards));
    }
  }

  // returns existing file size
  std::size_t ensure_file()
    requires(AccessMode == mio::access_mode::write)
  {
    if (filepath_.empty()) {
      throw std::runtime_error("filename not set or file doesn't exist: '" + filepath_.string() +
                               "'");
    }

    std::size_t existing_filesize = 0;
    if (std::filesystem::exists(filepath_)) {
      existing_filesize = std::filesystem::file_size(filepath_);
    } else {
      const std::ofstream tmp(filepath_); // "touch"
    }
    return existing_filesize;
  }

  void create_filetag()
    requires(AccessMode == mio::access_mode::write)
  {
    std::string       tagstr;
    std::stringstream tagstream(tagstr);
    tagstream << type_id() << '-' << std::setfill('0') << std::setw(4) << max_shards();
    copy_str_to_map(tagstream.str(), 0);
  }

  void create_index()
    requires(AccessMode == mio::access_mode::write)
  {
    index.resize(max_shards(), empty_offset);
    memcpy(&this->mmap[index_start], index.data(), index.size() * sizeof(offset_t));
  }

  void load_index() {
    index.resize(max_shards(), empty_offset);
    memcpy(index.data(), &this->mmap[index_start], index.size() * sizeof(offset_t));
    shards_ = static_cast<std::uint32_t>(
        count_if(index.begin(), index.end(), [](auto a) { return a != empty_offset; }));
  }

  void load_filters() {
    // always "load" all, even if as yet unpopulated
    filters.clear();
    filters.resize(max_shards()); // default constructed, which zeroes all values
    for (uint32_t prefix = 0; prefix != max_shards(); ++prefix) {
      if (auto offset = index[prefix]; offset != empty_offset) {
        filters[prefix].deserialize(&this->mmap[offset]);
      }
    }
  }

  // only does something for AccessMode = read
  // if you write a filter, you must reopen it in read mode
  void load() {
    if constexpr (AccessMode == mio::access_mode::write) {
      if (!std::filesystem::exists(filepath_)) {
        ensure_file();
        ensure_header();
        return;
      }
    }
    map_whole_file(); // read mode will fail here if not exists
    check_type_id();
    check_max_shards();
    load_index();
    load_filters();
  }

  // returns new_filesize
  std::size_t ensure_header()
    requires(AccessMode == mio::access_mode::write)
  {
    const std::size_t existing_filesize = ensure_file();
    std::size_t       new_size          = existing_filesize;
    if (existing_filesize < header_length + index_length()) {
      if (existing_filesize != 0) {
        throw std::runtime_error("corrupt file: header and index half written?!");
      }
      // existing_size == 0 here
      new_size += header_length + index_length();
      std::filesystem::resize_file(filepath_, new_size);
      map_whole_file();
      create_filetag();
      create_index();
      sync(); // write to disk
      load_filters();
      shards_ = 0;
    } else {
      // we have a header already
      map_whole_file();
      check_type_id();
      check_max_shards();
      load_index();
      load_filters();
    }
    return new_size;
  }
};

// easy to use aliases
using sharded_filter8_sink = sharded_filter<binary_fuse8_t, mio::access_mode::write>;

using sharded_filter8_source = sharded_filter<binary_fuse8_t, mio::access_mode::read>;

using sharded_filter16_sink = sharded_filter<binary_fuse16_t, mio::access_mode::write>;

using sharded_filter16_source = sharded_filter<binary_fuse16_t, mio::access_mode::read>;

} // namespace binfuse
