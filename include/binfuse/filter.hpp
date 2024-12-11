#pragma once

#include "binaryfusefilter.h"
#include "mio/mmap.hpp"
#include "mio/page.hpp"
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

namespace binfuse {

template <typename T>
concept filter_type = std::same_as<T, binary_fuse8_t> || std::same_as<T, binary_fuse16_t>;

// select which functions on the C-API will be called with specialisations of the function ptrs

template <filter_type FilterType>
struct ftype {};

template <>
struct ftype<binary_fuse8_t> {
  static constexpr auto* allocate            = binary_fuse8_allocate;
  static constexpr auto* populate            = binary_fuse8_populate;
  static constexpr auto* contains            = binary_fuse8_contain;
  static constexpr auto* free                = binary_fuse8_free;
  static constexpr auto* serialization_bytes = binary_fuse8_serialization_bytes;
  static constexpr auto* serialize           = binary_fuse8_serialize;
  static constexpr auto* deserialize_header  = binary_fuse8_deserialize_header;
  using fingerprint_t                        = std::uint8_t;
};

template <>
struct ftype<binary_fuse16_t> {
  static constexpr auto* allocate            = binary_fuse16_allocate;
  static constexpr auto* populate            = binary_fuse16_populate;
  static constexpr auto* contains            = binary_fuse16_contain;
  static constexpr auto* free                = binary_fuse16_free;
  static constexpr auto* serialization_bytes = binary_fuse16_serialization_bytes;
  static constexpr auto* serialize           = binary_fuse16_serialize;
  static constexpr auto* deserialize_header  = binary_fuse16_deserialize_header;
  using fingerprint_t                        = std::uint16_t;
};

/* binfuse::filter
 *
 * wraps a single binary_fuse(8|16)_filter
 */
template <filter_type FilterType>
class filter {
public:
  filter() = default;
  explicit filter(std::span<const std::uint64_t> keys) { populate(keys); }

  // accepts an r-value reference of the upstream `binary_fuse(8|16)_filter` object
  // will take ownership of any allocated memory pointed to by the `Fingerprints` member
  // will free that memory when this object is destroyed
  explicit filter(FilterType&& fil) : fil_(fil) {}

  filter(const filter& other)          = delete;
  filter& operator=(const filter& rhs) = delete;

  filter(filter&& other) noexcept
      : fil_(other.fil_), skip_free_fingerprints(other.skip_free_fingerprints) {
    other.fil_.Fingerprints = nullptr; // this object now owns any memory
  }
  filter& operator=(filter&& rhs) noexcept {
    if (this != &rhs) *this = fil_(std::move(rhs));
    return *this;
  }

  ~filter() {
    if (skip_free_fingerprints) {
      fil_.Fingerprints = nullptr;
    }
    ftype<FilterType>::free(&fil_);
  }

  void populate(std::span<const std::uint64_t> keys) {
    if (is_populated()) {
      throw std::runtime_error("filter is already populated. You must provide all data at once.");
    }

    if (!ftype<FilterType>::allocate(static_cast<std::uint32_t>(keys.size()), &fil_)) {
      throw std::runtime_error("failed to allocate memory.\n");
    }
    if (!ftype<FilterType>::populate(
            const_cast<std::uint64_t*>(keys.data()), // NOLINT const_cast until API changed
            static_cast<std::uint32_t>(keys.size()), &fil_)) {
      throw std::runtime_error("failed to populate the filter");
    }
  }

  [[nodiscard]] bool contains(std::uint64_t needle) const {
    if (!is_populated()) {
      throw std::runtime_error("filter is not populated.");
    }
    return ftype<FilterType>::contains(needle, &fil_);
  }

  [[nodiscard]] bool is_populated() const { return fil_.SegmentCount > 0; }

  [[nodiscard]] std::size_t serialization_bytes() const {
    // upstream API should be const
    return ftype<FilterType>::serialization_bytes(const_cast<FilterType*>(&fil_));
  }

  // caller provides and owns the buffer. Either malloc'd or a
  // writable mmap, typically.
  void serialize(char* buffer) const { ftype<FilterType>::serialize(&fil_, buffer); }

  // Caller provides and owns the buffer. The lifetime of the buffer
  // must exceed the lifetime of this object.
  //
  // Specifically the lifetime of the `fingerprints`, which follow the
  // basic struct members, must also exceed any calls to
  // `this->contains|verify` etc.  If using an `mmap` to provide the
  // buffer, then the file must remain mapped while calling
  // `this->contains`.
  //
  // Not respecting this will likely result in segfaults.
  void deserialize(const char* buffer) {
    const char* fingerprints = ftype<FilterType>::deserialize_header(&fil_, buffer);

    // set the freshly deserialized object's Fingerprint ptr (which is
    // where the bulk of the data is), to the byte immediately AFTER
    // the block where header bytes were deserialized FROM.
    fil_.Fingerprints = // NOLINTNEXTLINE const_cast & rein_cast due to upstream API
        reinterpret_cast<ftype<FilterType>::fingerprint_t*>(const_cast<char*>(fingerprints));

    skip_free_fingerprints = true; // do not attempt to free this external buffer (probably an mmap)
  }

  // Check that each of the provided keys are `contain`ed in the
  // filter. Any false negative, immediately returns false with a
  // message to std::cerr.
  [[nodiscard]] bool verify(std::span<const std::uint64_t> keys) const {
    for (auto key: keys) {
      if (!contains(key)) {
        std::cerr << "binfuse::filter::verify: Detected a false negative: " << std::hex
                  << std::setfill('0') << std::setw(16) << key << '\n';
        return false;
      }
    }
    return true;
  }

private:
  FilterType fil_{};
  bool       skip_free_fingerprints = false;
};

template <filter_type FilterType, mio::access_mode AccessMode>
class persistent_filter : public filter<FilterType> {

public:
  using filter<FilterType>::filter;

  void save(std::filesystem::path filepath)
    requires(AccessMode == mio::access_mode::write)
  {
    filepath_ = std::move(filepath);
    if (!this->is_populated()) {
      throw std::runtime_error("not populated. nothing to save");
    }
    ensure_file();
    auto filesize = header_length + this->serialization_bytes();
    std::filesystem::resize_file(filepath_, filesize);
    map_whole_file();
    create_filetag();
    this->serialize(&mmap[header_length]);
    sync();
  }

  void load(std::filesystem::path filepath)
    requires(AccessMode == mio::access_mode::read)
  {
    filepath_ = std::move(filepath);
    map_whole_file();
    check_type_id();
    this->deserialize(&mmap[header_length]);
  }

private:
  mio::basic_mmap<AccessMode, char> mmap;
  std::filesystem::path             filepath_;

  using offset_t = std::size_t;

  static constexpr std::size_t header_start  = 0;
  static constexpr std::size_t header_length = 16;

  static constexpr std::size_t index_start = header_start + header_length;

  void copy_str_to_map(std::string value, offset_t offset)
    requires(AccessMode == mio::access_mode::write)
  {
    memcpy(&mmap[offset], value.data(), value.size());
  }

  [[nodiscard]] std::string get_str_from_map(offset_t offset, std::size_t strsize) const {
    std::string value;
    value.resize(strsize);
    memcpy(value.data(), &mmap[offset], strsize);
    return value;
  }

  [[nodiscard]] std::string type_id() const {
    std::string       type_id;
    std::stringstream type_id_stream(type_id);
    type_id_stream << "binfuse" << std::setfill('0') << std::setw(2)
                   << sizeof(typename ftype<FilterType>::fingerprint_t) * 8;
    return type_id_stream.str();
  }

  void sync()
    requires(AccessMode == mio::access_mode::write)
  {
    std::error_code err;
    mmap.sync(err); // ensure any existing map is sync'd
    if (err) {
      throw std::runtime_error("sharded_bin_fuse_filter:: mmap.map(): " + err.message());
    }
  }

  void map_whole_file() {
    std::error_code err;
    mmap.map(filepath_.string(), err); // does unmap then remap
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
    copy_str_to_map(type_id(), 0);
  }
};

using filter8  = filter<binary_fuse8_t>;
using filter16 = filter<binary_fuse16_t>;

using filter8_sink   = persistent_filter<binary_fuse8_t, mio::access_mode::write>;
using filter8_source = persistent_filter<binary_fuse8_t, mio::access_mode::read>;

using filter16_sink   = persistent_filter<binary_fuse16_t, mio::access_mode::write>;
using filter16_source = persistent_filter<binary_fuse16_t, mio::access_mode::read>;

} // namespace binfuse
