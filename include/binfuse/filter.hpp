#pragma once

#include "binaryfusefilter.h"
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <span>
#include <stdexcept>

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

  filter(const filter& other)          = delete;
  filter& operator=(const filter& rhs) = delete;

  filter(filter&& other) noexcept
      : size_(other.size_), fil(other.fil), skip_free_fingerprints(other.skip_free_fingerprints) {
    other.fil.Fingerprints = nullptr;
  }
  filter& operator=(filter&& rhs) noexcept {
    if (this != &rhs) *this = fil(std::move(rhs));
    return *this;
  }

  ~filter() {
    if (skip_free_fingerprints) {
      fil.Fingerprints = nullptr;
    }
    ftype<FilterType>::free(&fil);
  }

  void populate(std::span<const std::uint64_t> keys) {
    if (size() > 0) {
      throw std::runtime_error("filter is already populated. You must provide all data at once.");
    }
    size_ = keys.size();

    if (!ftype<FilterType>::allocate(keys.size(), &fil)) {
      throw std::runtime_error("failed to allocate memory.\n");
    }
    if (!ftype<FilterType>::populate(
            const_cast<std::uint64_t*>(keys.data()), // NOLINT const_cast until API changed
            keys.size(), &fil)) {
      throw std::runtime_error("failed to populate the filter");
    }
  }

  [[nodiscard]] bool contains(std::uint64_t needle) const {
    if (!is_populated()) {
      throw std::runtime_error("filter is not populated.");
    }
    return ftype<FilterType>::contains(needle, &fil);
  }

  [[nodiscard]] bool is_populated() const {
    return fil.SegmentCount > 0;
  }

  [[nodiscard]] std::size_t serialization_bytes() const {
    // upstream API should be const
    return ftype<FilterType>::serialization_bytes(const_cast<FilterType*>(&fil));
  }

  // caller provides and owns the buffer. Either malloc'd or a
  // writable mmap, typically.
  void serialize(char* buffer) const { ftype<FilterType>::serialize(&fil, buffer); }

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
    const char* fingerprints = ftype<FilterType>::deserialize_header(&fil, buffer);

    // set the freshly deserialized object's Fingerprint ptr (which is
    // where the bulk of the data is), to the byte immediately AFTER
    // the block where header bytes were deserialized FROM.
    fil.Fingerprints = // NOLINTNEXTLINE const_cast & rein_cast due to upstream API
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

  [[nodiscard]] std::size_t size() const { return size_; }

private:
  std::size_t size_ = 0;
  FilterType  fil{};
  bool        skip_free_fingerprints = false;
};

using filter8  = filter<binary_fuse8_t>;
using filter16 = filter<binary_fuse16_t>;

} // namespace binfuse
