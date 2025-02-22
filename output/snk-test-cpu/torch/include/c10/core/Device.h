#pragma once

#include <c10/core/DeviceType.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>

#include <cstddef>
#include <functional>
#include <iosfwd>
#include <string>

namespace c10 {

/// An index representing a specific device; e.g., the 1 in GPU 1.
/// A DeviceIndex is not independently meaningful without knowing
/// the DeviceType it is associated; try to use Device rather than
/// DeviceIndex directly.
using DeviceIndex = int16_t;

/// Represents a a compute device on which a tensor is located. A device is
/// uniquely identified by a type, which specifies the type of machine it is
/// (e.g. CPU or CUDA GPU), and a device index or ordinal, which identifies the
/// specific compute device when there is more than one of a certain type. The
/// device index is optional, and in its defaulted state represents (abstractly)
/// "the current device". Further, there are two constraints on the value of the
/// device index, if one is explicitly stored:
/// 1. A negative index represents the current device, a non-negative index
/// represents a specific, concrete device,
/// 2. When the device type is CPU, the device index must be zero.
struct C10_API Device final {
  using Type = DeviceType;

  /// Constructs a new `Device` from a `DeviceType` and an optional device
  /// index.
  /* implicit */ Device(DeviceType type, DeviceIndex index = -1)
      : type_(type), index_(index) {
    validate();
  }

  /// Constructs a `Device` from a string description, for convenience.
  /// The string supplied must follow the following schema:
  /// `(cpu|cuda)[:<device-index>]`
  /// where `cpu` or `cuda` specifies the device type, and
  /// `:<device-index>` optionally specifies a device index.
  /* implicit */ Device(const std::string& device_string);

  /// Returns true if the type and index of this `Device` matches that of
  /// `other`.
  bool operator==(const Device& other) const noexcept {
    return this->type_ == other.type_ && this->index_ == other.index_;
  }

  /// Returns true if the type or index of this `Device` differs from that of
  /// `other`.
  bool operator!=(const Device& other) const noexcept {
    return !(*this == other);
  }

  /// Sets the device index.
  void set_index(DeviceIndex index) {
    index_ = index;
  }

  /// Returns the type of device this is.
  DeviceType type() const noexcept {
    return type_;
  }

  /// Returns the optional index.
  DeviceIndex index() const noexcept {
    return index_;
  }

  /// Returns true if the device has a non-default index.
  bool has_index() const noexcept {
    return index_ != -1;
  }

  /// Return true if the device is of CUDA type.
  bool is_cuda() const noexcept {
    return type_ == DeviceType::CUDA;
  }

  /// Return true if the device is of CPU type.
  bool is_cpu() const noexcept {
    return type_ == DeviceType::CPU;
  }

 private:
  DeviceType type_;
  DeviceIndex index_ = -1;
  void validate() {
    TORCH_CHECK(index_ == -1 || index_ >= 0,
        "Device index must be -1 or non-negative, got ", index_);
    TORCH_CHECK(!is_cpu() || index_ <= 0,
        "CPU device index must be -1 or zero, got ", index_);
  }
};

C10_API std::ostream& operator<<(
    std::ostream& stream,
    const Device& device);

} // namespace c10

namespace std {
template <>
struct hash<c10::Device> {
  size_t operator()(c10::Device d) const noexcept {
    // Are you here because this static assert failed?  Make sure you ensure
    // that the bitmasking code below is updated accordingly!
    static_assert(sizeof(c10::DeviceType) == 2, "DeviceType is not 16-bit");
    static_assert(sizeof(c10::DeviceIndex) == 2, "DeviceIndex is not 16-bit");
    // Note [Hazard when concatenating signed integers]
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // We must first convert to a same-sized unsigned type, before promoting to
    // the result type, to prevent sign extension when any of the values is -1.
    // If sign extension occurs, you'll clobber all of the values in the MSB
    // half of the resulting integer.
    //
    // Technically, by C/C++ integer promotion rules, we only need one of the
    // uint32_t casts to the result type, but we put in both for explicitness's sake.
    uint32_t bits =
        static_cast<uint32_t>(static_cast<uint16_t>(d.type())) << 16
      | static_cast<uint32_t>(static_cast<uint16_t>(d.index()));
    return std::hash<uint32_t>{}(bits);
  }
};
} // namespace std
