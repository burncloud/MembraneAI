#pragma once

#include <c10/core/impl/DeviceGuardImplInterface.h>

namespace c10 {
namespace impl {

/**
 * An implementation of DeviceGuardImplInterface which delegates
 * to virtual dispatch on the DeviceGuardImpl registry.
 */
class VirtualGuardImpl final : public DeviceGuardImplInterface {
public:
  VirtualGuardImpl(DeviceType device_type)
    : impl_(getDeviceGuardImpl(device_type)) {}
  // This constructor exists purely for testing
  VirtualGuardImpl(const DeviceGuardImplInterface* impl)
    : impl_(impl) {}

  // Copying and moving is OK!

  DeviceType type() const override {
    return impl_->type();
  }
  Device exchangeDevice(Device d) const override {
    return impl_->exchangeDevice(d);
  }
  Device getDevice() const override {
    return impl_->getDevice();
  }
  void setDevice(Device d) const override {
    impl_->setDevice(d);
  }
  void uncheckedSetDevice(Device d) const noexcept override {
    impl_->uncheckedSetDevice(d);
  }
  Stream getStream(Device d) const noexcept override {
    return impl_->getStream(d);
  }
  Stream exchangeStream(Stream s) const noexcept override {
    return impl_->exchangeStream(s);
  }
  DeviceIndex deviceCount() const noexcept override {
    return impl_->deviceCount();
  }
private:
  const DeviceGuardImplInterface* impl_ = nullptr;
};

}} // namespace c10::impl
