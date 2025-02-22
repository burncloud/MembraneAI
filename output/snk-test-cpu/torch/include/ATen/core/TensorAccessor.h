#pragma once

#include <c10/macros/Macros.h>
#include <stdint.h>
#include <cstddef>

namespace at {

// The PtrTraits argument to the TensorAccessor/PackedTensorAccessor
// is used to enable the __restrict__ keyword/modifier for the data
// passed to cuda.
template <typename T>
struct DefaultPtrTraits {
  typedef T* PtrType;
};

#if defined(__CUDACC__) || defined(__HIPCC__)
template <typename T>
struct RestrictPtrTraits {
  typedef T* __restrict__ PtrType;
};
#endif

// TensorAccessorBase and TensorAccessor are used for both CPU and CUDA tensors.
// For CUDA tensors it is used in device code (only). This means that we restrict ourselves
// to functions and types available there (e.g. IntArrayRef isn't).

// The PtrTraits argument is only relevant to cuda to support `__restrict__` pointers.
template<typename T, size_t N, template <typename U> class PtrTraits = DefaultPtrTraits, typename index_t = int64_t>
class TensorAccessorBase {
public:
  typedef typename PtrTraits<T>::PtrType PtrType;

  C10_HOST_DEVICE TensorAccessorBase(
      PtrType data_,
      const index_t* sizes_,
      const index_t* strides_)
      : data_(data_), sizes_(sizes_), strides_(strides_) {}
  C10_HOST IntArrayRef sizes() const {
    return IntArrayRef(sizes_,N);
  }
  C10_HOST IntArrayRef strides() const {
    return IntArrayRef(strides_,N);
  }
  C10_HOST_DEVICE index_t stride(index_t i) const {
    return strides_[i];
  }
  C10_HOST_DEVICE index_t size(index_t i) const {
    return sizes_[i];
  }
  C10_HOST_DEVICE PtrType data() {
    return data_;
  }
  C10_HOST_DEVICE const PtrType data() const {
    return data_;
  }
protected:
  PtrType data_;
  const index_t* sizes_;
  const index_t* strides_;
};

// The `TensorAccessor` is typically instantiated for CPU `Tensor`s using
// `Tensor.accessor<T, N>()`.
// For CUDA `Tensor`s, `PackedTensorAccessor` is used on the host and only
// indexing on the device uses `TensorAccessor`s.
template<typename T, size_t N, template <typename U> class PtrTraits = DefaultPtrTraits, typename index_t = int64_t>
class TensorAccessor : public TensorAccessorBase<T,N,PtrTraits,index_t> {
public:
  typedef typename PtrTraits<T>::PtrType PtrType;

  C10_HOST_DEVICE TensorAccessor(
      PtrType data_,
      const index_t* sizes_,
      const index_t* strides_)
      : TensorAccessorBase<T, N, PtrTraits, index_t>(data_,sizes_,strides_) {}

  C10_HOST_DEVICE TensorAccessor<T, N - 1, PtrTraits, index_t> operator[](index_t i) {
    return TensorAccessor<T,N-1,PtrTraits,index_t>(this->data_ + this->strides_[0]*i,this->sizes_+1,this->strides_+1);
  }

  C10_HOST_DEVICE const TensorAccessor<T, N-1, PtrTraits, index_t> operator[](index_t i) const {
    return TensorAccessor<T,N-1,PtrTraits,index_t>(this->data_ + this->strides_[0]*i,this->sizes_+1,this->strides_+1);
  }
};

template<typename T, template <typename U> class PtrTraits, typename index_t>
class TensorAccessor<T,1,PtrTraits,index_t> : public TensorAccessorBase<T,1,PtrTraits,index_t> {
public:
  typedef typename PtrTraits<T>::PtrType PtrType;

  C10_HOST_DEVICE TensorAccessor(
      PtrType data_,
      const index_t* sizes_,
      const index_t* strides_)
      : TensorAccessorBase<T, 1, PtrTraits, index_t>(data_,sizes_,strides_) {}
  C10_HOST_DEVICE T & operator[](index_t i) {
    return this->data_[this->strides_[0]*i];
  }
  C10_HOST_DEVICE const T & operator[](index_t i) const {
    return this->data_[this->strides_[0]*i];
  }
};


// PackedTensorAccessorBase and PackedTensorAccessor are used on for CUDA `Tensor`s on the host
// and as
// In contrast to `TensorAccessor`s, they copy the strides and sizes on instantiation (on the host)
// in order to transfer them on the device when calling kernels.
// On the device, indexing of multidimensional tensors gives to `TensorAccessor`s.
// Use RestrictPtrTraits as PtrTraits if you want the tensor's data pointer to be marked as __restrict__.
// Instantiation from data, sizes, strides is only needed on the host and std::copy isn't available
// on the device, so those functions are host only.
template<typename T, size_t N, template <typename U> class PtrTraits = DefaultPtrTraits, typename index_t = int64_t>
class PackedTensorAccessorBase {
public:
  typedef typename PtrTraits<T>::PtrType PtrType;
  C10_HOST PackedTensorAccessorBase(
      PtrType data_,
      const index_t* sizes_,
      const index_t* strides_)
      : data_(data_) {
    std::copy(sizes_, sizes_ + N, std::begin(this->sizes_));
    std::copy(strides_, strides_ + N, std::begin(this->strides_));
  }

  // if index_t is not int64_t, we want to have an int64_t constructor
  template <typename source_index_t, class = typename std::enable_if<std::is_same<source_index_t, int64_t>::value>::type>
  C10_HOST PackedTensorAccessorBase(
      PtrType data_,
      const source_index_t* sizes_,
      const source_index_t* strides_)
      : data_(data_) {
    for (int i = 0; i < N; i++) {
      this->sizes_[i] = sizes_[i];
      this->strides_[i] = strides_[i];
    }
  }

  C10_HOST_DEVICE index_t stride(index_t i) const {
    return strides_[i];
  }
  C10_HOST_DEVICE index_t size(index_t i) const {
    return sizes_[i];
  }
  C10_HOST_DEVICE PtrType data() {
    return data_;
  }
  C10_HOST_DEVICE const PtrType data() const {
    return data_;
  }
protected:
  PtrType data_;
  index_t sizes_[N];
  index_t strides_[N];
};

template<typename T, size_t N, template <typename U> class PtrTraits = DefaultPtrTraits, typename index_t = int64_t>
class PackedTensorAccessor : public PackedTensorAccessorBase<T,N,PtrTraits,index_t> {
public:
  typedef typename PtrTraits<T>::PtrType PtrType;

  C10_HOST PackedTensorAccessor(
      PtrType data_,
      const index_t* sizes_,
      const index_t* strides_)
      : PackedTensorAccessorBase<T, N, PtrTraits, index_t>(data_, sizes_, strides_) {}

  // if index_t is not int64_t, we want to have an int64_t constructor
  template <typename source_index_t, class = typename std::enable_if<std::is_same<source_index_t, int64_t>::value>::type>
  C10_HOST PackedTensorAccessor(
      PtrType data_,
      const source_index_t* sizes_,
      const source_index_t* strides_)
      : PackedTensorAccessorBase<T, N, PtrTraits, index_t>(data_, sizes_, strides_) {}

  C10_DEVICE TensorAccessor<T, N - 1, PtrTraits, index_t> operator[](index_t i) {
    index_t* new_sizes = this->sizes_ + 1;
    index_t* new_strides = this->strides_ + 1;
    return TensorAccessor<T,N-1,PtrTraits,index_t>(this->data_ + this->strides_[0]*i, new_sizes, new_strides);
  }

  C10_DEVICE const TensorAccessor<T, N - 1, PtrTraits, index_t> operator[](index_t i) const {
    const index_t* new_sizes = this->sizes_ + 1;
    const index_t* new_strides = this->strides_ + 1;
    return TensorAccessor<T,N-1,PtrTraits,index_t>(this->data_ + this->strides_[0]*i, new_sizes, new_strides);
  }
};

template<typename T, template <typename U> class PtrTraits, typename index_t>
class PackedTensorAccessor<T,1,PtrTraits,index_t> : public PackedTensorAccessorBase<T,1,PtrTraits,index_t> {
public:
  typedef typename PtrTraits<T>::PtrType PtrType;
  C10_HOST PackedTensorAccessor(
      PtrType data_,
      const index_t* sizes_,
      const index_t* strides_)
      : PackedTensorAccessorBase<T, 1, PtrTraits, index_t>(data_, sizes_, strides_) {}

  // if index_t is not int64_t, we want to have an int64_t constructor
  template <typename source_index_t, class = typename std::enable_if<std::is_same<source_index_t, int64_t>::value>::type>
  C10_HOST PackedTensorAccessor(
      PtrType data_,
      const source_index_t* sizes_,
      const source_index_t* strides_)
      : PackedTensorAccessorBase<T, 1, PtrTraits, index_t>(data_, sizes_, strides_) {}

  C10_DEVICE T & operator[](index_t i) {
    return this->data_[this->strides_[0] * i];
  }
  C10_DEVICE const T& operator[](index_t i) const {
    return this->data_[this->strides_[0]*i];
  }
};

}
