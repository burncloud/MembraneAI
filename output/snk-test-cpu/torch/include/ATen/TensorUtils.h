#pragma once

#include <ATen/Tensor.h>
#include <ATen/TensorGeometry.h>
#include <ATen/Utils.h>

// These functions are NOT in Utils.h, because this file has a dep on Tensor.h

namespace at {

// The following are utility functions for checking that arguments
// make sense.  These are particularly useful for native functions,
// which do NO argument checking by default.

struct CAFFE2_API TensorArg {
  Tensor tensor;
  const char* name;
  int pos; // 1-indexed
  TensorArg(Tensor tensor, const char* name, int pos)
    : tensor(std::move(tensor)), name(name), pos(pos) {}
  const Tensor* operator->() const { return &tensor; }
  const Tensor& operator*() const { return tensor; }
};

struct CAFFE2_API TensorGeometryArg {
  TensorGeometry tensor;
  const char* name;
  int pos; // 1-indexed
  /* implicit */ TensorGeometryArg(TensorArg arg)
    : tensor(TensorGeometry{arg.tensor}), name(arg.name), pos(arg.pos) {}
  TensorGeometryArg(TensorGeometry tensor, const char* name, int pos)
    : tensor(tensor), name(name), pos(pos) {}
  const TensorGeometry* operator->() const { return &tensor; }
  const TensorGeometry& operator*() const { return tensor; }
};

// A string describing which function did checks on its input
// arguments.
// TODO: Consider generalizing this into a call stack.
using CheckedFrom = const char*;

// The undefined convention: singular operators assume their arguments
// are defined, but functions which take multiple tensors will
// implicitly filter out undefined tensors (to make it easier to perform
// tests which should apply if the tensor is defined, and should not
// otherwise.)
//
// NB: This means that the n-ary operators take lists of TensorArg,
// not TensorGeometryArg, because the Tensor to TensorGeometry
// conversion will blow up if you have undefined tensors.

CAFFE2_API std::ostream& operator<<(std::ostream& out, TensorGeometryArg t);
CAFFE2_API void checkDim(
    CheckedFrom c,
    const TensorGeometryArg& t,
    int64_t dim);
// NB: this is an inclusive-exclusive range
CAFFE2_API void checkDimRange(
    CheckedFrom c,
    const TensorGeometryArg& t,
    int64_t dim_start,
    int64_t dim_end);
CAFFE2_API void checkSameDim(
    CheckedFrom c,
    const TensorGeometryArg& t1,
    const TensorGeometryArg& t2);
CAFFE2_API void checkContiguous(CheckedFrom c, const TensorGeometryArg& t);
CAFFE2_API void checkAllContiguous(CheckedFrom c, at::ArrayRef<TensorArg> ts);
CAFFE2_API void checkSize(
    CheckedFrom c,
    const TensorGeometryArg& t,
    IntArrayRef sizes);
CAFFE2_API void checkSize(
    CheckedFrom c,
    const TensorGeometryArg& t,
    int64_t dim,
    int64_t size);
CAFFE2_API void checkNumel(
    CheckedFrom c,
    const TensorGeometryArg& t,
    int64_t numel);
CAFFE2_API void checkSameNumel(
    CheckedFrom c,
    const TensorGeometryArg& t1,
    const TensorGeometryArg& t2);
CAFFE2_API void checkAllSameNumel(CheckedFrom c, ArrayRef<TensorArg> tensors);
CAFFE2_API void checkScalarType(
    CheckedFrom c,
    const TensorArg& t,
    ScalarType s);
CAFFE2_API void checkScalarTypes(
    CheckedFrom c,
    const TensorArg& t,
    at::ArrayRef<ScalarType> l);
CAFFE2_API void checkSameGPU(
    CheckedFrom c,
    const TensorArg& t1,
    const TensorArg& t2);
CAFFE2_API void checkAllSameGPU(CheckedFrom c, ArrayRef<TensorArg> tensors);
CAFFE2_API void checkSameType(
    CheckedFrom c,
    const TensorArg& t1,
    const TensorArg& t2);
CAFFE2_API void checkAllSameType(CheckedFrom c, ArrayRef<TensorArg> tensors);
CAFFE2_API void checkSameSize(
    CheckedFrom c,
    const TensorArg& t1,
    const TensorArg& t2);
CAFFE2_API void checkDefined(CheckedFrom c, const TensorArg& t);
CAFFE2_API void checkAllDefined(CheckedFrom c, at::ArrayRef<TensorArg> t);

// FixMe: does TensorArg slow things down?
CAFFE2_API void checkBackend(
    CheckedFrom c,
    at::ArrayRef<Tensor> t,
    at::Backend backend);

CAFFE2_API void checkDeviceType(
    CheckedFrom c,
    at::ArrayRef<Tensor> tensors,
    at::DeviceType device_type);

CAFFE2_API void checkLayout(CheckedFrom c, const Tensor& t, Layout layout);

CAFFE2_API void checkLayout(CheckedFrom c, at::ArrayRef<Tensor> tensors, at::Layout layout);

// Methods for getting data_ptr if tensor is defined
CAFFE2_API void* maybe_data_ptr(const Tensor& tensor);
CAFFE2_API void* maybe_data_ptr(const TensorArg& tensor);

// Return if the tensor geometry represented by `sizes` and `strides` is contiguous
// Although we cache is_contiguous in tensor now, this is till useful because it
// allows checking if a particular geometry is contiguous without explicitly
// constructing a tensor, e.g., when you want to choose a kernel strategy based
// on whether a subgeometry is contiguous.
CAFFE2_API bool geometry_is_contiguous(IntArrayRef sizes, IntArrayRef strides);

// Correspond to THCUNN_check_dim_size/THNN_check_dim_size
CAFFE2_API void check_dim_size(
    const Tensor& tensor,
    int64_t dim,
    int64_t dim_size,
    int64_t size);

namespace detail {
CAFFE2_API std::vector<int64_t> defaultStrides(IntArrayRef sizes);
CAFFE2_API int64_t computeStorageSize(IntArrayRef sizes, IntArrayRef strides);
CAFFE2_API c10::optional<std::vector<int64_t>> computeStride(
    IntArrayRef oldshape,
    IntArrayRef oldstride,
    IntArrayRef newshape);
} // namespace detail
} // namespace at
