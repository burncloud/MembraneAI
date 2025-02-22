#ifndef THC_GENERAL_INC
#define THC_GENERAL_INC

#include <TH/THGeneral.h>
#include <TH/THAllocator.h>
#undef log10
#undef log1p
#undef log2
#undef expm1

#include <c10/cuda/CUDAStream.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cusparse.h>

#define USE_MAGMA

# define THC_EXTERNC extern "C"

// TH & THC are now part of the same library as ATen and Caffe2
#define THC_API THC_EXTERNC CAFFE2_API
#define THC_CLASS CAFFE2_API

#ifndef THAssert
#define THAssert(exp)                                                   \
  do {                                                                  \
    if (!(exp)) {                                                       \
      _THError(__FILE__, __LINE__, "assert(%s) failed", #exp);          \
    }                                                                   \
  } while(0)
#endif

typedef struct THCState THCState;
struct THCState;

typedef struct _THCCudaResourcesPerDevice {
  /* cuBLAS handle is lazily initialized */
  cublasHandle_t blasHandle;
  /* cuSparse handle is lazily initialized */
  cusparseHandle_t sparseHandle;
  /* Size of scratch space per each stream on this device available */
  size_t scratchSpacePerStream;
} THCCudaResourcesPerDevice;

THC_API THCState* THCState_alloc(void);
THC_API void THCState_free(THCState* state);

THC_API void THCudaInit(THCState* state);
THC_API void THCudaShutdown(THCState* state);

/* If device `dev` can access allocations on device `devToAccess`, this will return */
/* 1; otherwise, 0. */
THC_API int THCState_getPeerToPeerAccess(THCState* state, int dev, int devToAccess);

THC_API c10::Allocator* THCState_getCudaHostAllocator(THCState* state);

THC_API void THCMagma_init(THCState *state);

/* State manipulators and accessors */
THC_API int THCState_getNumDevices(THCState* state);

THC_API cudaStream_t THCState_getCurrentStreamOnDevice(THCState *state, int device);
THC_API cudaStream_t THCState_getCurrentStream(THCState *state);

/* BLAS and sparse handles */
THC_API cublasHandle_t THCState_getCurrentBlasHandle(THCState *state);
THC_API cusparseHandle_t THCState_getCurrentSparseHandle(THCState *state);

/* For the current device and stream, returns the allocated scratch space */
THC_API size_t THCState_getCurrentDeviceScratchSpaceSize(THCState* state);

#define THCAssertSameGPU(expr) if (!expr) THError("arguments are located on different GPUs")
#define THCudaCheck(err)  __THCudaCheck(err, __FILE__, __LINE__)
#define THCudaCheckWarn(err)  __THCudaCheckWarn(err, __FILE__, __LINE__)
#define THCublasCheck(err)  __THCublasCheck(err,  __FILE__, __LINE__)
#define THCusparseCheck(err)  __THCusparseCheck(err,  __FILE__, __LINE__)

THC_API void __THCudaCheck(cudaError_t err, const char *file, const int line);
THC_API void __THCudaCheckWarn(cudaError_t err, const char *file, const int line);
THC_API void __THCublasCheck(cublasStatus_t status, const char *file, const int line);
THC_API void __THCusparseCheck(cusparseStatus_t status, const char *file, const int line);

THC_API void* THCudaMalloc(THCState *state, size_t size);
THC_API void THCudaFree(THCState *state, void* ptr);

at::DataPtr THCudaHostAlloc(THCState *state, size_t size);

THC_API void THCudaHostRecord(THCState *state, void *ptr);

THC_API cudaError_t THCudaMemGetInfo(THCState *state, size_t* freeBytes, size_t* totalBytes, size_t* largestBlock);

#endif
