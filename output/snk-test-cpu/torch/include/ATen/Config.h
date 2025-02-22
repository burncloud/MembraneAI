#pragma once

// Test these using #if AT_MKL_ENABLED(), not #ifdef, so that it's
// obvious if you forgot to include Config.h
//    c.f. https://stackoverflow.com/questions/33759787/generating-an-error-if-checked-boolean-macro-is-not-defined
//
// DO NOT put the macros for CUDA libraries in this file; they belong in cuda/CUDAConfig.h

#define AT_MKLDNN_ENABLED() 0
#define AT_MKL_ENABLED() 1
#define AT_NNPACK_ENABLED() 0
#define CAFFE2_STATIC_LINK_CUDA() 0
