#pragma once

// Test these using #if AT_CUDNN_ENABLED(), not #ifdef, so that it's
// obvious if you forgot to include Config.h
//    c.f. https://stackoverflow.com/questions/33759787/generating-an-error-if-checked-boolean-macro-is-not-defined
//
// NB: This header MUST NOT be included from other headers; it should
// only be included from C++ files.

#define AT_CUDNN_ENABLED() 1
#define AT_ROCM_ENABLED() 0

#define NVCC_FLAGS_EXTRA "-gencode;arch=compute_35,code=sm_35;-gencode;arch=compute_50,code=sm_50;-gencode;arch=compute_60,code=sm_60;-gencode;arch=compute_61,code=sm_61;-gencode;arch=compute_70,code=sm_70;-gencode;arch=compute_75,code=sm_75;-gencode;arch=compute_50,code=compute_50"
