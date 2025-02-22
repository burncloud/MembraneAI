#pragma once

// @generated by aten/src/ATen/gen.py

#include <ATen/ATen.h>

namespace at {
namespace native {
namespace legacy {
namespace cpu {

Tensor & _th_set_(Tensor & self, Storage source);
Tensor & _th_set_(Tensor & self, Storage source, int64_t storage_offset, IntArrayRef size, IntArrayRef stride);
Tensor & _th_set_(Tensor & self, const Tensor & source);
Tensor & _th_set_(Tensor & self);
Tensor & _th_fill_(Tensor & self, Scalar value);
Tensor & _th_fill_(Tensor & self, const Tensor & value);
bool _th_is_set_to(const Tensor & self, const Tensor & tensor);
Tensor & _th_masked_fill_(Tensor & self, const Tensor & mask, Scalar value);
Tensor & s__th_masked_fill_(Tensor & self, const Tensor & mask, Scalar value);
Tensor & _th_masked_fill_(Tensor & self, const Tensor & mask, const Tensor & value);
Tensor & s__th_masked_fill_(Tensor & self, const Tensor & mask, const Tensor & value);
Tensor & _th_masked_fill_bool_(Tensor & self, const Tensor & mask, Scalar value);
Tensor & s__th_masked_fill_bool_(Tensor & self, const Tensor & mask, Scalar value);
Tensor & _th_masked_fill_bool_(Tensor & self, const Tensor & mask, const Tensor & value);
Tensor & s__th_masked_fill_bool_(Tensor & self, const Tensor & mask, const Tensor & value);
Tensor & _th_masked_scatter_(Tensor & self, const Tensor & mask, const Tensor & source);
Tensor & s__th_masked_scatter_(Tensor & self, const Tensor & mask, const Tensor & source);
Tensor & _th_masked_scatter_bool_(Tensor & self, const Tensor & mask, const Tensor & source);
Tensor & s__th_masked_scatter_bool_(Tensor & self, const Tensor & mask, const Tensor & source);
Tensor & _th_masked_select_out(Tensor & result, const Tensor & self, const Tensor & mask);
Tensor & s__th_masked_select_out(Tensor & result, const Tensor & self, const Tensor & mask);
Tensor _th_masked_select(const Tensor & self, const Tensor & mask);
Tensor s__th_masked_select(const Tensor & self, const Tensor & mask);
Tensor & _th_masked_select_bool_out(Tensor & result, const Tensor & self, const Tensor & mask);
Tensor & s__th_masked_select_bool_out(Tensor & result, const Tensor & self, const Tensor & mask);
Tensor _th_masked_select_bool(const Tensor & self, const Tensor & mask);
Tensor s__th_masked_select_bool(const Tensor & self, const Tensor & mask);
Tensor & _th_nonzero_out(Tensor & result, const Tensor & self);
Tensor _th_nonzero(const Tensor & self);
Tensor _th_clone(const Tensor & self);
Tensor & _th_resize_as_(Tensor & self, const Tensor & the_template);
Tensor & _th_index_select_out(Tensor & result, const Tensor & self, int64_t dim, const Tensor & index);
Tensor _th_index_select(const Tensor & self, int64_t dim, const Tensor & index);
Tensor & _th_index_copy_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source);
Tensor & _th_take_out(Tensor & result, const Tensor & self, const Tensor & index);
Tensor _th_take(const Tensor & self, const Tensor & index);
Tensor & _th_put_(Tensor & self, const Tensor & index, const Tensor & source, bool accumulate);
Tensor & _th_index_add_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source);
Tensor & _th_index_fill_(Tensor & self, int64_t dim, const Tensor & index, Scalar value);
Tensor & _th_index_fill_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & value);
Tensor & _th_unfold_out(Tensor & result, const Tensor & self, int64_t dimension, int64_t size, int64_t step);
Tensor _th_unfold(const Tensor & self, int64_t dimension, int64_t size, int64_t step);
Tensor & _th_scatter_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & src);
Tensor & _th_scatter_(Tensor & self, int64_t dim, const Tensor & index, Scalar value);
Tensor & _th_scatter_add_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & src);
Tensor & _th_gather_out(Tensor & result, const Tensor & self, int64_t dim, const Tensor & index);
Tensor _th_gather(const Tensor & self, int64_t dim, const Tensor & index);
bool _th_equal(const Tensor & self, const Tensor & other);
Tensor & _th_and_out(Tensor & result, const Tensor & self, Scalar other);
Tensor _th_and(const Tensor & self, Scalar other);
Tensor & _th_and_out(Tensor & result, const Tensor & self, const Tensor & other);
Tensor & s__th_and_out(Tensor & result, const Tensor & self, const Tensor & other);
Tensor _th_and(const Tensor & self, const Tensor & other);
Tensor s__th_and(const Tensor & self, const Tensor & other);
Tensor & _th_iand_(Tensor & self, Scalar other);
Tensor & _th_iand_(Tensor & self, const Tensor & other);
Tensor & s__th_iand_(Tensor & self, const Tensor & other);
Tensor & _th_or_out(Tensor & result, const Tensor & self, Scalar other);
Tensor _th_or(const Tensor & self, Scalar other);
Tensor & _th_or_out(Tensor & result, const Tensor & self, const Tensor & other);
Tensor & s__th_or_out(Tensor & result, const Tensor & self, const Tensor & other);
Tensor _th_or(const Tensor & self, const Tensor & other);
Tensor s__th_or(const Tensor & self, const Tensor & other);
Tensor & _th_ior_(Tensor & self, Scalar other);
Tensor & _th_ior_(Tensor & self, const Tensor & other);
Tensor & s__th_ior_(Tensor & self, const Tensor & other);
Tensor & _th_xor_out(Tensor & result, const Tensor & self, Scalar other);
Tensor _th_xor(const Tensor & self, Scalar other);
Tensor & _th_xor_out(Tensor & result, const Tensor & self, const Tensor & other);
Tensor & s__th_xor_out(Tensor & result, const Tensor & self, const Tensor & other);
Tensor _th_xor(const Tensor & self, const Tensor & other);
Tensor s__th_xor(const Tensor & self, const Tensor & other);
Tensor & _th_ixor_(Tensor & self, Scalar other);
Tensor & _th_ixor_(Tensor & self, const Tensor & other);
Tensor & s__th_ixor_(Tensor & self, const Tensor & other);
Tensor & _th_lshift_out(Tensor & result, const Tensor & self, Scalar other);
Tensor _th_lshift(const Tensor & self, Scalar other);
Tensor & _th_lshift_out(Tensor & result, const Tensor & self, const Tensor & other);
Tensor & s__th_lshift_out(Tensor & result, const Tensor & self, const Tensor & other);
Tensor _th_lshift(const Tensor & self, const Tensor & other);
Tensor s__th_lshift(const Tensor & self, const Tensor & other);
Tensor & _th_ilshift_(Tensor & self, Scalar other);
Tensor & _th_ilshift_(Tensor & self, const Tensor & other);
Tensor & s__th_ilshift_(Tensor & self, const Tensor & other);
Tensor & _th_rshift_out(Tensor & result, const Tensor & self, Scalar other);
Tensor _th_rshift(const Tensor & self, Scalar other);
Tensor & _th_rshift_out(Tensor & result, const Tensor & self, const Tensor & other);
Tensor & s__th_rshift_out(Tensor & result, const Tensor & self, const Tensor & other);
Tensor _th_rshift(const Tensor & self, const Tensor & other);
Tensor s__th_rshift(const Tensor & self, const Tensor & other);
Tensor & _th_irshift_(Tensor & self, Scalar other);
Tensor & _th_irshift_(Tensor & self, const Tensor & other);
Tensor & s__th_irshift_(Tensor & self, const Tensor & other);
Tensor & _th_lt_out(Tensor & result, const Tensor & self, Scalar other);
Tensor _th_lt(const Tensor & self, Scalar other);
Tensor & _th_lt_out(Tensor & result, const Tensor & self, const Tensor & other);
Tensor & s__th_lt_out(Tensor & result, const Tensor & self, const Tensor & other);
Tensor _th_lt(const Tensor & self, const Tensor & other);
Tensor s__th_lt(const Tensor & self, const Tensor & other);
Tensor & _th_lt_byte_out(Tensor & result, const Tensor & self, Scalar other);
Tensor _th_lt_byte(const Tensor & self, Scalar other);
Tensor & _th_lt_byte_out(Tensor & result, const Tensor & self, const Tensor & other);
Tensor & s__th_lt_byte_out(Tensor & result, const Tensor & self, const Tensor & other);
Tensor _th_lt_byte(const Tensor & self, const Tensor & other);
Tensor s__th_lt_byte(const Tensor & self, const Tensor & other);
Tensor & _th_lt_(Tensor & self, Scalar other);
Tensor & _th_lt_(Tensor & self, const Tensor & other);
Tensor & s__th_lt_(Tensor & self, const Tensor & other);
Tensor & _th_gt_out(Tensor & result, const Tensor & self, Scalar other);
Tensor _th_gt(const Tensor & self, Scalar other);
Tensor & _th_gt_out(Tensor & result, const Tensor & self, const Tensor & other);
Tensor & s__th_gt_out(Tensor & result, const Tensor & self, const Tensor & other);
Tensor _th_gt(const Tensor & self, const Tensor & other);
Tensor s__th_gt(const Tensor & self, const Tensor & other);
Tensor & _th_gt_byte_out(Tensor & result, const Tensor & self, Scalar other);
Tensor _th_gt_byte(const Tensor & self, Scalar other);
Tensor & _th_gt_byte_out(Tensor & result, const Tensor & self, const Tensor & other);
Tensor & s__th_gt_byte_out(Tensor & result, const Tensor & self, const Tensor & other);
Tensor _th_gt_byte(const Tensor & self, const Tensor & other);
Tensor s__th_gt_byte(const Tensor & self, const Tensor & other);
Tensor & _th_gt_(Tensor & self, Scalar other);
Tensor & _th_gt_(Tensor & self, const Tensor & other);
Tensor & s__th_gt_(Tensor & self, const Tensor & other);
Tensor & _th_le_out(Tensor & result, const Tensor & self, Scalar other);
Tensor _th_le(const Tensor & self, Scalar other);
Tensor & _th_le_out(Tensor & result, const Tensor & self, const Tensor & other);
Tensor & s__th_le_out(Tensor & result, const Tensor & self, const Tensor & other);
Tensor _th_le(const Tensor & self, const Tensor & other);
Tensor s__th_le(const Tensor & self, const Tensor & other);
Tensor & _th_le_byte_out(Tensor & result, const Tensor & self, Scalar other);
Tensor _th_le_byte(const Tensor & self, Scalar other);
Tensor & _th_le_byte_out(Tensor & result, const Tensor & self, const Tensor & other);
Tensor & s__th_le_byte_out(Tensor & result, const Tensor & self, const Tensor & other);
Tensor _th_le_byte(const Tensor & self, const Tensor & other);
Tensor s__th_le_byte(const Tensor & self, const Tensor & other);
Tensor & _th_le_(Tensor & self, Scalar other);
Tensor & _th_le_(Tensor & self, const Tensor & other);
Tensor & s__th_le_(Tensor & self, const Tensor & other);
Tensor & _th_ge_out(Tensor & result, const Tensor & self, Scalar other);
Tensor _th_ge(const Tensor & self, Scalar other);
Tensor & _th_ge_out(Tensor & result, const Tensor & self, const Tensor & other);
Tensor & s__th_ge_out(Tensor & result, const Tensor & self, const Tensor & other);
Tensor _th_ge(const Tensor & self, const Tensor & other);
Tensor s__th_ge(const Tensor & self, const Tensor & other);
Tensor & _th_ge_byte_out(Tensor & result, const Tensor & self, Scalar other);
Tensor _th_ge_byte(const Tensor & self, Scalar other);
Tensor & _th_ge_byte_out(Tensor & result, const Tensor & self, const Tensor & other);
Tensor & s__th_ge_byte_out(Tensor & result, const Tensor & self, const Tensor & other);
Tensor _th_ge_byte(const Tensor & self, const Tensor & other);
Tensor s__th_ge_byte(const Tensor & self, const Tensor & other);
Tensor & _th_ge_(Tensor & self, Scalar other);
Tensor & _th_ge_(Tensor & self, const Tensor & other);
Tensor & s__th_ge_(Tensor & self, const Tensor & other);
Tensor & _th_eq_out(Tensor & result, const Tensor & self, Scalar other);
Tensor _th_eq(const Tensor & self, Scalar other);
Tensor & _th_eq_out(Tensor & result, const Tensor & self, const Tensor & other);
Tensor & s__th_eq_out(Tensor & result, const Tensor & self, const Tensor & other);
Tensor _th_eq(const Tensor & self, const Tensor & other);
Tensor s__th_eq(const Tensor & self, const Tensor & other);
Tensor & _th_eq_byte_out(Tensor & result, const Tensor & self, Scalar other);
Tensor _th_eq_byte(const Tensor & self, Scalar other);
Tensor & _th_eq_byte_out(Tensor & result, const Tensor & self, const Tensor & other);
Tensor & s__th_eq_byte_out(Tensor & result, const Tensor & self, const Tensor & other);
Tensor _th_eq_byte(const Tensor & self, const Tensor & other);
Tensor s__th_eq_byte(const Tensor & self, const Tensor & other);
Tensor & _th_eq_(Tensor & self, Scalar other);
Tensor & _th_eq_(Tensor & self, const Tensor & other);
Tensor & s__th_eq_(Tensor & self, const Tensor & other);
Tensor & _th_ne_out(Tensor & result, const Tensor & self, Scalar other);
Tensor _th_ne(const Tensor & self, Scalar other);
Tensor & _th_ne_out(Tensor & result, const Tensor & self, const Tensor & other);
Tensor & s__th_ne_out(Tensor & result, const Tensor & self, const Tensor & other);
Tensor _th_ne(const Tensor & self, const Tensor & other);
Tensor s__th_ne(const Tensor & self, const Tensor & other);
Tensor & _th_ne_byte_out(Tensor & result, const Tensor & self, Scalar other);
Tensor _th_ne_byte(const Tensor & self, Scalar other);
Tensor & _th_ne_byte_out(Tensor & result, const Tensor & self, const Tensor & other);
Tensor & s__th_ne_byte_out(Tensor & result, const Tensor & self, const Tensor & other);
Tensor _th_ne_byte(const Tensor & self, const Tensor & other);
Tensor s__th_ne_byte(const Tensor & self, const Tensor & other);
Tensor & _th_ne_(Tensor & self, Scalar other);
Tensor & _th_ne_(Tensor & self, const Tensor & other);
Tensor & s__th_ne_(Tensor & self, const Tensor & other);
Tensor & _th_min_out(Tensor & result, const Tensor & self, const Tensor & other);
Tensor & s__th_min_out(Tensor & result, const Tensor & self, const Tensor & other);
Tensor _th_min(const Tensor & self, const Tensor & other);
Tensor s__th_min(const Tensor & self, const Tensor & other);
Tensor _th_min(const Tensor & self);
std::tuple<Tensor &,Tensor &> _th_min_out(Tensor & min, Tensor & min_indices, const Tensor & self, int64_t dim, bool keepdim);
std::tuple<Tensor,Tensor> _th_min(const Tensor & self, int64_t dim, bool keepdim);
Tensor & _th_max_out(Tensor & result, const Tensor & self, const Tensor & other);
Tensor & s__th_max_out(Tensor & result, const Tensor & self, const Tensor & other);
Tensor _th_max(const Tensor & self, const Tensor & other);
Tensor s__th_max(const Tensor & self, const Tensor & other);
Tensor _th_max(const Tensor & self);
std::tuple<Tensor &,Tensor &> _th_max_out(Tensor & max, Tensor & max_indices, const Tensor & self, int64_t dim, bool keepdim);
std::tuple<Tensor,Tensor> _th_max(const Tensor & self, int64_t dim, bool keepdim);
std::tuple<Tensor &,Tensor &> _th_mode_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool keepdim);
std::tuple<Tensor,Tensor> _th_mode(const Tensor & self, int64_t dim, bool keepdim);
std::tuple<Tensor &,Tensor &> _th_sort_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool descending);
std::tuple<Tensor,Tensor> _th_sort(const Tensor & self, int64_t dim, bool descending);
std::tuple<Tensor &,Tensor &> _th_topk_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted);
std::tuple<Tensor,Tensor> _th_topk(const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted);
Tensor & _th_lgamma_out(Tensor & result, const Tensor & self);
Tensor _th_lgamma(const Tensor & self);
Tensor & _th_lgamma_(Tensor & self);
Tensor & _th_digamma_out(Tensor & result, const Tensor & self);
Tensor _th_digamma(const Tensor & self);
Tensor & _th_digamma_(Tensor & self);
Tensor & _th_polygamma_out(Tensor & result, int64_t n, const Tensor & self);
Tensor _th_polygamma(int64_t n, const Tensor & self);
Tensor & _th_polygamma_(Tensor & self, int64_t n);
Tensor & _th_erfinv_(Tensor & self);
Tensor & _th_erfinv_out(Tensor & result, const Tensor & self);
Tensor _th_erfinv(const Tensor & self);
Tensor & _th_var_out(Tensor & result, const Tensor & self, int64_t dim, bool unbiased, bool keepdim);
Tensor _th_var(const Tensor & self, int64_t dim, bool unbiased, bool keepdim);
Tensor _th_var(const Tensor & self, bool unbiased);
Tensor & _th_std_out(Tensor & result, const Tensor & self, int64_t dim, bool unbiased, bool keepdim);
Tensor _th_std(const Tensor & self, int64_t dim, bool unbiased, bool keepdim);
Tensor _th_std(const Tensor & self, bool unbiased);
Tensor & _th_renorm_out(Tensor & result, const Tensor & self, Scalar p, int64_t dim, Scalar maxnorm);
Tensor _th_renorm(const Tensor & self, Scalar p, int64_t dim, Scalar maxnorm);
Tensor & _th_renorm_(Tensor & self, Scalar p, int64_t dim, Scalar maxnorm);
Tensor _th_dist(const Tensor & self, const Tensor & other, Scalar p);
Tensor s__th_dist(const Tensor & self, const Tensor & other, Scalar p);
Tensor & _th_atan2_out(Tensor & result, const Tensor & self, const Tensor & other);
Tensor & s__th_atan2_out(Tensor & result, const Tensor & self, const Tensor & other);
Tensor _th_atan2(const Tensor & self, const Tensor & other);
Tensor s__th_atan2(const Tensor & self, const Tensor & other);
Tensor & _th_atan2_(Tensor & self, const Tensor & other);
Tensor & s__th_atan2_(Tensor & self, const Tensor & other);
Tensor & _th_pow_out(Tensor & result, const Tensor & self, Scalar exponent);
Tensor _th_pow(const Tensor & self, Scalar exponent);
Tensor & _th_pow_out(Tensor & result, const Tensor & self, const Tensor & exponent);
Tensor & s__th_pow_out(Tensor & result, const Tensor & self, const Tensor & exponent);
Tensor _th_pow(const Tensor & self, const Tensor & exponent);
Tensor s__th_pow(const Tensor & self, const Tensor & exponent);
Tensor & _th_pow_out(Tensor & result, Scalar self, const Tensor & exponent);
Tensor _th_pow(Scalar self, const Tensor & exponent);
Tensor & _th_pow_(Tensor & self, Scalar exponent);
Tensor & _th_pow_(Tensor & self, const Tensor & exponent);
Tensor & s__th_pow_(Tensor & self, const Tensor & exponent);
Tensor & _th_histc_out(Tensor & result, const Tensor & self, int64_t bins, Scalar min, Scalar max);
Tensor _th_histc(const Tensor & self, int64_t bins, Scalar min, Scalar max);
Tensor & _th_zero_(Tensor & self);
Tensor & _th_cumsum_out(Tensor & result, const Tensor & self, int64_t dim);
Tensor _th_cumsum(const Tensor & self, int64_t dim);
Tensor & _th_cumprod_out(Tensor & result, const Tensor & self, int64_t dim);
Tensor _th_cumprod(const Tensor & self, int64_t dim);
Tensor & _th_sign_out(Tensor & result, const Tensor & self);
Tensor _th_sign(const Tensor & self);
Tensor & _th_sign_(Tensor & self);
Tensor _th_trace(const Tensor & self);
Tensor & _th_fmod_out(Tensor & result, const Tensor & self, Scalar other);
Tensor _th_fmod(const Tensor & self, Scalar other);
Tensor & _th_fmod_out(Tensor & result, const Tensor & self, const Tensor & other);
Tensor & s__th_fmod_out(Tensor & result, const Tensor & self, const Tensor & other);
Tensor _th_fmod(const Tensor & self, const Tensor & other);
Tensor s__th_fmod(const Tensor & self, const Tensor & other);
Tensor & _th_fmod_(Tensor & self, Scalar other);
Tensor & _th_fmod_(Tensor & self, const Tensor & other);
Tensor & s__th_fmod_(Tensor & self, const Tensor & other);
Tensor & _th_remainder_out(Tensor & result, const Tensor & self, Scalar other);
Tensor _th_remainder(const Tensor & self, Scalar other);
Tensor & _th_remainder_out(Tensor & result, const Tensor & self, const Tensor & other);
Tensor & s__th_remainder_out(Tensor & result, const Tensor & self, const Tensor & other);
Tensor _th_remainder(const Tensor & self, const Tensor & other);
Tensor s__th_remainder(const Tensor & self, const Tensor & other);
Tensor & _th_remainder_(Tensor & self, Scalar other);
Tensor & _th_remainder_(Tensor & self, const Tensor & other);
Tensor & s__th_remainder_(Tensor & self, const Tensor & other);
Tensor & _th_clamp_out(Tensor & result, const Tensor & self, Scalar min, Scalar max);
Tensor _th_clamp(const Tensor & self, Scalar min, Scalar max);
Tensor & _th_clamp_min_out(Tensor & result, const Tensor & self, Scalar min);
Tensor _th_clamp_min(const Tensor & self, Scalar min);
Tensor & _th_clamp_max_out(Tensor & result, const Tensor & self, Scalar max);
Tensor _th_clamp_max(const Tensor & self, Scalar max);
Tensor _th_dot(const Tensor & self, const Tensor & tensor);
Tensor & _th_diag_out(Tensor & result, const Tensor & self, int64_t diagonal);
Tensor _th_diag(const Tensor & self, int64_t diagonal);
Tensor & _th_addmm_out(Tensor & result, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha);
Tensor & s__th_addmm_out(Tensor & result, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha);
Tensor _th_addmm(const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha);
Tensor s__th_addmm(const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha);
Tensor & _th_addmm_(Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha);
Tensor & _th_addmv_out(Tensor & result, const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha);
Tensor & s__th_addmv_out(Tensor & result, const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha);
Tensor _th_addmv(const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha);
Tensor s__th_addmv(const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha);
Tensor & _th_addmv_(Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha);
Tensor & _th_addr_out(Tensor & result, const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha);
Tensor & s__th_addr_out(Tensor & result, const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha);
Tensor _th_addr(const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha);
Tensor s__th_addr(const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha);
Tensor & _th_addr_(Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha);
Tensor & _th_ger_out(Tensor & result, const Tensor & self, const Tensor & vec2);
Tensor _th_ger(const Tensor & self, const Tensor & vec2);
Tensor & _th_mv_out(Tensor & result, const Tensor & self, const Tensor & vec);
Tensor _th_mv(const Tensor & self, const Tensor & vec);
Tensor & _th_mm_out(Tensor & result, const Tensor & self, const Tensor & mat2);
Tensor _th_mm(const Tensor & self, const Tensor & mat2);
Tensor & _th_addbmm_out(Tensor & result, const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha);
Tensor & s__th_addbmm_out(Tensor & result, const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha);
Tensor _th_addbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha);
Tensor s__th_addbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha);
Tensor & _th_addbmm_(Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha);
Tensor & _th_addcmul_out(Tensor & result, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value);
Tensor & s__th_addcmul_out(Tensor & result, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value);
Tensor _th_addcmul(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value);
Tensor s__th_addcmul(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value);
Tensor & _th_addcmul_(Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value);
Tensor & s__th_addcmul_(Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value);
Tensor & _th_addcdiv_out(Tensor & result, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value);
Tensor & s__th_addcdiv_out(Tensor & result, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value);
Tensor _th_addcdiv(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value);
Tensor s__th_addcdiv(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value);
Tensor & _th_addcdiv_(Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value);
Tensor & s__th_addcdiv_(Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value);
std::tuple<Tensor &,Tensor &> _th_gels_out(Tensor & res1, Tensor & res2, const Tensor & self, const Tensor & A);
std::tuple<Tensor,Tensor> _th_gels(const Tensor & self, const Tensor & A);
std::tuple<Tensor &,Tensor &> _th_eig_out(Tensor & res1, Tensor & res2, const Tensor & self, bool eigenvectors);
std::tuple<Tensor,Tensor> _th_eig(const Tensor & self, bool eigenvectors);
Tensor & _th_potri_out(Tensor & output, const Tensor & self, bool upper);
Tensor _th_potri(const Tensor & self, bool upper);
std::tuple<Tensor &,Tensor &> _th_geqrf_out(Tensor & res1, Tensor & res2, const Tensor & self);
std::tuple<Tensor,Tensor> _th_geqrf(const Tensor & self);
Tensor & _th_orgqr_out(Tensor & result, const Tensor & self, const Tensor & input2);
Tensor _th_orgqr(const Tensor & self, const Tensor & input2);
Tensor & _th_ormqr_out(Tensor & result, const Tensor & self, const Tensor & input2, const Tensor & input3, bool left, bool transpose);
Tensor _th_ormqr(const Tensor & self, const Tensor & input2, const Tensor & input3, bool left, bool transpose);
Tensor & _th_random_(Tensor & self, int64_t from, int64_t to, Generator * generator);
Tensor & _th_random_(Tensor & self, int64_t to, Generator * generator);
Tensor & _th_random_(Tensor & self, Generator * generator);
std::tuple<Tensor &,Tensor &> _th_multinomial_alias_setup_out(Tensor & J, Tensor & q, const Tensor & probs);
std::tuple<Tensor,Tensor> _th_multinomial_alias_setup(const Tensor & probs);
Tensor & _th_multinomial_alias_draw_out(Tensor & result, const Tensor & q, const Tensor & J, int64_t num_samples, Generator * generator);
Tensor _th_multinomial_alias_draw(const Tensor & q, const Tensor & J, int64_t num_samples, Generator * generator);
Tensor & _th_multinomial_out(Tensor & result, const Tensor & self, int64_t num_samples, bool replacement, Generator * generator);
Tensor _th_multinomial(const Tensor & self, int64_t num_samples, bool replacement, Generator * generator);
Tensor & _th_uniform_(Tensor & self, double from, double to, Generator * generator);
Tensor & _th_normal_out(Tensor & output, const Tensor & mean, double std, Generator * generator);
Tensor _th_normal(const Tensor & mean, double std, Generator * generator);
Tensor & _th_normal_out(Tensor & output, double mean, const Tensor & std, Generator * generator);
Tensor _th_normal(double mean, const Tensor & std, Generator * generator);
Tensor & _th_normal_out(Tensor & output, const Tensor & mean, const Tensor & std, Generator * generator);
Tensor _th_normal(const Tensor & mean, const Tensor & std, Generator * generator);
Tensor & _th_normal_(Tensor & self, double mean, double std, Generator * generator);
Tensor & _th_cauchy_(Tensor & self, double median, double sigma, Generator * generator);
Tensor & _th_log_normal_(Tensor & self, double mean, double std, Generator * generator);
Tensor & _th_exponential_(Tensor & self, double lambd, Generator * generator);
Tensor & _th_geometric_(Tensor & self, double p, Generator * generator);
Tensor & _th_cat_out(Tensor & self, TensorList tensors, int64_t dim);
Tensor _th_cat(TensorList tensors, int64_t dim);
Tensor & _thnn_binary_cross_entropy_forward_out(Tensor & output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction);
Tensor _thnn_binary_cross_entropy_forward(const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction);
Tensor & _thnn_binary_cross_entropy_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction);
Tensor _thnn_binary_cross_entropy_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction);
Tensor & _thnn_l1_loss_forward_out(Tensor & output, const Tensor & self, const Tensor & target, int64_t reduction);
Tensor _thnn_l1_loss_forward(const Tensor & self, const Tensor & target, int64_t reduction);
Tensor & _thnn_l1_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction);
Tensor _thnn_l1_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction);
Tensor & _thnn_mse_loss_forward_out(Tensor & output, const Tensor & self, const Tensor & target, int64_t reduction);
Tensor _thnn_mse_loss_forward(const Tensor & self, const Tensor & target, int64_t reduction);
Tensor & _thnn_mse_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction);
Tensor _thnn_mse_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction);
Tensor & _thnn_multi_margin_loss_forward_out(Tensor & output, const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, int64_t reduction);
Tensor _thnn_multi_margin_loss_forward(const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, int64_t reduction);
Tensor & _thnn_multi_margin_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, int64_t reduction);
Tensor _thnn_multi_margin_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, int64_t reduction);
std::tuple<Tensor &,Tensor &> _thnn_multilabel_margin_loss_forward_out(Tensor & output, Tensor & is_target, const Tensor & self, const Tensor & target, int64_t reduction);
std::tuple<Tensor,Tensor> _thnn_multilabel_margin_loss_forward(const Tensor & self, const Tensor & target, int64_t reduction);
Tensor & _thnn_multilabel_margin_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction, const Tensor & is_target);
Tensor _thnn_multilabel_margin_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction, const Tensor & is_target);
std::tuple<Tensor &,Tensor &> _thnn_nll_loss_forward_out(Tensor & output, Tensor & total_weight, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index);
std::tuple<Tensor,Tensor> _thnn_nll_loss_forward(const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index);
Tensor & _thnn_nll_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index, const Tensor & total_weight);
Tensor _thnn_nll_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index, const Tensor & total_weight);
std::tuple<Tensor &,Tensor &> _thnn_nll_loss2d_forward_out(Tensor & output, Tensor & total_weight, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index);
std::tuple<Tensor,Tensor> _thnn_nll_loss2d_forward(const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index);
Tensor & _thnn_nll_loss2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index, const Tensor & total_weight);
Tensor _thnn_nll_loss2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index, const Tensor & total_weight);
Tensor & _thnn_smooth_l1_loss_forward_out(Tensor & output, const Tensor & self, const Tensor & target, int64_t reduction);
Tensor _thnn_smooth_l1_loss_forward(const Tensor & self, const Tensor & target, int64_t reduction);
Tensor & _thnn_smooth_l1_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction);
Tensor _thnn_smooth_l1_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction);
Tensor & _thnn_soft_margin_loss_forward_out(Tensor & output, const Tensor & self, const Tensor & target, int64_t reduction);
Tensor _thnn_soft_margin_loss_forward(const Tensor & self, const Tensor & target, int64_t reduction);
Tensor & _thnn_soft_margin_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction);
Tensor _thnn_soft_margin_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction);
Tensor & _thnn_elu_forward_out(Tensor & output, const Tensor & self, Scalar alpha, Scalar scale, Scalar input_scale);
Tensor _thnn_elu_forward(const Tensor & self, Scalar alpha, Scalar scale, Scalar input_scale);
Tensor & _thnn_elu_backward_out(Tensor & grad_input, const Tensor & grad_output, Scalar alpha, Scalar scale, Scalar input_scale, const Tensor & output);
Tensor _thnn_elu_backward(const Tensor & grad_output, Scalar alpha, Scalar scale, Scalar input_scale, const Tensor & output);
Tensor & _thnn_elu_forward_(Tensor & self, Scalar alpha, Scalar scale, Scalar input_scale);
Tensor & _thnn_glu_forward_out(Tensor & output, const Tensor & self, int64_t dim);
Tensor _thnn_glu_forward(const Tensor & self, int64_t dim);
Tensor & _thnn_glu_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, int64_t dim);
Tensor _thnn_glu_backward(const Tensor & grad_output, const Tensor & self, int64_t dim);
Tensor & _thnn_hardtanh_forward_out(Tensor & output, const Tensor & self, Scalar min_val, Scalar max_val);
Tensor _thnn_hardtanh_forward(const Tensor & self, Scalar min_val, Scalar max_val);
Tensor & _thnn_hardtanh_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar min_val, Scalar max_val);
Tensor _thnn_hardtanh_backward(const Tensor & grad_output, const Tensor & self, Scalar min_val, Scalar max_val);
Tensor & _thnn_hardtanh_forward_(Tensor & self, Scalar min_val, Scalar max_val);
Tensor & _thnn_leaky_relu_forward_out(Tensor & output, const Tensor & self, Scalar negative_slope);
Tensor _thnn_leaky_relu_forward(const Tensor & self, Scalar negative_slope);
Tensor & _thnn_leaky_relu_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar negative_slope);
Tensor _thnn_leaky_relu_backward(const Tensor & grad_output, const Tensor & self, Scalar negative_slope);
Tensor & _thnn_leaky_relu_forward_(Tensor & self, Scalar negative_slope);
std::tuple<Tensor &,Tensor &> _thnn_log_sigmoid_forward_out(Tensor & output, Tensor & buffer, const Tensor & self);
std::tuple<Tensor,Tensor> _thnn_log_sigmoid_forward(const Tensor & self);
Tensor & _thnn_log_sigmoid_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & buffer);
Tensor _thnn_log_sigmoid_backward(const Tensor & grad_output, const Tensor & self, const Tensor & buffer);
Tensor & _thnn_rrelu_with_noise_forward_out(Tensor & output, const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator * generator);
Tensor _thnn_rrelu_with_noise_forward(const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator * generator);
Tensor & _thnn_rrelu_with_noise_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training);
Tensor _thnn_rrelu_with_noise_backward(const Tensor & grad_output, const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training);
Tensor & _thnn_rrelu_with_noise_forward_(Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator * generator);
Tensor & _thnn_softplus_forward_out(Tensor & output, const Tensor & self, Scalar beta, Scalar threshold);
Tensor _thnn_softplus_forward(const Tensor & self, Scalar beta, Scalar threshold);
Tensor & _thnn_softplus_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar beta, Scalar threshold, const Tensor & output);
Tensor _thnn_softplus_backward(const Tensor & grad_output, const Tensor & self, Scalar beta, Scalar threshold, const Tensor & output);
Tensor & _thnn_softshrink_forward_out(Tensor & output, const Tensor & self, Scalar lambd);
Tensor _thnn_softshrink_forward(const Tensor & self, Scalar lambd);
Tensor & _thnn_softshrink_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar lambd);
Tensor _thnn_softshrink_backward(const Tensor & grad_output, const Tensor & self, Scalar lambd);
Tensor & _thnn_sigmoid_forward_out(Tensor & output, const Tensor & self);
Tensor _thnn_sigmoid_forward(const Tensor & self);
Tensor & _thnn_sigmoid_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & output);
Tensor _thnn_sigmoid_backward(const Tensor & grad_output, const Tensor & output);
Tensor & _thnn_tanh_forward_out(Tensor & output, const Tensor & self);
Tensor _thnn_tanh_forward(const Tensor & self);
Tensor & _thnn_tanh_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & output);
Tensor _thnn_tanh_backward(const Tensor & grad_output, const Tensor & output);
std::tuple<Tensor &,Tensor &,Tensor &> _thnn_conv2d_forward_out(Tensor & output, Tensor & finput, Tensor & fgrad_input, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding);
std::tuple<Tensor,Tensor,Tensor> _thnn_conv2d_forward(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding);
std::tuple<Tensor &,Tensor &,Tensor &> _thnn_conv2d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, const Tensor & finput, const Tensor & fgrad_input);
std::tuple<Tensor,Tensor,Tensor> _thnn_conv2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, const Tensor & finput, const Tensor & fgrad_input, std::array<bool,3> output_mask);
std::tuple<Tensor &,Tensor &,Tensor &> _thnn_conv3d_forward_out(Tensor & output, Tensor & finput, Tensor & fgrad_input, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding);
std::tuple<Tensor,Tensor,Tensor> _thnn_conv3d_forward(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding);
std::tuple<Tensor &,Tensor &,Tensor &> _thnn_conv3d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, const Tensor & finput, const Tensor & fgrad_input);
std::tuple<Tensor,Tensor,Tensor> _thnn_conv3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, const Tensor & finput, const Tensor & fgrad_input, std::array<bool,3> output_mask);

} // namespace th
} // namespace legacy
} // namespace native
} // namespace at
