// This file contains all native_functions that can be registered to
// and the schema string that they should be registered with

Tensor _cast_Byte(const Tensor & self, bool non_blocking); // aten::_cast_Byte(Tensor self, bool non_blocking=False) -> Tensor
Tensor _cast_Char(const Tensor & self, bool non_blocking); // aten::_cast_Char(Tensor self, bool non_blocking=False) -> Tensor
Tensor _cast_Double(const Tensor & self, bool non_blocking); // aten::_cast_Double(Tensor self, bool non_blocking=False) -> Tensor
Tensor _cast_Float(const Tensor & self, bool non_blocking); // aten::_cast_Float(Tensor self, bool non_blocking=False) -> Tensor
Tensor _cast_Int(const Tensor & self, bool non_blocking); // aten::_cast_Int(Tensor self, bool non_blocking=False) -> Tensor
Tensor _cast_Long(const Tensor & self, bool non_blocking); // aten::_cast_Long(Tensor self, bool non_blocking=False) -> Tensor
Tensor _cast_Short(const Tensor & self, bool non_blocking); // aten::_cast_Short(Tensor self, bool non_blocking=False) -> Tensor
Tensor _cast_Half(const Tensor & self, bool non_blocking); // aten::_cast_Half(Tensor self, bool non_blocking=False) -> Tensor
void backward(const Tensor & self, const Tensor & gradient, bool keep_graph, bool create_graph); // aten::backward(Tensor self, Tensor? gradient=None, bool keep_graph=False, bool create_graph=False) -> void
void set_data(const Tensor & self, const Tensor & new_data); // aten::set_data(Tensor(a!) self, Tensor new_data) -> void
std::tuple<Tensor,Tensor> _cudnn_ctc_loss(const Tensor & log_probs, const Tensor & targets, IntArrayRef input_lengths, IntArrayRef target_lengths, int64_t blank, bool deterministic, bool zero_infinity); // aten::_cudnn_ctc_loss(Tensor log_probs, Tensor targets, int[] input_lengths, int[] target_lengths, int blank, bool deterministic, bool zero_infinity) -> (Tensor, Tensor)
Tensor _cudnn_rnn_flatten_weight(TensorList weight_arr, int64_t weight_stride0, int64_t input_size, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, bool bidirectional); // aten::_cudnn_rnn_flatten_weight(Tensor[] weight_arr, int weight_stride0, int input_size, int mode, int hidden_size, int num_layers, bool batch_first, bool bidirectional) -> Tensor
std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor> _cudnn_rnn(const Tensor & input, TensorList weight, int64_t weight_stride0, const Tensor & weight_buf, const Tensor & hx, const Tensor & cx, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, IntArrayRef batch_sizes, const Tensor & dropout_state); // aten::_cudnn_rnn(Tensor input, Tensor[] weight, int weight_stride0, Tensor? weight_buf, Tensor hx, Tensor? cx, int mode, int hidden_size, int num_layers, bool batch_first, float dropout, bool train, bool bidirectional, int[] batch_sizes, Tensor? dropout_state) -> (Tensor, Tensor, Tensor, Tensor, Tensor)
std::tuple<Tensor,Tensor,Tensor,std::vector<Tensor>> _cudnn_rnn_backward(const Tensor & input, TensorList weight, int64_t weight_stride0, const Tensor & weight_buf, const Tensor & hx, const Tensor & cx, const Tensor & output, const Tensor & grad_output, const Tensor & grad_hy, const Tensor & grad_cy, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, IntArrayRef batch_sizes, const Tensor & dropout_state, const Tensor & reserve, std::array<bool,4> output_mask); // aten::_cudnn_rnn_backward(Tensor input, Tensor[] weight, int weight_stride0, Tensor weight_buf, Tensor hx, Tensor? cx, Tensor output, Tensor? grad_output, Tensor? grad_hy, Tensor? grad_cy, int mode, int hidden_size, int num_layers, bool batch_first, float dropout, bool train, bool bidirectional, int[] batch_sizes, Tensor? dropout_state, Tensor reserve, bool[4] output_mask) -> (Tensor, Tensor, Tensor, Tensor[])
Tensor _cudnn_init_dropout_state(double dropout, bool train, int64_t dropout_seed, const TensorOptions & options); // aten::_cudnn_init_dropout_state(float dropout, bool train, int dropout_seed, *, ScalarType dtype, Layout layout, Device device, bool pin_memory=False) -> Tensor
int64_t _debug_has_internal_overlap(const Tensor & self); // aten::_debug_has_internal_overlap(Tensor self) -> int
std::tuple<Tensor,Tensor> _fused_dropout(const Tensor & self, double p, Generator * generator); // aten::_fused_dropout(Tensor self, float p, Generator? generator=None) -> (Tensor, Tensor)
Tensor _masked_scale(const Tensor & self, const Tensor & mask, double scale); // aten::_masked_scale(Tensor self, Tensor mask, float scale) -> Tensor
std::tuple<Tensor,Tensor> _sobol_engine_draw(const Tensor & quasi, int64_t n, const Tensor & sobolstate, int64_t dimension, int64_t num_generated, c10::optional<ScalarType> dtype); // aten::_sobol_engine_draw(Tensor quasi, int n, Tensor sobolstate, int dimension, int num_generated, ScalarType? dtype) -> (Tensor, Tensor)
Tensor & _sobol_engine_ff_(Tensor & self, int64_t n, const Tensor & sobolstate, int64_t dimension, int64_t num_generated); // aten::_sobol_engine_ff_(Tensor(a!) self, int n, Tensor sobolstate, int dimension, int num_generated) -> Tensor(a!)
Tensor & _sobol_engine_scramble_(Tensor & self, const Tensor & ltm, int64_t dimension); // aten::_sobol_engine_scramble_(Tensor(a!) self, Tensor ltm, int dimension) -> Tensor(a!)
Tensor & _sobol_engine_initialize_state_(Tensor & self, int64_t dimension); // aten::_sobol_engine_initialize_state_(Tensor(a!) self, int dimension) -> Tensor(a!)
Tensor _reshape_from_tensor(const Tensor & self, const Tensor & shape); // aten::_reshape_from_tensor(Tensor self, Tensor shape) -> Tensor
Tensor _shape_as_tensor(const Tensor & self); // aten::_shape_as_tensor(Tensor self) -> Tensor
Tensor dropout(const Tensor & input, double p, bool train); // aten::dropout(Tensor input, float p, bool train) -> Tensor
Tensor & dropout_(Tensor & self, double p, bool train); // aten::dropout_(Tensor(a!) self, float p, bool train) -> Tensor(a!)
Tensor feature_dropout(const Tensor & input, double p, bool train); // aten::feature_dropout(Tensor input, float p, bool train) -> Tensor
Tensor & feature_dropout_(Tensor & self, double p, bool train); // aten::feature_dropout_(Tensor(a!) self, float p, bool train) -> Tensor(a!)
Tensor alpha_dropout(const Tensor & input, double p, bool train); // aten::alpha_dropout(Tensor input, float p, bool train) -> Tensor
Tensor & alpha_dropout_(Tensor & self, double p, bool train); // aten::alpha_dropout_(Tensor(a!) self, float p, bool train) -> Tensor(a!)
Tensor feature_alpha_dropout(const Tensor & input, double p, bool train); // aten::feature_alpha_dropout(Tensor input, float p, bool train) -> Tensor
Tensor & feature_alpha_dropout_(Tensor & self, double p, bool train); // aten::feature_alpha_dropout_(Tensor(a!) self, float p, bool train) -> Tensor(a!)
Tensor abs(const Tensor & self); // aten::abs(Tensor self) -> Tensor
Tensor & abs_(Tensor & self); // aten::abs_(Tensor(a!) self) -> Tensor(a!)
Tensor & abs_out(Tensor & out, const Tensor & self); // aten::abs(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
Tensor acos(const Tensor & self); // aten::acos(Tensor self) -> Tensor
Tensor & acos_(Tensor & self); // aten::acos_(Tensor(a!) self) -> Tensor(a!)
Tensor & acos_out(Tensor & out, const Tensor & self); // aten::acos(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
Tensor avg_pool1d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad); // aten::avg_pool1d(Tensor self, int[1] kernel_size, int[1] stride=[], int[1] padding=0, bool ceil_mode=False, bool count_include_pad=True) -> Tensor
Tensor adaptive_avg_pool1d(const Tensor & self, IntArrayRef output_size); // aten::adaptive_avg_pool1d(Tensor self, int[1] output_size) -> Tensor
std::tuple<Tensor,Tensor> adaptive_max_pool1d(const Tensor & self, IntArrayRef output_size); // aten::adaptive_max_pool1d(Tensor self, int[1] output_size) -> (Tensor, Tensor)
Tensor add(const Tensor & self, const Tensor & other, Scalar alpha); // aten::add(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
Tensor & add_(Tensor & self, const Tensor & other, Scalar alpha); // aten::add_(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)
Tensor & add_out(Tensor & out, const Tensor & self, const Tensor & other, Scalar alpha); // aten::add(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
Tensor add(const Tensor & self, Scalar other, Scalar alpha); // aten::add(Tensor self, Scalar other, Scalar alpha=1) -> Tensor
Tensor & add_(Tensor & self, Scalar other, Scalar alpha); // aten::add_(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)
Tensor addmv(const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha); // aten::addmv(Tensor self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1) -> Tensor
Tensor & addmv_(Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha); // aten::addmv_(Tensor(a!) self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)
Tensor & addmv_out(Tensor & out, const Tensor & self, const Tensor & mat, const Tensor & vec, Scalar beta, Scalar alpha); // aten::addmv(Tensor self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
Tensor addr(const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha); // aten::addr(Tensor self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
Tensor & addr_(Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha); // aten::addr_(Tensor(a!) self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)
Tensor & addr_out(Tensor & out, const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha); // aten::addr(Tensor self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
Tensor affine_grid_generator(const Tensor & theta, IntArrayRef size); // aten::affine_grid_generator(Tensor theta, int[] size) -> Tensor
Tensor affine_grid_generator_backward(const Tensor & grad, IntArrayRef size); // aten::affine_grid_generator_backward(Tensor grad, int[] size) -> Tensor
Tensor all(const Tensor & self, int64_t dim, bool keepdim); // aten::all(Tensor self, int dim, bool keepdim=False) -> Tensor
Tensor & all_out(Tensor & out, const Tensor & self, int64_t dim, bool keepdim); // aten::all(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
bool allclose(const Tensor & self, const Tensor & other, double rtol, double atol, bool equal_nan); // aten::allclose(Tensor self, Tensor other, float rtol=1e-05, float atol=1e-08, bool equal_nan=False) -> bool
Tensor any(const Tensor & self, int64_t dim, bool keepdim); // aten::any(Tensor self, int dim, bool keepdim=False) -> Tensor
Tensor & any_out(Tensor & out, const Tensor & self, int64_t dim, bool keepdim); // aten::any(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
Tensor arange(Scalar end, const TensorOptions & options); // aten::arange(Scalar end, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
Tensor arange(Scalar start, Scalar end, const TensorOptions & options); // aten::arange(Scalar start, Scalar end, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
Tensor arange(Scalar start, Scalar end, Scalar step, const TensorOptions & options); // aten::arange(Scalar start, Scalar end, Scalar step, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
Tensor & arange_out(Tensor & out, Scalar end); // aten::arange(Scalar end, *, Tensor(a!) out) -> Tensor(a!)
Tensor & arange_out(Tensor & out, Scalar start, Scalar end, Scalar step); // aten::arange(Scalar start, Scalar end, Scalar step=1, *, Tensor(a!) out) -> Tensor(a!)
Tensor _dim_arange(const Tensor & like, int64_t dim); // aten::_dim_arange(Tensor like, int dim) -> Tensor
Tensor argmax(const Tensor & self, c10::optional<int64_t> dim, bool keepdim); // aten::argmax(Tensor self, int? dim=None, bool keepdim=False) -> Tensor
Tensor argmin(const Tensor & self, c10::optional<int64_t> dim, bool keepdim); // aten::argmin(Tensor self, int? dim=None, bool keepdim=False) -> Tensor
Tensor as_strided(const Tensor & self, IntArrayRef size, IntArrayRef stride, c10::optional<int64_t> storage_offset); // aten::as_strided(Tensor(a) self, int[] size, int[] stride, int? storage_offset=None) -> Tensor(a)
Tensor & as_strided_(Tensor & self, IntArrayRef size, IntArrayRef stride, c10::optional<int64_t> storage_offset); // aten::as_strided_(Tensor(a!) self, int[] size, int[] stride, int? storage_offset=None) -> Tensor(a!)
Tensor asin(const Tensor & self); // aten::asin(Tensor self) -> Tensor
Tensor & asin_(Tensor & self); // aten::asin_(Tensor(a!) self) -> Tensor(a!)
Tensor & asin_out(Tensor & out, const Tensor & self); // aten::asin(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
Tensor atan(const Tensor & self); // aten::atan(Tensor self) -> Tensor
Tensor & atan_(Tensor & self); // aten::atan_(Tensor(a!) self) -> Tensor(a!)
Tensor & atan_out(Tensor & out, const Tensor & self); // aten::atan(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
Tensor baddbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha); // aten::baddbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
Tensor & baddbmm_(Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha); // aten::baddbmm_(Tensor(a!) self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)
Tensor & _baddbmm_mkl_(Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha); // aten::_baddbmm_mkl_(Tensor(a!) self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)
Tensor & baddbmm_out(Tensor & out, const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha); // aten::baddbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
Tensor bartlett_window(int64_t window_length, const TensorOptions & options); // aten::bartlett_window(int window_length, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
Tensor bartlett_window(int64_t window_length, bool periodic, const TensorOptions & options); // aten::bartlett_window(int window_length, bool periodic, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
Tensor batch_norm(const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps, bool cudnn_enabled); // aten::batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, bool cudnn_enabled) -> Tensor
std::tuple<Tensor,Tensor,Tensor,int64_t> _batch_norm_impl_index(const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps, bool cudnn_enabled); // aten::_batch_norm_impl_index(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, bool cudnn_enabled) -> (Tensor, Tensor, Tensor, int)
std::tuple<Tensor,Tensor,Tensor> _batch_norm_impl_index_backward(int64_t impl_index, const Tensor & input, const Tensor & grad_output, const Tensor & weight, const Tensor & running_mean, const Tensor & running_var, const Tensor & save_mean, const Tensor & save_var_transform, bool train, double eps, std::array<bool,3> output_mask); // aten::_batch_norm_impl_index_backward(int impl_index, Tensor input, Tensor grad_output, Tensor? weight, Tensor? running_mean, Tensor? running_var, Tensor? save_mean, Tensor? save_var_transform, bool train, float eps, bool[3] output_mask) -> (Tensor, Tensor, Tensor)
Tensor bernoulli(const Tensor & self, Generator * generator); // aten::bernoulli(Tensor self, *, Generator? generator=None) -> Tensor
Tensor & bernoulli_out(Tensor & out, const Tensor & self, Generator * generator); // aten::bernoulli(Tensor self, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)
Tensor & bernoulli_(Tensor & self, const Tensor & p, Generator * generator); // aten::bernoulli_(Tensor(a!) self, Tensor p, *, Generator? generator=None) -> Tensor(a!)
Tensor & bernoulli_(Tensor & self, double p, Generator * generator); // aten::bernoulli_(Tensor(a!) self, float p=0.5, *, Generator? generator=None) -> Tensor(a!)
Tensor bernoulli(const Tensor & self, double p, Generator * generator); // aten::bernoulli(Tensor self, float p, *, Generator? generator=None) -> Tensor
Tensor bilinear(const Tensor & input1, const Tensor & input2, const Tensor & weight, const Tensor & bias); // aten::bilinear(Tensor input1, Tensor input2, Tensor weight, Tensor? bias) -> Tensor
Tensor binary_cross_entropy_with_logits(const Tensor & self, const Tensor & target, const Tensor & weight, const Tensor & pos_weight, int64_t reduction); // aten::binary_cross_entropy_with_logits(Tensor self, Tensor target, Tensor? weight=None, Tensor? pos_weight=None, int reduction=Mean) -> Tensor
Tensor binary_cross_entropy_with_logits_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, const Tensor & pos_weight, int64_t reduction); // aten::binary_cross_entropy_with_logits_backward(Tensor grad_output, Tensor self, Tensor target, Tensor? weight=None, Tensor? pos_weight=None, int reduction=Mean) -> Tensor
Tensor bincount(const Tensor & self, const Tensor & weights, int64_t minlength); // aten::bincount(Tensor self, Tensor? weights=None, int minlength=0) -> Tensor
Tensor bitwise_not(const Tensor & self); // aten::bitwise_not(Tensor self) -> Tensor
Tensor & bitwise_not_(Tensor & self); // aten::bitwise_not_(Tensor(a!) self) -> Tensor(a!)
Tensor & bitwise_not_out(Tensor & out, const Tensor & self); // aten::bitwise_not(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
Tensor blackman_window(int64_t window_length, const TensorOptions & options); // aten::blackman_window(int window_length, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
Tensor blackman_window(int64_t window_length, bool periodic, const TensorOptions & options); // aten::blackman_window(int window_length, bool periodic, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
Tensor bmm(const Tensor & self, const Tensor & mat2); // aten::bmm(Tensor self, Tensor mat2) -> Tensor
Tensor & bmm_out(Tensor & out, const Tensor & self, const Tensor & mat2); // aten::bmm(Tensor self, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)
std::vector<Tensor> broadcast_tensors(TensorList tensors); // aten::broadcast_tensors(Tensor[] tensors) -> Tensor[]
Tensor cat(TensorList tensors, int64_t dim); // aten::cat(Tensor[] tensors, int dim=0) -> Tensor
Tensor & cat_out(Tensor & out, TensorList tensors, int64_t dim); // aten::cat(Tensor[] tensors, int dim=0, *, Tensor(a!) out) -> Tensor(a!)
Tensor ceil(const Tensor & self); // aten::ceil(Tensor self) -> Tensor
Tensor & ceil_(Tensor & self); // aten::ceil_(Tensor(a!) self) -> Tensor(a!)
Tensor & ceil_out(Tensor & out, const Tensor & self); // aten::ceil(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
Tensor chain_matmul(TensorList matrices); // aten::chain_matmul(Tensor[] matrices) -> Tensor
std::vector<Tensor> chunk(const Tensor & self, int64_t chunks, int64_t dim); // aten::chunk(Tensor(a) self, int chunks, int dim=0) -> Tensor(a)[]
Tensor clamp(const Tensor & self, c10::optional<Scalar> min, c10::optional<Scalar> max); // aten::clamp(Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor
Tensor & clamp_(Tensor & self, c10::optional<Scalar> min, c10::optional<Scalar> max); // aten::clamp_(Tensor(a!) self, Scalar? min=None, Scalar? max=None) -> Tensor(a!)
Tensor & clamp_out(Tensor & out, const Tensor & self, c10::optional<Scalar> min, c10::optional<Scalar> max); // aten::clamp(Tensor self, Scalar? min=None, Scalar? max=None, *, Tensor(a!) out) -> Tensor(a!)
Tensor clamp_max(const Tensor & self, Scalar max); // aten::clamp_max(Tensor self, Scalar max) -> Tensor
Tensor & clamp_max_(Tensor & self, Scalar max); // aten::clamp_max_(Tensor(a!) self, Scalar max) -> Tensor(a!)
Tensor & clamp_max_out(Tensor & out, const Tensor & self, Scalar max); // aten::clamp_max(Tensor self, Scalar max, *, Tensor(a!) out) -> Tensor(a!)
Tensor clamp_min(const Tensor & self, Scalar min); // aten::clamp_min(Tensor self, Scalar min) -> Tensor
Tensor & clamp_min_(Tensor & self, Scalar min); // aten::clamp_min_(Tensor(a!) self, Scalar min) -> Tensor(a!)
Tensor & clamp_min_out(Tensor & out, const Tensor & self, Scalar min); // aten::clamp_min(Tensor self, Scalar min, *, Tensor(a!) out) -> Tensor(a!)
bool cudnn_is_acceptable(const Tensor & self); // aten::cudnn_is_acceptable(Tensor self) -> bool
Tensor constant_pad_nd(const Tensor & self, IntArrayRef pad, Scalar value); // aten::constant_pad_nd(Tensor self, int[] pad, Scalar value=0) -> Tensor
Tensor contiguous(const Tensor & self, MemoryFormat memory_format); // aten::contiguous(Tensor self, *, MemoryFormat memory_format=contiguous_format) -> Tensor
Tensor convolution(const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool transposed, IntArrayRef output_padding, int64_t groups); // aten::convolution(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups) -> Tensor
Tensor _convolution(const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool transposed, IntArrayRef output_padding, int64_t groups, bool benchmark, bool deterministic, bool cudnn_enabled); // aten::_convolution(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups, bool benchmark, bool deterministic, bool cudnn_enabled) -> Tensor
Tensor _convolution_nogroup(const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool transposed, IntArrayRef output_padding); // aten::_convolution_nogroup(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding) -> Tensor
std::tuple<Tensor,Tensor,Tensor> _convolution_double_backward(const Tensor & ggI, const Tensor & ggW, const Tensor & ggb, const Tensor & gO, const Tensor & weight, const Tensor & self, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool transposed, IntArrayRef output_padding, int64_t groups, bool benchmark, bool deterministic, bool cudnn_enabled, std::array<bool,3> output_mask); // aten::_convolution_double_backward(Tensor? ggI, Tensor? ggW, Tensor? ggb, Tensor gO, Tensor weight, Tensor self, int[] stride, int[] padding, int[] dilation, bool transposed, int[] output_padding, int groups, bool benchmark, bool deterministic, bool cudnn_enabled, bool[3] output_mask) -> (Tensor, Tensor, Tensor)
Tensor conv1d(const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, int64_t groups); // aten::conv1d(Tensor input, Tensor weight, Tensor? bias=None, int[1] stride=1, int[1] padding=0, int[1] dilation=1, int groups=1) -> Tensor
Tensor conv2d(const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, int64_t groups); // aten::conv2d(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1, int groups=1) -> Tensor
Tensor conv3d(const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, int64_t groups); // aten::conv3d(Tensor input, Tensor weight, Tensor? bias=None, int[3] stride=1, int[3] padding=0, int[3] dilation=1, int groups=1) -> Tensor
Tensor conv_tbc(const Tensor & self, const Tensor & weight, const Tensor & bias, int64_t pad); // aten::conv_tbc(Tensor self, Tensor weight, Tensor bias, int pad=0) -> Tensor
std::tuple<Tensor,Tensor,Tensor> conv_tbc_backward(const Tensor & self, const Tensor & input, const Tensor & weight, const Tensor & bias, int64_t pad); // aten::conv_tbc_backward(Tensor self, Tensor input, Tensor weight, Tensor bias, int pad) -> (Tensor, Tensor, Tensor)
Tensor conv_transpose1d(const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, int64_t groups, IntArrayRef dilation); // aten::conv_transpose1d(Tensor input, Tensor weight, Tensor? bias=None, int[1] stride=1, int[1] padding=0, int[1] output_padding=0, int groups=1, int[1] dilation=1) -> Tensor
Tensor conv_transpose2d(const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, int64_t groups, IntArrayRef dilation); // aten::conv_transpose2d(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] output_padding=0, int groups=1, int[2] dilation=1) -> Tensor
Tensor conv_transpose3d(const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, int64_t groups, IntArrayRef dilation); // aten::conv_transpose3d(Tensor input, Tensor weight, Tensor? bias=None, int[3] stride=1, int[3] padding=0, int[3] output_padding=0, int groups=1, int[3] dilation=1) -> Tensor
Tensor & copy_(Tensor & self, const Tensor & src, bool non_blocking); // aten::copy_(Tensor(a!) self, Tensor src, bool non_blocking=False) -> Tensor(a!)
Tensor _copy_from(const Tensor & self, const Tensor & dst, bool non_blocking); // aten::_copy_from(Tensor self, Tensor dst, bool non_blocking=False) -> Tensor
Tensor cos(const Tensor & self); // aten::cos(Tensor self) -> Tensor
Tensor & cos_(Tensor & self); // aten::cos_(Tensor(a!) self) -> Tensor(a!)
Tensor & cos_out(Tensor & out, const Tensor & self); // aten::cos(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
Tensor cosh(const Tensor & self); // aten::cosh(Tensor self) -> Tensor
Tensor & cosh_(Tensor & self); // aten::cosh_(Tensor(a!) self) -> Tensor(a!)
Tensor & cosh_out(Tensor & out, const Tensor & self); // aten::cosh(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
Tensor cosine_embedding_loss(const Tensor & input1, const Tensor & input2, const Tensor & target, double margin, int64_t reduction); // aten::cosine_embedding_loss(Tensor input1, Tensor input2, Tensor target, float margin=0.0, int reduction=Mean) -> Tensor
Tensor cudnn_affine_grid_generator(const Tensor & theta, int64_t N, int64_t C, int64_t H, int64_t W); // aten::cudnn_affine_grid_generator(Tensor theta, int N, int C, int H, int W) -> Tensor grid
Tensor cudnn_affine_grid_generator_backward(const Tensor & grad, int64_t N, int64_t C, int64_t H, int64_t W); // aten::cudnn_affine_grid_generator_backward(Tensor grad, int N, int C, int H, int W) -> Tensor grad_theta
std::tuple<Tensor,Tensor,Tensor> cudnn_batch_norm(const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double exponential_average_factor, double epsilon); // aten::cudnn_batch_norm(Tensor input, Tensor weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float exponential_average_factor, float epsilon) -> (Tensor, Tensor, Tensor)
std::tuple<Tensor,Tensor,Tensor> cudnn_batch_norm_backward(const Tensor & input, const Tensor & grad_output, const Tensor & weight, const Tensor & running_mean, const Tensor & running_var, const Tensor & save_mean, const Tensor & save_var, double epsilon); // aten::cudnn_batch_norm_backward(Tensor input, Tensor grad_output, Tensor weight, Tensor? running_mean, Tensor? running_var, Tensor? save_mean, Tensor? save_var, float epsilon) -> (Tensor, Tensor, Tensor)
Tensor cudnn_convolution(const Tensor & self, const Tensor & weight, const Tensor & bias, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic); // aten::cudnn_convolution(Tensor self, Tensor weight, Tensor? bias, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor
Tensor cudnn_convolution_backward_input(IntArrayRef self_size, const Tensor & grad_output, const Tensor & weight, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic); // aten::cudnn_convolution_backward_input(int[] self_size, Tensor grad_output, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor
std::tuple<Tensor,Tensor,Tensor> cudnn_convolution_backward(const Tensor & self, const Tensor & grad_output, const Tensor & weight, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, std::array<bool,3> output_mask); // aten::cudnn_convolution_backward(Tensor self, Tensor grad_output, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic, bool[3] output_mask) -> (Tensor, Tensor, Tensor)
Tensor cudnn_convolution_backward_bias(const Tensor & grad_output); // aten::cudnn_convolution_backward_bias(Tensor grad_output) -> Tensor
Tensor cudnn_convolution_backward_weight(IntArrayRef weight_size, const Tensor & grad_output, const Tensor & self, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic); // aten::cudnn_convolution_backward_weight(int[] weight_size, Tensor grad_output, Tensor self, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor
Tensor cudnn_convolution_transpose(const Tensor & self, const Tensor & weight, const Tensor & bias, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic); // aten::cudnn_convolution_transpose(Tensor self, Tensor weight, Tensor? bias, int[] padding, int[] output_padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor
std::tuple<Tensor,Tensor,Tensor> cudnn_convolution_transpose_backward(const Tensor & self, const Tensor & grad_output, const Tensor & weight, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, std::array<bool,3> output_mask); // aten::cudnn_convolution_transpose_backward(Tensor self, Tensor grad_output, Tensor weight, int[] padding, int[] output_padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic, bool[3] output_mask) -> (Tensor, Tensor, Tensor)
Tensor cudnn_convolution_transpose_backward_bias(const Tensor & grad_output); // aten::cudnn_convolution_transpose_backward_bias(Tensor grad_output) -> Tensor
Tensor cudnn_convolution_transpose_backward_input(const Tensor & grad_output, const Tensor & weight, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic); // aten::cudnn_convolution_transpose_backward_input(Tensor grad_output, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor
Tensor cudnn_convolution_transpose_backward_weight(IntArrayRef weight_size, const Tensor & grad_output, const Tensor & self, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic); // aten::cudnn_convolution_transpose_backward_weight(int[] weight_size, Tensor grad_output, Tensor self, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor
Tensor cudnn_grid_sampler(const Tensor & self, const Tensor & grid); // aten::cudnn_grid_sampler(Tensor self, Tensor grid) -> Tensor output
std::tuple<Tensor,Tensor> cudnn_grid_sampler_backward(const Tensor & self, const Tensor & grid, const Tensor & grad_output); // aten::cudnn_grid_sampler_backward(Tensor self, Tensor grid, Tensor grad_output) -> (Tensor grad_self, Tensor grad_grid)
Tensor cumsum(const Tensor & self, int64_t dim, c10::optional<ScalarType> dtype); // aten::cumsum(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor
Tensor & cumsum_out(Tensor & out, const Tensor & self, int64_t dim, c10::optional<ScalarType> dtype); // aten::cumsum(Tensor self, int dim, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
Tensor cumprod(const Tensor & self, int64_t dim, c10::optional<ScalarType> dtype); // aten::cumprod(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor
Tensor & cumprod_out(Tensor & out, const Tensor & self, int64_t dim, c10::optional<ScalarType> dtype); // aten::cumprod(Tensor self, int dim, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
Tensor ctc_loss(const Tensor & log_probs, const Tensor & targets, IntArrayRef input_lengths, IntArrayRef target_lengths, int64_t blank, int64_t reduction, bool zero_infinity); // aten::ctc_loss(Tensor log_probs, Tensor targets, int[] input_lengths, int[] target_lengths, int blank=0, int reduction=Mean, bool zero_infinity=False) -> Tensor
Tensor ctc_loss(const Tensor & log_probs, const Tensor & targets, const Tensor & input_lengths, const Tensor & target_lengths, int64_t blank, int64_t reduction, bool zero_infinity); // aten::ctc_loss(Tensor log_probs, Tensor targets, Tensor input_lengths, Tensor target_lengths, int blank=0, int reduction=Mean, bool zero_infinity=False) -> Tensor
std::tuple<Tensor,Tensor> _ctc_loss(const Tensor & log_probs, const Tensor & targets, IntArrayRef input_lengths, IntArrayRef target_lengths, int64_t blank, bool zero_infinity); // aten::_ctc_loss(Tensor log_probs, Tensor targets, int[] input_lengths, int[] target_lengths, int blank=0, bool zero_infinity=False) -> (Tensor, Tensor)
Tensor _ctc_loss_backward(const Tensor & grad, const Tensor & log_probs, const Tensor & targets, IntArrayRef input_lengths, IntArrayRef target_lengths, const Tensor & neg_log_likelihood, const Tensor & log_alpha, int64_t blank, bool zero_infinity); // aten::_ctc_loss_backward(Tensor grad, Tensor log_probs, Tensor targets, int[] input_lengths, int[] target_lengths, Tensor neg_log_likelihood, Tensor log_alpha, int blank, bool zero_infinity=False) -> Tensor
Tensor det(const Tensor & self); // aten::det(Tensor self) -> Tensor
Tensor diag_embed(const Tensor & self, int64_t offset, int64_t dim1, int64_t dim2); // aten::diag_embed(Tensor self, int offset=0, int dim1=-2, int dim2=-1) -> Tensor
Tensor diagflat(const Tensor & self, int64_t offset); // aten::diagflat(Tensor self, int offset=0) -> Tensor
Tensor diagonal(const Tensor & self, int64_t offset, int64_t dim1, int64_t dim2); // aten::diagonal(Tensor(a) self, int offset=0, int dim1=0, int dim2=1) -> Tensor(a)
Tensor & fill_diagonal_(Tensor & self, Scalar fill_value, bool wrap); // aten::fill_diagonal_(Tensor(a!) self, Scalar fill_value, bool wrap=False) -> Tensor(a!)
Tensor div(const Tensor & self, const Tensor & other); // aten::div(Tensor self, Tensor other) -> Tensor
Tensor & div_(Tensor & self, const Tensor & other); // aten::div_(Tensor(a!) self, Tensor other) -> Tensor(a!)
Tensor & div_out(Tensor & out, const Tensor & self, const Tensor & other); // aten::div(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
Tensor div(const Tensor & self, Scalar other); // aten::div(Tensor self, Scalar other) -> Tensor
Tensor & div_(Tensor & self, Scalar other); // aten::div_(Tensor(a!) self, Scalar other) -> Tensor(a!)
Tensor dot(const Tensor & self, const Tensor & tensor); // aten::dot(Tensor self, Tensor tensor) -> Tensor
Tensor & dot_out(Tensor & out, const Tensor & self, const Tensor & tensor); // aten::dot(Tensor self, Tensor tensor, *, Tensor(a!) out) -> Tensor(a!)
Tensor einsum(std::string equation, TensorList tensors); // aten::einsum(str equation, Tensor[] tensors) -> Tensor
Tensor embedding(const Tensor & weight, const Tensor & indices, int64_t padding_idx, bool scale_grad_by_freq, bool sparse); // aten::embedding(Tensor weight, Tensor indices, int padding_idx=-1, bool scale_grad_by_freq=False, bool sparse=False) -> Tensor
Tensor embedding_backward(const Tensor & grad, const Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq, bool sparse); // aten::embedding_backward(Tensor grad, Tensor indices, int num_weights, int padding_idx, bool scale_grad_by_freq, bool sparse) -> Tensor
Tensor embedding_dense_backward(const Tensor & grad_output, const Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq); // aten::embedding_dense_backward(Tensor grad_output, Tensor indices, int num_weights, int padding_idx, bool scale_grad_by_freq) -> Tensor
Tensor & embedding_renorm_(Tensor & self, const Tensor & indices, double max_norm, double norm_type); // aten::embedding_renorm_(Tensor(a!) self, Tensor indices, float max_norm, float norm_type) -> Tensor(a!)
Tensor embedding_sparse_backward(const Tensor & grad, const Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq); // aten::embedding_sparse_backward(Tensor grad, Tensor indices, int num_weights, int padding_idx, bool scale_grad_by_freq) -> Tensor
std::tuple<Tensor,Tensor,Tensor,Tensor> embedding_bag(const Tensor & weight, const Tensor & indices, const Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse, const Tensor & per_sample_weights); // aten::embedding_bag(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq=False, int mode=0, bool sparse=False, Tensor? per_sample_weights=None) -> (Tensor, Tensor, Tensor, Tensor)
std::tuple<Tensor,Tensor,Tensor,Tensor> _embedding_bag(const Tensor & weight, const Tensor & indices, const Tensor & offsets, bool scale_grad_by_freq, int64_t mode, bool sparse, const Tensor & per_sample_weights); // aten::_embedding_bag(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq=False, int mode=0, bool sparse=False, Tensor? per_sample_weights=None) -> (Tensor, Tensor, Tensor, Tensor)
Tensor _embedding_bag_backward(const Tensor & grad, const Tensor & indices, const Tensor & offsets, const Tensor & offset2bag, const Tensor & bag_size, const Tensor & maximum_indices, int64_t num_weights, bool scale_grad_by_freq, int64_t mode, bool sparse, const Tensor & per_sample_weights); // aten::_embedding_bag_backward(Tensor grad, Tensor indices, Tensor offsets, Tensor offset2bag, Tensor bag_size, Tensor maximum_indices, int num_weights, bool scale_grad_by_freq, int mode, bool sparse, Tensor? per_sample_weights) -> Tensor
Tensor _embedding_bag_sparse_backward(const Tensor & grad, const Tensor & indices, const Tensor & offsets, const Tensor & offset2bag, const Tensor & bag_size, int64_t num_weights, bool scale_grad_by_freq, int64_t mode, const Tensor & per_sample_weights); // aten::_embedding_bag_sparse_backward(Tensor grad, Tensor indices, Tensor offsets, Tensor offset2bag, Tensor bag_size, int num_weights, bool scale_grad_by_freq, int mode, Tensor? per_sample_weights) -> Tensor
Tensor _embedding_bag_dense_backward(const Tensor & grad, const Tensor & indices, const Tensor & offsets, const Tensor & offset2bag, const Tensor & bag_size, const Tensor & maximum_indices, int64_t num_weights, bool scale_grad_by_freq, int64_t mode, const Tensor & per_sample_weights); // aten::_embedding_bag_dense_backward(Tensor grad, Tensor indices, Tensor offsets, Tensor offset2bag, Tensor bag_size, Tensor maximum_indices, int num_weights, bool scale_grad_by_freq, int mode, Tensor? per_sample_weights) -> Tensor
Tensor _embedding_bag_per_sample_weights_backward(const Tensor & grad, const Tensor & weight, const Tensor & indices, const Tensor & offsets, const Tensor & offset2bag, int64_t mode); // aten::_embedding_bag_per_sample_weights_backward(Tensor grad, Tensor weight, Tensor indices, Tensor offsets, Tensor offset2bag, int mode) -> Tensor
Tensor empty(IntArrayRef size, const TensorOptions & options, c10::optional<MemoryFormat> memory_format); // aten::empty(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
Tensor _empty_affine_quantized(IntArrayRef size, const TensorOptions & options, double scale, int64_t zero_point, c10::optional<MemoryFormat> memory_format); // aten::_empty_affine_quantized(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, float scale=1, int zero_point=0, MemoryFormat? memory_format=contiguous_format) -> Tensor
Tensor & resize_(Tensor & self, IntArrayRef size); // aten::resize_(Tensor(a!) self, int[] size) -> Tensor(a!)
Tensor & empty_out(Tensor & out, IntArrayRef size, c10::optional<MemoryFormat> memory_format); // aten::empty(int[] size, *, MemoryFormat? memory_format=None, Tensor(a!) out) -> Tensor(a!)
Tensor empty_like(const Tensor & self); // aten::empty_like(Tensor self) -> Tensor
Tensor empty_like(const Tensor & self, const TensorOptions & options, c10::optional<MemoryFormat> memory_format); // aten::empty_like(Tensor self, *, ScalarType dtype, Layout layout, Device device, bool pin_memory=False, MemoryFormat? memory_format=contiguous_format) -> Tensor
Tensor empty_strided(IntArrayRef size, IntArrayRef stride, const TensorOptions & options); // aten::empty_strided(int[] size, int[] stride, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
Tensor erf(const Tensor & self); // aten::erf(Tensor self) -> Tensor
Tensor & erf_(Tensor & self); // aten::erf_(Tensor(a!) self) -> Tensor(a!)
Tensor & erf_out(Tensor & out, const Tensor & self); // aten::erf(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
Tensor erfc(const Tensor & self); // aten::erfc(Tensor self) -> Tensor
Tensor & erfc_(Tensor & self); // aten::erfc_(Tensor(a!) self) -> Tensor(a!)
Tensor & erfc_out(Tensor & out, const Tensor & self); // aten::erfc(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
Tensor exp(const Tensor & self); // aten::exp(Tensor self) -> Tensor
Tensor & exp_(Tensor & self); // aten::exp_(Tensor(a!) self) -> Tensor(a!)
Tensor & exp_out(Tensor & out, const Tensor & self); // aten::exp(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
Tensor expm1(const Tensor & self); // aten::expm1(Tensor self) -> Tensor
Tensor & expm1_(Tensor & self); // aten::expm1_(Tensor(a!) self) -> Tensor(a!)
Tensor & expm1_out(Tensor & out, const Tensor & self); // aten::expm1(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
Tensor expand(const Tensor & self, IntArrayRef size, bool implicit); // aten::expand(Tensor(a) self, int[] size, *, bool implicit=False) -> Tensor(a)
Tensor expand_as(const Tensor & self, const Tensor & other); // aten::expand_as(Tensor self, Tensor other) -> Tensor
Tensor eye(int64_t n, const TensorOptions & options); // aten::eye(int n, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
Tensor eye(int64_t n, int64_t m, const TensorOptions & options); // aten::eye(int n, int m, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
Tensor & eye_out(Tensor & out, int64_t n); // aten::eye(int n, *, Tensor(a!) out) -> Tensor(a!)
Tensor & eye_out(Tensor & out, int64_t n, int64_t m); // aten::eye(int n, int m, *, Tensor(a!) out) -> Tensor(a!)
Tensor flatten(const Tensor & self, int64_t start_dim, int64_t end_dim); // aten::flatten(Tensor self, int start_dim=0, int end_dim=-1) -> Tensor
Tensor & fill_(Tensor & self, Scalar value); // aten::fill_(Tensor(a!) self, Scalar value) -> Tensor(a!)
Tensor & fill_(Tensor & self, const Tensor & value); // aten::fill_(Tensor(a!) self, Tensor value) -> Tensor(a!)
Tensor floor(const Tensor & self); // aten::floor(Tensor self) -> Tensor
Tensor & floor_(Tensor & self); // aten::floor_(Tensor(a!) self) -> Tensor(a!)
Tensor & floor_out(Tensor & out, const Tensor & self); // aten::floor(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
Tensor frac(const Tensor & self); // aten::frac(Tensor self) -> Tensor
Tensor & frac_(Tensor & self); // aten::frac_(Tensor(a!) self) -> Tensor(a!)
Tensor & frac_out(Tensor & out, const Tensor & self); // aten::frac(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
Tensor full(IntArrayRef size, Scalar fill_value, const TensorOptions & options); // aten::full(int[] size, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
Tensor & full_out(Tensor & out, IntArrayRef size, Scalar fill_value); // aten::full(int[] size, Scalar fill_value, *, Tensor(a!) out) -> Tensor(a!)
Tensor full_like(const Tensor & self, Scalar fill_value); // aten::full_like(Tensor self, Scalar fill_value) -> Tensor
Tensor full_like(const Tensor & self, Scalar fill_value, const TensorOptions & options); // aten::full_like(Tensor self, Scalar fill_value, *, ScalarType dtype, Layout layout, Device device, bool pin_memory=False) -> Tensor
Tensor from_file(std::string filename, c10::optional<bool> shared, c10::optional<int64_t> size, const TensorOptions & options); // aten::from_file(str filename, bool? shared=None, int? size=0, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
Tensor grid_sampler(const Tensor & input, const Tensor & grid, int64_t interpolation_mode, int64_t padding_mode); // aten::grid_sampler(Tensor input, Tensor grid, int interpolation_mode, int padding_mode) -> Tensor
Tensor grid_sampler_2d(const Tensor & input, const Tensor & grid, int64_t interpolation_mode, int64_t padding_mode); // aten::grid_sampler_2d(Tensor input, Tensor grid, int interpolation_mode, int padding_mode) -> Tensor
std::tuple<Tensor,Tensor> grid_sampler_2d_backward(const Tensor & grad_output, const Tensor & input, const Tensor & grid, int64_t interpolation_mode, int64_t padding_mode); // aten::grid_sampler_2d_backward(Tensor grad_output, Tensor input, Tensor grid, int interpolation_mode, int padding_mode) -> (Tensor, Tensor)
Tensor grid_sampler_3d(const Tensor & input, const Tensor & grid, int64_t interpolation_mode, int64_t padding_mode); // aten::grid_sampler_3d(Tensor input, Tensor grid, int interpolation_mode, int padding_mode) -> Tensor
std::tuple<Tensor,Tensor> grid_sampler_3d_backward(const Tensor & grad_output, const Tensor & input, const Tensor & grid, int64_t interpolation_mode, int64_t padding_mode); // aten::grid_sampler_3d_backward(Tensor grad_output, Tensor input, Tensor grid, int interpolation_mode, int padding_mode) -> (Tensor, Tensor)
Tensor hann_window(int64_t window_length, const TensorOptions & options); // aten::hann_window(int window_length, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
Tensor hann_window(int64_t window_length, bool periodic, const TensorOptions & options); // aten::hann_window(int window_length, bool periodic, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
Tensor hamming_window(int64_t window_length, const TensorOptions & options); // aten::hamming_window(int window_length, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
Tensor hamming_window(int64_t window_length, bool periodic, const TensorOptions & options); // aten::hamming_window(int window_length, bool periodic, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
Tensor hamming_window(int64_t window_length, bool periodic, double alpha, const TensorOptions & options); // aten::hamming_window(int window_length, bool periodic, float alpha, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
Tensor hamming_window(int64_t window_length, bool periodic, double alpha, double beta, const TensorOptions & options); // aten::hamming_window(int window_length, bool periodic, float alpha, float beta, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
Tensor hinge_embedding_loss(const Tensor & self, const Tensor & target, double margin, int64_t reduction); // aten::hinge_embedding_loss(Tensor self, Tensor target, float margin=1.0, int reduction=Mean) -> Tensor
Tensor ger(const Tensor & self, const Tensor & vec2); // aten::ger(Tensor self, Tensor vec2) -> Tensor
Tensor & ger_out(Tensor & out, const Tensor & self, const Tensor & vec2); // aten::ger(Tensor self, Tensor vec2, *, Tensor(a!) out) -> Tensor(a!)
Tensor group_norm(const Tensor & input, int64_t num_groups, const Tensor & weight, const Tensor & bias, double eps, bool cudnn_enabled); // aten::group_norm(Tensor input, int num_groups, Tensor? weight=None, Tensor? bias=None, float eps=1e-05, bool cudnn_enabled=True) -> Tensor
Tensor fft(const Tensor & self, int64_t signal_ndim, bool normalized); // aten::fft(Tensor self, int signal_ndim, bool normalized=False) -> Tensor
Tensor ifft(const Tensor & self, int64_t signal_ndim, bool normalized); // aten::ifft(Tensor self, int signal_ndim, bool normalized=False) -> Tensor
Tensor rfft(const Tensor & self, int64_t signal_ndim, bool normalized, bool onesided); // aten::rfft(Tensor self, int signal_ndim, bool normalized=False, bool onesided=True) -> Tensor
Tensor irfft(const Tensor & self, int64_t signal_ndim, bool normalized, bool onesided, IntArrayRef signal_sizes); // aten::irfft(Tensor self, int signal_ndim, bool normalized=False, bool onesided=True, int[] signal_sizes=[]) -> Tensor
Tensor _fft_with_size(const Tensor & self, int64_t signal_ndim, bool complex_input, bool complex_output, bool inverse, IntArrayRef checked_signal_sizes, bool normalized, bool onesided, IntArrayRef output_sizes); // aten::_fft_with_size(Tensor self, int signal_ndim, bool complex_input, bool complex_output, bool inverse, int[] checked_signal_sizes, bool normalized, bool onesided, int[] output_sizes) -> Tensor
int64_t _cufft_get_plan_cache_size(int64_t device_index); // aten::_cufft_get_plan_cache_size(int device_index) -> int
int64_t _cufft_get_plan_cache_max_size(int64_t device_index); // aten::_cufft_get_plan_cache_max_size(int device_index) -> int
void _cufft_set_plan_cache_max_size(int64_t device_index, int64_t max_size); // aten::_cufft_set_plan_cache_max_size(int device_index, int max_size) -> void
void _cufft_clear_plan_cache(int64_t device_index); // aten::_cufft_clear_plan_cache(int device_index) -> void
Tensor index(const Tensor & self, TensorList indices); // aten::index(Tensor self, Tensor?[] indices) -> Tensor
Tensor & index_copy_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source); // aten::index_copy_(Tensor(a!) self, int dim, Tensor index, Tensor source) -> Tensor(a!)
Tensor index_copy(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & source); // aten::index_copy(Tensor self, int dim, Tensor index, Tensor source) -> Tensor
Tensor & index_put_(Tensor & self, TensorList indices, const Tensor & values, bool accumulate); // aten::index_put_(Tensor(a!) self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor(a!)
Tensor index_put(const Tensor & self, TensorList indices, const Tensor & values, bool accumulate); // aten::index_put(Tensor self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor
Tensor & _index_put_impl_(Tensor & self, TensorList indices, const Tensor & values, bool accumulate, bool unsafe); // aten::_index_put_impl_(Tensor(a!) self, Tensor?[] indices, Tensor values, bool accumulate=False, bool unsafe=False) -> Tensor(a!)
Tensor instance_norm(const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool use_input_stats, double momentum, double eps, bool cudnn_enabled); // aten::instance_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool use_input_stats, float momentum, float eps, bool cudnn_enabled) -> Tensor
Tensor inverse(const Tensor & self); // aten::inverse(Tensor self) -> Tensor
Tensor & inverse_out(Tensor & out, const Tensor & self); // aten::inverse(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
Tensor _inverse_helper(const Tensor & self); // aten::_inverse_helper(Tensor self) -> Tensor
Tensor isclose(const Tensor & self, const Tensor & other, double rtol, double atol, bool equal_nan); // aten::isclose(Tensor self, Tensor other, float rtol=1e-05, float atol=1e-08, bool equal_nan=False) -> Tensor
Tensor isnan(const Tensor & self); // aten::isnan(Tensor self) -> Tensor
bool is_distributed(const Tensor & self); // aten::is_distributed(Tensor self) -> bool
bool is_floating_point(const Tensor & self); // aten::is_floating_point(Tensor self) -> bool
bool is_complex(const Tensor & self); // aten::is_complex(Tensor self) -> bool
bool is_nonzero(const Tensor & self); // aten::is_nonzero(Tensor self) -> bool
bool is_same_size(const Tensor & self, const Tensor & other); // aten::is_same_size(Tensor self, Tensor other) -> bool
bool is_signed(const Tensor & self); // aten::is_signed(Tensor self) -> bool
Tensor kl_div(const Tensor & self, const Tensor & target, int64_t reduction); // aten::kl_div(Tensor self, Tensor target, int reduction=Mean) -> Tensor
Tensor kl_div_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction); // aten::kl_div_backward(Tensor grad_output, Tensor self, Tensor target, int reduction=Mean) -> Tensor
std::tuple<Tensor,Tensor> kthvalue(const Tensor & self, int64_t k, int64_t dim, bool keepdim); // aten::kthvalue(Tensor self, int k, int dim=-1, bool keepdim=False) -> (Tensor values, Tensor indices)
std::tuple<Tensor &,Tensor &> kthvalue_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t k, int64_t dim, bool keepdim); // aten::kthvalue(Tensor self, int k, int dim=-1, bool keepdim=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
Tensor layer_norm(const Tensor & input, IntArrayRef normalized_shape, const Tensor & weight, const Tensor & bias, double eps, bool cudnn_enable); // aten::layer_norm(Tensor input, int[] normalized_shape, Tensor? weight=None, Tensor? bias=None, float eps=1e-05, bool cudnn_enable=True) -> Tensor
std::tuple<Tensor,Tensor,Tensor> native_layer_norm(const Tensor & input, const Tensor & weight, const Tensor & bias, int64_t M, int64_t N, double eps); // aten::native_layer_norm(Tensor input, Tensor? weight, Tensor? bias, int M, int N, float eps) -> (Tensor, Tensor, Tensor)
std::tuple<Tensor,Tensor,Tensor> native_layer_norm_backward(const Tensor & grad_out, const Tensor & input, const Tensor & mean, const Tensor & rstd, const Tensor & weight, int64_t M, int64_t N, std::array<bool,3> output_mask); // aten::native_layer_norm_backward(Tensor grad_out, Tensor input, Tensor mean, Tensor rstd, Tensor? weight, int M, int N, bool[3] output_mask) -> (Tensor, Tensor, Tensor)
std::tuple<Tensor,Tensor,Tensor> native_layer_norm_double_backward(const Tensor & ggI, const Tensor & ggW, const Tensor & ggb, const Tensor & gO, const Tensor & input, const Tensor & mean, const Tensor & rstd, const Tensor & weight, int64_t M, int64_t N, std::array<bool,3> output_mask); // aten::native_layer_norm_double_backward(Tensor? ggI, Tensor? ggW, Tensor? ggb, Tensor gO, Tensor input, Tensor mean, Tensor rstd, Tensor? weight, int M, int N, bool[3] output_mask) -> (Tensor, Tensor, Tensor)
Tensor linear(const Tensor & input, const Tensor & weight, const Tensor & bias); // aten::linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor
Tensor mkldnn_linear(const Tensor & input, const Tensor & weight, const Tensor & bias); // aten::mkldnn_linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor
Tensor fbgemm_linear_int8_weight(const Tensor & input, const Tensor & weight, const Tensor & packed, const Tensor & col_offsets, Scalar weight_scale, Scalar weight_zero_point, const Tensor & bias); // aten::fbgemm_linear_int8_weight(Tensor input, Tensor weight, Tensor packed, Tensor col_offsets, Scalar weight_scale, Scalar weight_zero_point, Tensor bias) -> Tensor
std::tuple<Tensor,Tensor,double,int64_t> fbgemm_linear_quantize_weight(const Tensor & input); // aten::fbgemm_linear_quantize_weight(Tensor input) -> (Tensor, Tensor, float, int)
Tensor fbgemm_pack_gemm_matrix_fp16(const Tensor & input); // aten::fbgemm_pack_gemm_matrix_fp16(Tensor input) -> Tensor
Tensor fbgemm_linear_fp16_weight(const Tensor & input, const Tensor & packed_weight, const Tensor & bias); // aten::fbgemm_linear_fp16_weight(Tensor input, Tensor packed_weight, Tensor bias) -> Tensor
Tensor fbgemm_pack_quantized_matrix(const Tensor & input, int64_t K, int64_t N); // aten::fbgemm_pack_quantized_matrix(Tensor input, int K, int N) -> Tensor
bool fbgemm_is_cpu_supported(); // aten::fbgemm_is_cpu_supported() -> bool
Tensor linspace(Scalar start, Scalar end, int64_t steps, const TensorOptions & options); // aten::linspace(Scalar start, Scalar end, int steps=100, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
Tensor & linspace_out(Tensor & out, Scalar start, Scalar end, int64_t steps); // aten::linspace(Scalar start, Scalar end, int steps=100, *, Tensor(a!) out) -> Tensor(a!)
Tensor log(const Tensor & self); // aten::log(Tensor self) -> Tensor
Tensor & log_(Tensor & self); // aten::log_(Tensor(a!) self) -> Tensor(a!)
Tensor & log_out(Tensor & out, const Tensor & self); // aten::log(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
Tensor log10(const Tensor & self); // aten::log10(Tensor self) -> Tensor
Tensor & log10_(Tensor & self); // aten::log10_(Tensor(a!) self) -> Tensor(a!)
Tensor & log10_out(Tensor & out, const Tensor & self); // aten::log10(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
Tensor log1p(const Tensor & self); // aten::log1p(Tensor self) -> Tensor
Tensor & log1p_(Tensor & self); // aten::log1p_(Tensor(a!) self) -> Tensor(a!)
Tensor & log1p_out(Tensor & out, const Tensor & self); // aten::log1p(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
Tensor log2(const Tensor & self); // aten::log2(Tensor self) -> Tensor
Tensor & log2_(Tensor & self); // aten::log2_(Tensor(a!) self) -> Tensor(a!)
Tensor & log2_out(Tensor & out, const Tensor & self); // aten::log2(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
Tensor logdet(const Tensor & self); // aten::logdet(Tensor self) -> Tensor
Tensor logspace(Scalar start, Scalar end, int64_t steps, double base, const TensorOptions & options); // aten::logspace(Scalar start, Scalar end, int steps=100, float base=10.0, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
Tensor & logspace_out(Tensor & out, Scalar start, Scalar end, int64_t steps, double base); // aten::logspace(Scalar start, Scalar end, int steps=100, float base=10.0, *, Tensor(a!) out) -> Tensor(a!)
Tensor log_softmax(const Tensor & self, int64_t dim, c10::optional<ScalarType> dtype); // aten::log_softmax(Tensor self, int dim, ScalarType? dtype=None) -> Tensor
Tensor _log_softmax(const Tensor & self, int64_t dim, bool half_to_float); // aten::_log_softmax(Tensor self, int dim, bool half_to_float) -> Tensor
Tensor _log_softmax_backward_data(const Tensor & grad_output, const Tensor & output, int64_t dim, const Tensor & self); // aten::_log_softmax_backward_data(Tensor grad_output, Tensor output, int dim, Tensor self) -> Tensor
Tensor logsumexp(const Tensor & self, IntArrayRef dim, bool keepdim); // aten::logsumexp(Tensor self, int[1] dim, bool keepdim=False) -> Tensor
Tensor & logsumexp_out(Tensor & out, const Tensor & self, IntArrayRef dim, bool keepdim); // aten::logsumexp(Tensor self, int[1] dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
Tensor margin_ranking_loss(const Tensor & input1, const Tensor & input2, const Tensor & target, double margin, int64_t reduction); // aten::margin_ranking_loss(Tensor input1, Tensor input2, Tensor target, float margin=0.0, int reduction=Mean) -> Tensor
Tensor matmul(const Tensor & self, const Tensor & other); // aten::matmul(Tensor self, Tensor other) -> Tensor
Tensor & matmul_out(Tensor & out, const Tensor & self, const Tensor & other); // aten::matmul(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
Tensor matrix_rank(const Tensor & self, double tol, bool symmetric); // aten::matrix_rank(Tensor self, float tol, bool symmetric=False) -> Tensor
Tensor matrix_rank(const Tensor & self, bool symmetric); // aten::matrix_rank(Tensor self, bool symmetric=False) -> Tensor
Tensor matrix_power(const Tensor & self, int64_t n); // aten::matrix_power(Tensor self, int n) -> Tensor
std::tuple<Tensor,Tensor> max(const Tensor & self, int64_t dim, bool keepdim); // aten::max(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)
std::tuple<Tensor &,Tensor &> max_out(Tensor & max, Tensor & max_values, const Tensor & self, int64_t dim, bool keepdim); // aten::max(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) max, Tensor(b!) max_values) -> (Tensor(a!) values, Tensor(b!) indices)
Tensor max_values(const Tensor & self, IntArrayRef dim, bool keepdim); // aten::max_values(Tensor self, int[1] dim, bool keepdim=False) -> Tensor
std::tuple<Tensor,Tensor> max_pool1d_with_indices(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode); // aten::max_pool1d_with_indices(Tensor self, int[1] kernel_size, int[1] stride=[], int[1] padding=0, int[1] dilation=1, bool ceil_mode=False) -> (Tensor, Tensor)
Tensor max_pool1d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode); // aten::max_pool1d(Tensor self, int[1] kernel_size, int[1] stride=[], int[1] padding=0, int[1] dilation=1, bool ceil_mode=False) -> Tensor
Tensor max_pool2d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode); // aten::max_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> Tensor
Tensor mkldnn_max_pool2d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode); // aten::mkldnn_max_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> Tensor
Tensor quantized_max_pool2d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation); // aten::quantized_max_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1) -> Tensor
Tensor max_pool3d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode); // aten::max_pool3d(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, int[3] dilation=1, bool ceil_mode=False) -> Tensor
Tensor mean(const Tensor & self, c10::optional<ScalarType> dtype); // aten::mean(Tensor self, *, ScalarType? dtype=None) -> Tensor
Tensor mean(const Tensor & self, IntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype); // aten::mean(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
Tensor & mean_out(Tensor & out, const Tensor & self, IntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype); // aten::mean(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
std::tuple<Tensor,Tensor> median(const Tensor & self, int64_t dim, bool keepdim); // aten::median(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)
std::tuple<Tensor &,Tensor &> median_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool keepdim); // aten::median(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
std::tuple<Tensor,Tensor> min(const Tensor & self, int64_t dim, bool keepdim); // aten::min(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)
std::tuple<Tensor &,Tensor &> min_out(Tensor & min, Tensor & min_indices, const Tensor & self, int64_t dim, bool keepdim); // aten::min(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) min, Tensor(b!) min_indices) -> (Tensor(a!) values, Tensor(b!) indices)
Tensor min_values(const Tensor & self, IntArrayRef dim, bool keepdim); // aten::min_values(Tensor self, int[1] dim, bool keepdim=False) -> Tensor
Tensor mkldnn_convolution(const Tensor & self, const Tensor & weight, const Tensor & bias, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups); // aten::mkldnn_convolution(Tensor self, Tensor weight, Tensor? bias, int[] padding, int[] stride, int[] dilation, int groups) -> Tensor
Tensor mkldnn_convolution_backward_input(IntArrayRef self_size, const Tensor & grad_output, const Tensor & weight, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool bias_defined); // aten::mkldnn_convolution_backward_input(int[] self_size, Tensor grad_output, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool bias_defined) -> Tensor
std::tuple<Tensor,Tensor> mkldnn_convolution_backward_weights(IntArrayRef weight_size, const Tensor & grad_output, const Tensor & self, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool bias_defined); // aten::mkldnn_convolution_backward_weights(int[] weight_size, Tensor grad_output, Tensor self, int[] padding, int[] stride, int[] dilation, int groups, bool bias_defined) -> (Tensor, Tensor)
std::tuple<Tensor,Tensor,Tensor> mkldnn_convolution_backward(const Tensor & self, const Tensor & grad_output, const Tensor & weight, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, std::array<bool,3> output_mask); // aten::mkldnn_convolution_backward(Tensor self, Tensor grad_output, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool[3] output_mask) -> (Tensor, Tensor, Tensor)
std::tuple<Tensor,Tensor,Tensor> miopen_batch_norm(const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double exponential_average_factor, double epsilon); // aten::miopen_batch_norm(Tensor input, Tensor weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float exponential_average_factor, float epsilon) -> (Tensor, Tensor, Tensor)
std::tuple<Tensor,Tensor,Tensor> miopen_batch_norm_backward(const Tensor & input, const Tensor & grad_output, const Tensor & weight, const Tensor & running_mean, const Tensor & running_var, const Tensor & save_mean, const Tensor & save_var, double epsilon); // aten::miopen_batch_norm_backward(Tensor input, Tensor grad_output, Tensor weight, Tensor? running_mean, Tensor? running_var, Tensor? save_mean, Tensor? save_var, float epsilon) -> (Tensor, Tensor, Tensor)
Tensor miopen_convolution(const Tensor & self, const Tensor & weight, const Tensor & bias, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic); // aten::miopen_convolution(Tensor self, Tensor weight, Tensor? bias, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor
Tensor miopen_convolution_backward_input(IntArrayRef self_size, const Tensor & grad_output, const Tensor & weight, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic); // aten::miopen_convolution_backward_input(int[] self_size, Tensor grad_output, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor
std::tuple<Tensor,Tensor,Tensor> miopen_convolution_backward(const Tensor & self, const Tensor & grad_output, const Tensor & weight, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, std::array<bool,3> output_mask); // aten::miopen_convolution_backward(Tensor self, Tensor grad_output, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic, bool[3] output_mask) -> (Tensor, Tensor, Tensor)
Tensor miopen_convolution_backward_bias(const Tensor & grad_output); // aten::miopen_convolution_backward_bias(Tensor grad_output) -> Tensor
Tensor miopen_convolution_backward_weight(IntArrayRef weight_size, const Tensor & grad_output, const Tensor & self, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic); // aten::miopen_convolution_backward_weight(int[] weight_size, Tensor grad_output, Tensor self, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor
Tensor miopen_convolution_transpose(const Tensor & self, const Tensor & weight, const Tensor & bias, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic); // aten::miopen_convolution_transpose(Tensor self, Tensor weight, Tensor? bias, int[] padding, int[] output_padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor
std::tuple<Tensor,Tensor,Tensor> miopen_convolution_transpose_backward(const Tensor & self, const Tensor & grad_output, const Tensor & weight, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, std::array<bool,3> output_mask); // aten::miopen_convolution_transpose_backward(Tensor self, Tensor grad_output, Tensor weight, int[] padding, int[] output_padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic, bool[3] output_mask) -> (Tensor, Tensor, Tensor)
Tensor miopen_convolution_transpose_backward_input(const Tensor & grad_output, const Tensor & weight, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic); // aten::miopen_convolution_transpose_backward_input(Tensor grad_output, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor
Tensor miopen_convolution_transpose_backward_weight(IntArrayRef weight_size, const Tensor & grad_output, const Tensor & self, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic); // aten::miopen_convolution_transpose_backward_weight(int[] weight_size, Tensor grad_output, Tensor self, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor
Tensor miopen_depthwise_convolution(const Tensor & self, const Tensor & weight, const Tensor & bias, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic); // aten::miopen_depthwise_convolution(Tensor self, Tensor weight, Tensor? bias, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor
Tensor miopen_depthwise_convolution_backward_input(IntArrayRef self_size, const Tensor & grad_output, const Tensor & weight, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic); // aten::miopen_depthwise_convolution_backward_input(int[] self_size, Tensor grad_output, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor
std::tuple<Tensor,Tensor,Tensor> miopen_depthwise_convolution_backward(const Tensor & self, const Tensor & grad_output, const Tensor & weight, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic, std::array<bool,3> output_mask); // aten::miopen_depthwise_convolution_backward(Tensor self, Tensor grad_output, Tensor weight, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic, bool[3] output_mask) -> (Tensor, Tensor, Tensor)
Tensor miopen_depthwise_convolution_backward_weight(IntArrayRef weight_size, const Tensor & grad_output, const Tensor & self, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups, bool benchmark, bool deterministic); // aten::miopen_depthwise_convolution_backward_weight(int[] weight_size, Tensor grad_output, Tensor self, int[] padding, int[] stride, int[] dilation, int groups, bool benchmark, bool deterministic) -> Tensor
std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor> miopen_rnn(const Tensor & input, TensorList weight, int64_t weight_stride0, const Tensor & hx, const Tensor & cx, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, IntArrayRef batch_sizes, const Tensor & dropout_state); // aten::miopen_rnn(Tensor input, Tensor[] weight, int weight_stride0, Tensor hx, Tensor? cx, int mode, int hidden_size, int num_layers, bool batch_first, float dropout, bool train, bool bidirectional, int[] batch_sizes, Tensor? dropout_state) -> (Tensor, Tensor, Tensor, Tensor, Tensor)
std::tuple<Tensor,Tensor,Tensor,std::vector<Tensor>> miopen_rnn_backward(const Tensor & input, TensorList weight, int64_t weight_stride0, const Tensor & weight_buf, const Tensor & hx, const Tensor & cx, const Tensor & output, const Tensor & grad_output, const Tensor & grad_hy, const Tensor & grad_cy, int64_t mode, int64_t hidden_size, int64_t num_layers, bool batch_first, double dropout, bool train, bool bidirectional, IntArrayRef batch_sizes, const Tensor & dropout_state, const Tensor & reserve, std::array<bool,4> output_mask); // aten::miopen_rnn_backward(Tensor input, Tensor[] weight, int weight_stride0, Tensor weight_buf, Tensor hx, Tensor? cx, Tensor output, Tensor? grad_output, Tensor? grad_hy, Tensor? grad_cy, int mode, int hidden_size, int num_layers, bool batch_first, float dropout, bool train, bool bidirectional, int[] batch_sizes, Tensor? dropout_state, Tensor reserve, bool[4] output_mask) -> (Tensor, Tensor, Tensor, Tensor[])
Tensor mm(const Tensor & self, const Tensor & mat2); // aten::mm(Tensor self, Tensor mat2) -> Tensor
Tensor & mm_out(Tensor & out, const Tensor & self, const Tensor & mat2); // aten::mm(Tensor self, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)
Tensor _sparse_mm(const Tensor & sparse, const Tensor & dense); // aten::_sparse_mm(Tensor sparse, Tensor dense) -> Tensor
std::tuple<Tensor,Tensor> mode(const Tensor & self, int64_t dim, bool keepdim); // aten::mode(Tensor self, int dim=-1, bool keepdim=False) -> (Tensor values, Tensor indices)
std::tuple<Tensor &,Tensor &> mode_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool keepdim); // aten::mode(Tensor self, int dim=-1, bool keepdim=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
Tensor mul(const Tensor & self, const Tensor & other); // aten::mul(Tensor self, Tensor other) -> Tensor
Tensor & mul_(Tensor & self, const Tensor & other); // aten::mul_(Tensor(a!) self, Tensor other) -> Tensor(a!)
Tensor & mul_out(Tensor & out, const Tensor & self, const Tensor & other); // aten::mul(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
Tensor mul(const Tensor & self, Scalar other); // aten::mul(Tensor self, Scalar other) -> Tensor
Tensor & mul_(Tensor & self, Scalar other); // aten::mul_(Tensor(a!) self, Scalar other) -> Tensor(a!)
Tensor mv(const Tensor & self, const Tensor & vec); // aten::mv(Tensor self, Tensor vec) -> Tensor
Tensor & mv_out(Tensor & out, const Tensor & self, const Tensor & vec); // aten::mv(Tensor self, Tensor vec, *, Tensor(a!) out) -> Tensor(a!)
Tensor mvlgamma(const Tensor & self, int64_t p); // aten::mvlgamma(Tensor self, int p) -> Tensor
Tensor & mvlgamma_(Tensor & self, int64_t p); // aten::mvlgamma_(Tensor(a!) self, int p) -> Tensor(a!)
Tensor narrow_copy(const Tensor & self, int64_t dim, int64_t start, int64_t length); // aten::narrow_copy(Tensor self, int dim, int start, int length) -> Tensor
Tensor narrow(const Tensor & self, int64_t dim, int64_t start, int64_t length); // aten::narrow(Tensor(a) self, int dim, int start, int length) -> Tensor(a)
std::tuple<Tensor,Tensor,Tensor> native_batch_norm(const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & running_mean, const Tensor & running_var, bool training, double momentum, double eps); // aten::native_batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)
std::tuple<Tensor,Tensor> batch_norm_stats(const Tensor & input, double eps); // aten::batch_norm_stats(Tensor input, float eps) -> (Tensor, Tensor)
Tensor batch_norm_elemt(const Tensor & input, const Tensor & weight, const Tensor & bias, const Tensor & mean, const Tensor & invstd, double eps); // aten::batch_norm_elemt(Tensor input, Tensor? weight, Tensor? bias, Tensor mean, Tensor invstd, float eps) -> Tensor
std::tuple<Tensor,Tensor> batch_norm_gather_stats(const Tensor & input, const Tensor & mean, const Tensor & invstd, const Tensor & running_mean, const Tensor & running_var, double momentum, double eps, int64_t count); // aten::batch_norm_gather_stats(Tensor input, Tensor mean, Tensor invstd, Tensor? running_mean, Tensor? running_var, float momentum, float eps, int count) -> (Tensor, Tensor)
std::tuple<Tensor,Tensor> batch_norm_gather_stats_with_counts(const Tensor & input, const Tensor & mean, const Tensor & invstd, const Tensor & running_mean, const Tensor & running_var, double momentum, double eps, IntArrayRef counts); // aten::batch_norm_gather_stats_with_counts(Tensor input, Tensor mean, Tensor invstd, Tensor? running_mean, Tensor? running_var, float momentum, float eps, int[] counts) -> (Tensor, Tensor)
std::tuple<Tensor,Tensor,Tensor> native_batch_norm_backward(const Tensor & grad_out, const Tensor & input, const Tensor & weight, const Tensor & running_mean, const Tensor & running_var, const Tensor & save_mean, const Tensor & save_invstd, bool train, double eps, std::array<bool,3> output_mask); // aten::native_batch_norm_backward(Tensor grad_out, Tensor input, Tensor? weight, Tensor? running_mean, Tensor? running_var, Tensor? save_mean, Tensor? save_invstd, bool train, float eps, bool[3] output_mask) -> (Tensor, Tensor, Tensor)
std::tuple<Tensor,Tensor,Tensor,Tensor> batch_norm_backward_reduce(const Tensor & grad_out, const Tensor & input, const Tensor & mean, const Tensor & invstd, bool input_g, bool weight_g, bool bias_g); // aten::batch_norm_backward_reduce(Tensor grad_out, Tensor input, Tensor mean, Tensor invstd, bool input_g, bool weight_g, bool bias_g) -> (Tensor, Tensor, Tensor, Tensor)
Tensor batch_norm_backward_elemt(const Tensor & grad_out, const Tensor & input, const Tensor & mean, const Tensor & invstd, const Tensor & weight, const Tensor & mean_dy, const Tensor & mean_dy_xmu); // aten::batch_norm_backward_elemt(Tensor grad_out, Tensor input, Tensor mean, Tensor invstd, Tensor? weight, Tensor mean_dy, Tensor mean_dy_xmu) -> Tensor
std::tuple<Tensor,Tensor> batch_norm_update_stats(const Tensor & input, const Tensor & running_mean, const Tensor & running_var, double momentum); // aten::batch_norm_update_stats(Tensor input, Tensor? running_mean, Tensor? running_var, float momentum) -> (Tensor, Tensor)
bool _nnpack_available(); // aten::_nnpack_available() -> bool
Tensor _nnpack_spatial_convolution(const Tensor & input, const Tensor & weight, const Tensor & bias, IntArrayRef padding); // aten::_nnpack_spatial_convolution(Tensor input, Tensor weight, Tensor? bias, int[2] padding) -> Tensor
std::tuple<Tensor,Tensor,Tensor> _nnpack_spatial_convolution_backward(const Tensor & input, const Tensor & grad_output, const Tensor & weight, IntArrayRef padding, std::array<bool,3> output_mask); // aten::_nnpack_spatial_convolution_backward(Tensor input, Tensor grad_output, Tensor weight, int[2] padding, bool[3] output_mask) -> (Tensor, Tensor, Tensor)
Tensor _nnpack_spatial_convolution_backward_input(const Tensor & input, const Tensor & grad_output, const Tensor & weight, IntArrayRef padding); // aten::_nnpack_spatial_convolution_backward_input(Tensor input, Tensor grad_output, Tensor weight, int[2] padding) -> Tensor
Tensor _nnpack_spatial_convolution_backward_weight(const Tensor & input, IntArrayRef weightsize, const Tensor & grad_output, IntArrayRef padding); // aten::_nnpack_spatial_convolution_backward_weight(Tensor input, int[] weightsize, Tensor grad_output, int[2] padding) -> Tensor
Tensor ones(IntArrayRef size, const TensorOptions & options); // aten::ones(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
Tensor & ones_out(Tensor & out, IntArrayRef size); // aten::ones(int[] size, *, Tensor(a!) out) -> Tensor(a!)
Tensor ones_like(const Tensor & self); // aten::ones_like(Tensor self) -> Tensor
Tensor ones_like(const Tensor & self, const TensorOptions & options); // aten::ones_like(Tensor self, *, ScalarType dtype, Layout layout, Device device, bool pin_memory=False) -> Tensor
Tensor pairwise_distance(const Tensor & x1, const Tensor & x2, double p, double eps, bool keepdim); // aten::pairwise_distance(Tensor x1, Tensor x2, float p=2, float eps=1e-06, bool keepdim=False) -> Tensor
Tensor cdist(const Tensor & x1, const Tensor & x2, double p); // aten::cdist(Tensor x1, Tensor x2, float p=2) -> Tensor
Tensor _cdist_backward(const Tensor & grad, const Tensor & x1, const Tensor & x2, double p, const Tensor & cdist); // aten::_cdist_backward(Tensor grad, Tensor x1, Tensor x2, float p, Tensor cdist) -> Tensor
Tensor pdist(const Tensor & self, double p); // aten::pdist(Tensor self, float p=2) -> Tensor
Tensor _pdist_forward(const Tensor & self, double p); // aten::_pdist_forward(Tensor self, float p=2) -> Tensor
Tensor _pdist_backward(const Tensor & grad, const Tensor & self, double p, const Tensor & pdist); // aten::_pdist_backward(Tensor grad, Tensor self, float p, Tensor pdist) -> Tensor
Tensor cosine_similarity(const Tensor & x1, const Tensor & x2, int64_t dim, double eps); // aten::cosine_similarity(Tensor x1, Tensor x2, int dim=1, float eps=1e-08) -> Tensor
Tensor permute(const Tensor & self, IntArrayRef dims); // aten::permute(Tensor(a) self, int[] dims) -> Tensor(a)
Tensor numpy_T(const Tensor & self); // aten::numpy_T(Tensor(a) self) -> Tensor(a)
Tensor pixel_shuffle(const Tensor & self, int64_t upscale_factor); // aten::pixel_shuffle(Tensor self, int upscale_factor) -> Tensor
Tensor pin_memory(const Tensor & self); // aten::pin_memory(Tensor self) -> Tensor
Tensor pinverse(const Tensor & self, double rcond); // aten::pinverse(Tensor self, float rcond=1e-15) -> Tensor
Tensor poisson_nll_loss(const Tensor & input, const Tensor & target, bool log_input, bool full, double eps, int64_t reduction); // aten::poisson_nll_loss(Tensor input, Tensor target, bool log_input, bool full, float eps, int reduction) -> Tensor
Tensor scalar_tensor(Scalar s, const TensorOptions & options); // aten::scalar_tensor(Scalar s, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
Tensor rand(IntArrayRef size, const TensorOptions & options); // aten::rand(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
Tensor rand(IntArrayRef size, Generator * generator, const TensorOptions & options); // aten::rand(int[] size, *, Generator? generator, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
Tensor & rand_out(Tensor & out, IntArrayRef size); // aten::rand(int[] size, *, Tensor(a!) out) -> Tensor(a!)
Tensor & rand_out(Tensor & out, IntArrayRef size, Generator * generator); // aten::rand(int[] size, *, Generator? generator, Tensor(a!) out) -> Tensor(a!)
Tensor rand_like(const Tensor & self); // aten::rand_like(Tensor self) -> Tensor
Tensor rand_like(const Tensor & self, const TensorOptions & options); // aten::rand_like(Tensor self, *, ScalarType dtype, Layout layout, Device device, bool pin_memory=False) -> Tensor
Tensor randint(int64_t high, IntArrayRef size, const TensorOptions & options); // aten::randint(int high, int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
Tensor randint(int64_t high, IntArrayRef size, Generator * generator, const TensorOptions & options); // aten::randint(int high, int[] size, *, Generator? generator, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
Tensor randint(int64_t low, int64_t high, IntArrayRef size, const TensorOptions & options); // aten::randint(int low, int high, int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
Tensor randint(int64_t low, int64_t high, IntArrayRef size, Generator * generator, const TensorOptions & options); // aten::randint(int low, int high, int[] size, *, Generator? generator, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
Tensor & randint_out(Tensor & out, int64_t high, IntArrayRef size); // aten::randint(int high, int[] size, *, Tensor(a!) out) -> Tensor(a!)
Tensor & randint_out(Tensor & out, int64_t high, IntArrayRef size, Generator * generator); // aten::randint(int high, int[] size, *, Generator? generator, Tensor(a!) out) -> Tensor(a!)
Tensor & randint_out(Tensor & out, int64_t low, int64_t high, IntArrayRef size); // aten::randint(int low, int high, int[] size, *, Tensor(a!) out) -> Tensor(a!)
Tensor & randint_out(Tensor & out, int64_t low, int64_t high, IntArrayRef size, Generator * generator); // aten::randint(int low, int high, int[] size, *, Generator? generator, Tensor(a!) out) -> Tensor(a!)
Tensor randint_like(const Tensor & self, int64_t high); // aten::randint_like(Tensor self, int high) -> Tensor
Tensor randint_like(const Tensor & self, int64_t low, int64_t high); // aten::randint_like(Tensor self, int low, int high) -> Tensor
Tensor randint_like(const Tensor & self, int64_t high, const TensorOptions & options); // aten::randint_like(Tensor self, int high, *, ScalarType dtype, Layout layout, Device device, bool pin_memory=False) -> Tensor
Tensor randint_like(const Tensor & self, int64_t low, int64_t high, const TensorOptions & options); // aten::randint_like(Tensor self, int low, int high, *, ScalarType dtype, Layout layout, Device device, bool pin_memory=False) -> Tensor
Tensor randn(IntArrayRef size, const TensorOptions & options); // aten::randn(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
Tensor randn(IntArrayRef size, Generator * generator, const TensorOptions & options); // aten::randn(int[] size, *, Generator? generator, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
Tensor & randn_out(Tensor & out, IntArrayRef size); // aten::randn(int[] size, *, Tensor(a!) out) -> Tensor(a!)
Tensor & randn_out(Tensor & out, IntArrayRef size, Generator * generator); // aten::randn(int[] size, *, Generator? generator, Tensor(a!) out) -> Tensor(a!)
Tensor randn_like(const Tensor & self); // aten::randn_like(Tensor self) -> Tensor
Tensor randn_like(const Tensor & self, const TensorOptions & options); // aten::randn_like(Tensor self, *, ScalarType dtype, Layout layout, Device device, bool pin_memory=False) -> Tensor
Tensor randperm(int64_t n, const TensorOptions & options); // aten::randperm(int n, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
Tensor randperm(int64_t n, Generator * generator, const TensorOptions & options); // aten::randperm(int n, *, Generator? generator, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
Tensor & randperm_out(Tensor & out, int64_t n); // aten::randperm(int n, *, Tensor(a!) out) -> Tensor(a!)
Tensor & randperm_out(Tensor & out, int64_t n, Generator * generator); // aten::randperm(int n, *, Generator? generator, Tensor(a!) out) -> Tensor(a!)
Tensor range(Scalar start, Scalar end, Scalar step, const TensorOptions & options); // aten::range(Scalar start, Scalar end, Scalar step=1, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
Tensor range(Scalar start, Scalar end, const TensorOptions & options); // aten::range(Scalar start, Scalar end, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
Tensor & range_out(Tensor & out, Scalar start, Scalar end, Scalar step); // aten::range(Scalar start, Scalar end, Scalar step=1, *, Tensor(a!) out) -> Tensor(a!)
Tensor reciprocal(const Tensor & self); // aten::reciprocal(Tensor self) -> Tensor
Tensor & reciprocal_(Tensor & self); // aten::reciprocal_(Tensor(a!) self) -> Tensor(a!)
Tensor & reciprocal_out(Tensor & out, const Tensor & self); // aten::reciprocal(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
Tensor neg(const Tensor & self); // aten::neg(Tensor self) -> Tensor
Tensor & neg_(Tensor & self); // aten::neg_(Tensor(a!) self) -> Tensor(a!)
Tensor & neg_out(Tensor & out, const Tensor & self); // aten::neg(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
Tensor repeat(const Tensor & self, IntArrayRef repeats); // aten::repeat(Tensor self, int[] repeats) -> Tensor
Tensor repeat_interleave(const Tensor & repeats); // aten::repeat_interleave(Tensor repeats) -> Tensor
Tensor repeat_interleave(const Tensor & self, const Tensor & repeats, c10::optional<int64_t> dim); // aten::repeat_interleave(Tensor self, Tensor repeats, int? dim=None) -> Tensor
Tensor repeat_interleave(const Tensor & self, int64_t repeats, c10::optional<int64_t> dim); // aten::repeat_interleave(Tensor self, int repeats, int? dim=None) -> Tensor
Tensor reshape(const Tensor & self, IntArrayRef shape); // aten::reshape(Tensor self, int[] shape) -> Tensor
Tensor _mkldnn_reshape(const Tensor & self, IntArrayRef shape); // aten::_mkldnn_reshape(Tensor self, int[] shape) -> Tensor
Tensor reshape_as(const Tensor & self, const Tensor & other); // aten::reshape_as(Tensor self, Tensor other) -> Tensor
Tensor round(const Tensor & self); // aten::round(Tensor self) -> Tensor
Tensor & round_(Tensor & self); // aten::round_(Tensor(a!) self) -> Tensor(a!)
Tensor & round_out(Tensor & out, const Tensor & self); // aten::round(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
Tensor rrelu(const Tensor & self, Scalar lower, Scalar upper, bool training, Generator * generator); // aten::rrelu(Tensor self, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=False, Generator? generator=None) -> Tensor
Tensor & rrelu_(Tensor & self, Scalar lower, Scalar upper, bool training, Generator * generator); // aten::rrelu_(Tensor(a!) self, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=False, Generator? generator=None) -> Tensor(a!)
Tensor relu(const Tensor & self); // aten::relu(Tensor self) -> Tensor
Tensor & relu_(Tensor & self); // aten::relu_(Tensor(a!) self) -> Tensor(a!)
Tensor prelu(const Tensor & self, const Tensor & weight); // aten::prelu(Tensor self, Tensor weight) -> Tensor
std::tuple<Tensor,Tensor> prelu_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight); // aten::prelu_backward(Tensor grad_output, Tensor self, Tensor weight) -> (Tensor, Tensor)
Tensor gelu(const Tensor & self); // aten::gelu(Tensor self) -> Tensor
Tensor gelu_backward(const Tensor & grad, const Tensor & self); // aten::gelu_backward(Tensor grad, Tensor self) -> Tensor
Tensor hardshrink(const Tensor & self, Scalar lambd); // aten::hardshrink(Tensor self, Scalar lambd=0.5) -> Tensor
Tensor hardshrink_backward(const Tensor & grad_out, const Tensor & self, Scalar lambd); // aten::hardshrink_backward(Tensor grad_out, Tensor self, Scalar lambd) -> Tensor
Tensor rsqrt(const Tensor & self); // aten::rsqrt(Tensor self) -> Tensor
Tensor & rsqrt_(Tensor & self); // aten::rsqrt_(Tensor(a!) self) -> Tensor(a!)
Tensor & rsqrt_out(Tensor & out, const Tensor & self); // aten::rsqrt(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
Tensor select(const Tensor & self, int64_t dim, int64_t index); // aten::select(Tensor(a) self, int dim, int index) -> Tensor(a)
Tensor selu(const Tensor & self); // aten::selu(Tensor self) -> Tensor
Tensor & selu_(Tensor & self); // aten::selu_(Tensor(a!) self) -> Tensor(a!)
Tensor celu(const Tensor & self, Scalar alpha); // aten::celu(Tensor self, Scalar alpha=1.0) -> Tensor
Tensor & celu_(Tensor & self, Scalar alpha); // aten::celu_(Tensor(a!) self, Scalar alpha=1.0) -> Tensor(a!)
Tensor sigmoid(const Tensor & self); // aten::sigmoid(Tensor self) -> Tensor
Tensor & sigmoid_(Tensor & self); // aten::sigmoid_(Tensor(a!) self) -> Tensor(a!)
Tensor & sigmoid_out(Tensor & out, const Tensor & self); // aten::sigmoid(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
Tensor sin(const Tensor & self); // aten::sin(Tensor self) -> Tensor
Tensor & sin_(Tensor & self); // aten::sin_(Tensor(a!) self) -> Tensor(a!)
Tensor & sin_out(Tensor & out, const Tensor & self); // aten::sin(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
Tensor sinh(const Tensor & self); // aten::sinh(Tensor self) -> Tensor
Tensor & sinh_(Tensor & self); // aten::sinh_(Tensor(a!) self) -> Tensor(a!)
Tensor & sinh_out(Tensor & out, const Tensor & self); // aten::sinh(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
Tensor detach(const Tensor & self); // aten::detach(Tensor self) -> Tensor
Tensor & detach_(Tensor & self); // aten::detach_(Tensor(a!) self) -> Tensor(a!)
int64_t size(const Tensor & self, int64_t dim); // aten::size(Tensor self, int dim) -> int
Tensor slice(const Tensor & self, int64_t dim, int64_t start, int64_t end, int64_t step); // aten::slice(Tensor(a) self, int dim=0, int start=0, int end=9223372036854775807, int step=1) -> Tensor(a)
std::tuple<Tensor,Tensor> slogdet(const Tensor & self); // aten::slogdet(Tensor self) -> (Tensor sign, Tensor logabsdet)
Tensor smm(const Tensor & self, const Tensor & mat2); // aten::smm(Tensor self, Tensor mat2) -> Tensor
Tensor softmax(const Tensor & self, int64_t dim, c10::optional<ScalarType> dtype); // aten::softmax(Tensor self, int dim, ScalarType? dtype=None) -> Tensor
Tensor _softmax(const Tensor & self, int64_t dim, bool half_to_float); // aten::_softmax(Tensor self, int dim, bool half_to_float) -> Tensor
Tensor _softmax_backward_data(const Tensor & grad_output, const Tensor & output, int64_t dim, const Tensor & self); // aten::_softmax_backward_data(Tensor grad_output, Tensor output, int dim, Tensor self) -> Tensor
Tensor & _sparse_add_out(Tensor & out, const Tensor & self, const Tensor & other, Scalar alpha); // aten::_sparse_add(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
Tensor & _sparse_dense_add_out(Tensor & out, const Tensor & self, const Tensor & other, Scalar alpha); // aten::_sparse_dense_add(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
Tensor & _sparse_div_zerodim_out(Tensor & out, const Tensor & self, const Tensor & other); // aten::_sparse_div_zerodim(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
Tensor & _sparse_div_scalar_out(Tensor & out, const Tensor & self, Scalar other); // aten::_sparse_div_scalar(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
Tensor & _sparse_mul_out(Tensor & out, const Tensor & self, const Tensor & other); // aten::_sparse_mul(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
Tensor & _sparse_mul_zerodim_out(Tensor & out, const Tensor & self, const Tensor & other); // aten::_sparse_mul_zerodim(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
Tensor & _sparse_mul_scalar_out(Tensor & out, const Tensor & self, Scalar other); // aten::_sparse_mul_scalar(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
std::vector<Tensor> split(const Tensor & self, int64_t split_size, int64_t dim); // aten::split(Tensor(a) self, int split_size, int dim=0) -> Tensor(a)[]
std::vector<Tensor> split_with_sizes(const Tensor & self, IntArrayRef split_sizes, int64_t dim); // aten::split_with_sizes(Tensor self, int[] split_sizes, int dim=0) -> Tensor[]
Tensor squeeze(const Tensor & self); // aten::squeeze(Tensor(a) self) -> Tensor(a)
Tensor squeeze(const Tensor & self, int64_t dim); // aten::squeeze(Tensor(a) self, int dim) -> Tensor(a)
Tensor & squeeze_(Tensor & self); // aten::squeeze_(Tensor(a!) self) -> Tensor(a!)
Tensor & squeeze_(Tensor & self, int64_t dim); // aten::squeeze_(Tensor(a!) self, int dim) -> Tensor(a!)
Tensor sspaddmm(const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha); // aten::sspaddmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
Tensor & sspaddmm_out(Tensor & out, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha); // aten::sspaddmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
Tensor stack(TensorList tensors, int64_t dim); // aten::stack(Tensor[] tensors, int dim=0) -> Tensor
Tensor & stack_out(Tensor & out, TensorList tensors, int64_t dim); // aten::stack(Tensor[] tensors, int dim=0, *, Tensor(a!) out) -> Tensor(a!)
Tensor stft(const Tensor & self, int64_t n_fft, c10::optional<int64_t> hop_length, c10::optional<int64_t> win_length, const Tensor & window, bool normalized, bool onesided); // aten::stft(Tensor self, int n_fft, int? hop_length=None, int? win_length=None, Tensor? window=None, bool normalized=False, bool onesided=True) -> Tensor
int64_t stride(const Tensor & self, int64_t dim); // aten::stride(Tensor self, int dim) -> int
Tensor sum(const Tensor & self, c10::optional<ScalarType> dtype); // aten::sum(Tensor self, *, ScalarType? dtype=None) -> Tensor
Tensor sum(const Tensor & self, IntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype); // aten::sum(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
Tensor & sum_out(Tensor & out, const Tensor & self, IntArrayRef dim, bool keepdim, c10::optional<ScalarType> dtype); // aten::sum(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
Tensor sum_to_size(const Tensor & self, IntArrayRef size); // aten::sum_to_size(Tensor self, int[] size) -> Tensor
Tensor sqrt(const Tensor & self); // aten::sqrt(Tensor self) -> Tensor
Tensor & sqrt_(Tensor & self); // aten::sqrt_(Tensor(a!) self) -> Tensor(a!)
Tensor & sqrt_out(Tensor & out, const Tensor & self); // aten::sqrt(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
Tensor std(const Tensor & self, bool unbiased); // aten::std(Tensor self, bool unbiased=True) -> Tensor
Tensor std(const Tensor & self, IntArrayRef dim, bool unbiased, bool keepdim); // aten::std(Tensor self, int[1] dim, bool unbiased=True, bool keepdim=False) -> Tensor
std::tuple<Tensor,Tensor> std_mean(const Tensor & self, bool unbiased); // aten::std_mean(Tensor self, bool unbiased=True) -> (Tensor, Tensor)
std::tuple<Tensor,Tensor> std_mean(const Tensor & self, IntArrayRef dim, bool unbiased, bool keepdim); // aten::std_mean(Tensor self, int[1] dim, bool unbiased=True, bool keepdim=False) -> (Tensor, Tensor)
Tensor & std_out(Tensor & out, const Tensor & self, IntArrayRef dim, bool unbiased, bool keepdim); // aten::std(Tensor self, int[1] dim, bool unbiased=True, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
Tensor prod(const Tensor & self, c10::optional<ScalarType> dtype); // aten::prod(Tensor self, *, ScalarType? dtype=None) -> Tensor
Tensor prod(const Tensor & self, int64_t dim, bool keepdim, c10::optional<ScalarType> dtype); // aten::prod(Tensor self, int dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
Tensor & prod_out(Tensor & out, const Tensor & self, int64_t dim, bool keepdim, c10::optional<ScalarType> dtype); // aten::prod(Tensor self, int dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
Tensor t(const Tensor & self); // aten::t(Tensor(a) self) -> Tensor(a)
Tensor & t_(Tensor & self); // aten::t_(Tensor(a!) self) -> Tensor(a!)
Tensor tan(const Tensor & self); // aten::tan(Tensor self) -> Tensor
Tensor & tan_(Tensor & self); // aten::tan_(Tensor(a!) self) -> Tensor(a!)
Tensor & tan_out(Tensor & out, const Tensor & self); // aten::tan(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
Tensor tanh(const Tensor & self); // aten::tanh(Tensor self) -> Tensor
Tensor & tanh_(Tensor & self); // aten::tanh_(Tensor(a!) self) -> Tensor(a!)
Tensor & tanh_out(Tensor & out, const Tensor & self); // aten::tanh(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
Tensor tensordot(const Tensor & self, const Tensor & other, IntArrayRef dims_self, IntArrayRef dims_other); // aten::tensordot(Tensor self, Tensor other, int[] dims_self, int[] dims_other) -> Tensor
Tensor threshold(const Tensor & self, Scalar threshold, Scalar value); // aten::threshold(Tensor self, Scalar threshold, Scalar value) -> Tensor
Tensor & threshold_(Tensor & self, Scalar threshold, Scalar value); // aten::threshold_(Tensor(a!) self, Scalar threshold, Scalar value) -> Tensor(a!)
Tensor & threshold_out(Tensor & out, const Tensor & self, Scalar threshold, Scalar value); // aten::threshold(Tensor self, Scalar threshold, Scalar value, *, Tensor(a!) out) -> Tensor(a!)
Tensor threshold_backward(const Tensor & grad_output, const Tensor & self, Scalar threshold); // aten::threshold_backward(Tensor grad_output, Tensor self, Scalar threshold) -> Tensor
Tensor transpose(const Tensor & self, int64_t dim0, int64_t dim1); // aten::transpose(Tensor(a) self, int dim0, int dim1) -> Tensor(a)
Tensor _mkldnn_transpose(const Tensor & self, int64_t dim0, int64_t dim1); // aten::_mkldnn_transpose(Tensor self, int dim0, int dim1) -> Tensor
Tensor & transpose_(Tensor & self, int64_t dim0, int64_t dim1); // aten::transpose_(Tensor(a!) self, int dim0, int dim1) -> Tensor(a!)
Tensor & _mkldnn_transpose_(Tensor & self, int64_t dim0, int64_t dim1); // aten::_mkldnn_transpose_(Tensor(a!) self, int dim0, int dim1) -> Tensor(a!)
Tensor one_hot(const Tensor & self, int64_t num_classes); // aten::one_hot(Tensor self, int num_classes=-1) -> Tensor
Tensor flip(const Tensor & self, IntArrayRef dims); // aten::flip(Tensor self, int[] dims) -> Tensor
Tensor roll(const Tensor & self, IntArrayRef shifts, IntArrayRef dims); // aten::roll(Tensor self, int[1] shifts, int[1] dims=[]) -> Tensor
Tensor rot90(const Tensor & self, int64_t k, IntArrayRef dims); // aten::rot90(Tensor self, int k=1, int[] dims=[0,1]) -> Tensor
Tensor trapz(const Tensor & y, const Tensor & x, int64_t dim); // aten::trapz(Tensor y, Tensor x, *, int dim=-1) -> Tensor
Tensor trapz(const Tensor & y, double dx, int64_t dim); // aten::trapz(Tensor y, *, float dx=1, int dim=-1) -> Tensor
Tensor _trilinear(const Tensor & i1, const Tensor & i2, const Tensor & i3, IntArrayRef expand1, IntArrayRef expand2, IntArrayRef expand3, IntArrayRef sumdim, int64_t unroll_dim); // aten::_trilinear(Tensor i1, Tensor i2, Tensor i3, int[] expand1, int[] expand2, int[] expand3, int[] sumdim, int unroll_dim=1) -> Tensor
Tensor triplet_margin_loss(const Tensor & anchor, const Tensor & positive, const Tensor & negative, double margin, double p, double eps, bool swap, int64_t reduction); // aten::triplet_margin_loss(Tensor anchor, Tensor positive, Tensor negative, float margin=1.0, float p=2, float eps=1e-06, bool swap=False, int reduction=Mean) -> Tensor
Tensor trunc(const Tensor & self); // aten::trunc(Tensor self) -> Tensor
Tensor & trunc_(Tensor & self); // aten::trunc_(Tensor(a!) self) -> Tensor(a!)
Tensor & trunc_out(Tensor & out, const Tensor & self); // aten::trunc(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
Tensor type_as(const Tensor & self, const Tensor & other); // aten::type_as(Tensor self, Tensor other) -> Tensor
bool _has_compatible_shallow_copy_type(const Tensor & self, const Tensor & from); // aten::_has_compatible_shallow_copy_type(Tensor self, Tensor from) -> bool
std::tuple<Tensor,Tensor> _unique(const Tensor & self, bool sorted, bool return_inverse); // aten::_unique(Tensor self, bool sorted=True, bool return_inverse=False) -> (Tensor, Tensor)
std::tuple<Tensor,Tensor,Tensor> unique_dim(const Tensor & self, int64_t dim, bool sorted, bool return_inverse, bool return_counts); // aten::unique_dim(Tensor self, int dim, bool sorted=True, bool return_inverse=False, bool return_counts=False) -> (Tensor, Tensor, Tensor)
std::tuple<Tensor,Tensor,Tensor> unique_consecutive(const Tensor & self, bool return_inverse, bool return_counts, c10::optional<int64_t> dim); // aten::unique_consecutive(Tensor self, bool return_inverse=False, bool return_counts=False, int? dim=None) -> (Tensor, Tensor, Tensor)
std::tuple<Tensor,Tensor,Tensor> unique_dim_consecutive(const Tensor & self, int64_t dim, bool return_inverse, bool return_counts); // aten::unique_dim_consecutive(Tensor self, int dim, bool return_inverse=False, bool return_counts=False) -> (Tensor, Tensor, Tensor)
std::tuple<Tensor,Tensor,Tensor> _unique2(const Tensor & self, bool sorted, bool return_inverse, bool return_counts); // aten::_unique2(Tensor self, bool sorted=True, bool return_inverse=False, bool return_counts=False) -> (Tensor, Tensor, Tensor)
Tensor _unsafe_view(const Tensor & self, IntArrayRef size); // aten::_unsafe_view(Tensor self, int[] size) -> Tensor
Tensor unsqueeze(const Tensor & self, int64_t dim); // aten::unsqueeze(Tensor(a) self, int dim) -> Tensor(a)
Tensor & unsqueeze_(Tensor & self, int64_t dim); // aten::unsqueeze_(Tensor(a!) self, int dim) -> Tensor(a!)
Tensor var(const Tensor & self, bool unbiased); // aten::var(Tensor self, bool unbiased=True) -> Tensor
Tensor var(const Tensor & self, IntArrayRef dim, bool unbiased, bool keepdim); // aten::var(Tensor self, int[1] dim, bool unbiased=True, bool keepdim=False) -> Tensor
Tensor & var_out(Tensor & out, const Tensor & self, IntArrayRef dim, bool unbiased, bool keepdim); // aten::var(Tensor self, int[1] dim, bool unbiased=True, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
std::tuple<Tensor,Tensor> var_mean(const Tensor & self, bool unbiased); // aten::var_mean(Tensor self, bool unbiased=True) -> (Tensor, Tensor)
std::tuple<Tensor,Tensor> var_mean(const Tensor & self, IntArrayRef dim, bool unbiased, bool keepdim); // aten::var_mean(Tensor self, int[1] dim, bool unbiased=True, bool keepdim=False) -> (Tensor, Tensor)
Tensor view_as(const Tensor & self, const Tensor & other); // aten::view_as(Tensor self, Tensor other) -> Tensor
Tensor where(const Tensor & condition, const Tensor & self, const Tensor & other); // aten::where(Tensor condition, Tensor self, Tensor other) -> Tensor
std::vector<Tensor> where(const Tensor & condition); // aten::where(Tensor condition) -> Tensor[]
Tensor _s_where(const Tensor & condition, const Tensor & self, const Tensor & other); // aten::_s_where(Tensor condition, Tensor self, Tensor other) -> Tensor
Tensor norm_except_dim(const Tensor & v, int64_t pow, int64_t dim); // aten::norm_except_dim(Tensor v, int pow=2, int dim=0) -> Tensor
Tensor _weight_norm(const Tensor & v, const Tensor & g, int64_t dim); // aten::_weight_norm(Tensor v, Tensor g, int dim=0) -> Tensor
std::tuple<Tensor,Tensor> _weight_norm_cuda_interface(const Tensor & v, const Tensor & g, int64_t dim); // aten::_weight_norm_cuda_interface(Tensor v, Tensor g, int dim=0) -> (Tensor, Tensor)
std::tuple<Tensor,Tensor> _weight_norm_cuda_interface_backward(const Tensor & grad_w, const Tensor & saved_v, const Tensor & saved_g, const Tensor & saved_norms, int64_t dim); // aten::_weight_norm_cuda_interface_backward(Tensor grad_w, Tensor saved_v, Tensor saved_g, Tensor saved_norms, int dim) -> (Tensor, Tensor)
std::tuple<Tensor,Tensor> _weight_norm_differentiable_backward(const Tensor & grad_w, const Tensor & saved_v, const Tensor & saved_g, const Tensor & saved_norms, int64_t dim); // aten::_weight_norm_differentiable_backward(Tensor grad_w, Tensor saved_v, Tensor saved_g, Tensor saved_norms, int dim) -> (Tensor, Tensor)
Tensor zeros(IntArrayRef size, const TensorOptions & options); // aten::zeros(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
Tensor & zeros_out(Tensor & out, IntArrayRef size); // aten::zeros(int[] size, *, Tensor(a!) out) -> Tensor(a!)
Tensor zeros_like(const Tensor & self); // aten::zeros_like(Tensor self) -> Tensor
Tensor zeros_like(const Tensor & self, const TensorOptions & options); // aten::zeros_like(Tensor self, *, ScalarType dtype, Layout layout, Device device, bool pin_memory=False) -> Tensor
Tensor _standard_gamma_grad(const Tensor & self, const Tensor & output); // aten::_standard_gamma_grad(Tensor self, Tensor output) -> Tensor
Tensor _standard_gamma(const Tensor & self, Generator * generator); // aten::_standard_gamma(Tensor self, Generator? generator=None) -> Tensor
Tensor _dirichlet_grad(const Tensor & x, const Tensor & alpha, const Tensor & total); // aten::_dirichlet_grad(Tensor x, Tensor alpha, Tensor total) -> Tensor
Tensor _sample_dirichlet(const Tensor & self, Generator * generator); // aten::_sample_dirichlet(Tensor self, Generator? generator=None) -> Tensor
Tensor poisson(const Tensor & self, Generator * generator); // aten::poisson(Tensor self, Generator? generator=None) -> Tensor
Tensor native_norm(const Tensor & self, Scalar p); // aten::native_norm(Tensor self, Scalar p=2) -> Tensor
Tensor _sparse_sum(const Tensor & self); // aten::_sparse_sum(Tensor self) -> Tensor
Tensor _sparse_sum(const Tensor & self, ScalarType dtype); // aten::_sparse_sum(Tensor self, *, ScalarType dtype) -> Tensor
Tensor _sparse_sum(const Tensor & self, IntArrayRef dim); // aten::_sparse_sum(Tensor self, int[1] dim) -> Tensor
Tensor _sparse_sum(const Tensor & self, IntArrayRef dim, ScalarType dtype); // aten::_sparse_sum(Tensor self, int[1] dim, *, ScalarType dtype) -> Tensor
Tensor _sparse_sum_backward(const Tensor & grad, const Tensor & self, IntArrayRef dim); // aten::_sparse_sum_backward(Tensor grad, Tensor self, int[] dim) -> Tensor
Tensor norm(const Tensor & self, c10::optional<Scalar> p, ScalarType dtype); // aten::norm(Tensor self, Scalar? p, *, ScalarType dtype) -> Tensor
Tensor norm(const Tensor & self, Scalar p); // aten::norm(Tensor self, Scalar p=2) -> Tensor
Tensor norm(const Tensor & self, c10::optional<Scalar> p, IntArrayRef dim, bool keepdim, ScalarType dtype); // aten::norm(Tensor self, Scalar? p, int[1] dim, bool keepdim, *, ScalarType dtype) -> Tensor
Tensor norm(const Tensor & self, c10::optional<Scalar> p, IntArrayRef dim, bool keepdim); // aten::norm(Tensor self, Scalar? p, int[1] dim, bool keepdim=False) -> Tensor
Tensor & norm_out(Tensor & out, const Tensor & self, c10::optional<Scalar> p, IntArrayRef dim, bool keepdim, ScalarType dtype); // aten::norm(Tensor self, Scalar? p, int[1] dim, bool keepdim, *, ScalarType dtype, Tensor(a!) out) -> Tensor(a!)
Tensor & norm_out(Tensor & out, const Tensor & self, c10::optional<Scalar> p, IntArrayRef dim, bool keepdim); // aten::norm(Tensor self, Scalar? p, int[1] dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
Tensor frobenius_norm(const Tensor & self); // aten::frobenius_norm(Tensor self) -> Tensor
Tensor frobenius_norm(const Tensor & self, IntArrayRef dim, bool keepdim); // aten::frobenius_norm(Tensor self, int[1] dim, bool keepdim=False) -> Tensor
Tensor & frobenius_norm_out(Tensor & out, const Tensor & self, IntArrayRef dim, bool keepdim); // aten::frobenius_norm(Tensor self, int[1] dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
Tensor nuclear_norm(const Tensor & self, bool keepdim); // aten::nuclear_norm(Tensor self, bool keepdim=False) -> Tensor
Tensor & nuclear_norm_out(Tensor & out, const Tensor & self, bool keepdim); // aten::nuclear_norm(Tensor self, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
Tensor nuclear_norm(const Tensor & self, IntArrayRef dim, bool keepdim); // aten::nuclear_norm(Tensor self, int[2] dim, bool keepdim=False) -> Tensor
Tensor & nuclear_norm_out(Tensor & out, const Tensor & self, IntArrayRef dim, bool keepdim); // aten::nuclear_norm(Tensor self, int[2] dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
Tensor clone(const Tensor & self); // aten::clone(Tensor self) -> Tensor
Tensor & resize_as_(Tensor & self, const Tensor & the_template); // aten::resize_as_(Tensor(a!) self, Tensor the_template) -> Tensor(a!)
Tensor & pow_out(Tensor & out, const Tensor & self, Scalar exponent); // aten::pow(Tensor self, Scalar exponent, *, Tensor(a!) out) -> Tensor(a!)
Tensor pow(const Tensor & self, Scalar exponent); // aten::pow(Tensor self, Scalar exponent) -> Tensor
Tensor & zero_(Tensor & self); // aten::zero_(Tensor(a!) self) -> Tensor(a!)
Tensor & sub_out(Tensor & out, const Tensor & self, const Tensor & other, Scalar alpha); // aten::sub(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
Tensor sub(const Tensor & self, const Tensor & other, Scalar alpha); // aten::sub(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
Tensor & sub_(Tensor & self, const Tensor & other, Scalar alpha); // aten::sub_(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)
Tensor sub(const Tensor & self, Scalar other, Scalar alpha); // aten::sub(Tensor self, Scalar other, Scalar alpha=1) -> Tensor
Tensor & sub_(Tensor & self, Scalar other, Scalar alpha); // aten::sub_(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)
Tensor rsub(const Tensor & self, const Tensor & other, Scalar alpha); // aten::rsub(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
Tensor rsub(const Tensor & self, Scalar other, Scalar alpha); // aten::rsub(Tensor self, Scalar other, Scalar alpha=1) -> Tensor
Tensor & s_native_addmm_out(Tensor & out, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha); // aten::s_native_addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
Tensor s_native_addmm(const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha); // aten::s_native_addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
Tensor & s_native_addmm_(Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha); // aten::s_native_addmm_(Tensor(a!) self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)
Tensor _sparse_addmm(const Tensor & self, const Tensor & sparse, const Tensor & dense, Scalar beta, Scalar alpha); // aten::_sparse_addmm(Tensor self, Tensor sparse, Tensor dense, *, Scalar beta=1, Scalar alpha=1) -> Tensor
Tensor & addmm_out(Tensor & out, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha); // aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
Tensor addmm(const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha); // aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
Tensor & addmm_(Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha); // aten::addmm_(Tensor(a!) self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)
Tensor sparse_coo_tensor(IntArrayRef size, const TensorOptions & options); // aten::sparse_coo_tensor(int[] size, *, ScalarType dtype, Layout layout, Device device, bool pin_memory=False) -> Tensor
Tensor sparse_coo_tensor(const Tensor & indices, const Tensor & values, const TensorOptions & options); // aten::sparse_coo_tensor(Tensor indices, Tensor values, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
Tensor sparse_coo_tensor(const Tensor & indices, const Tensor & values, IntArrayRef size, const TensorOptions & options); // aten::sparse_coo_tensor(Tensor indices, Tensor values, int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
Tensor _sparse_coo_tensor_unsafe(const Tensor & indices, const Tensor & values, IntArrayRef size, const TensorOptions & options); // aten::_sparse_coo_tensor_unsafe(Tensor indices, Tensor values, int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
Tensor _sparse_coo_tensor_with_dims(int64_t sparse_dim, int64_t dense_dim, IntArrayRef size, const TensorOptions & options); // aten::_sparse_coo_tensor_with_dims(int sparse_dim, int dense_dim, int[] size, *, ScalarType dtype, Layout layout, Device device, bool pin_memory=False) -> Tensor
Tensor _sparse_coo_tensor_with_dims_and_tensors(int64_t sparse_dim, int64_t dense_dim, IntArrayRef size, const Tensor & indices, const Tensor & values, const TensorOptions & options); // aten::_sparse_coo_tensor_with_dims_and_tensors(int sparse_dim, int dense_dim, int[] size, Tensor indices, Tensor values, *, ScalarType dtype, Layout layout, Device device, bool pin_memory=False) -> Tensor
Tensor & sparse_resize_(Tensor & self, IntArrayRef size, int64_t sparse_dim, int64_t dense_dim); // aten::sparse_resize_(Tensor(a!) self, int[] size, int sparse_dim, int dense_dim) -> Tensor(a!)
Tensor & sparse_resize_and_clear_(Tensor & self, IntArrayRef size, int64_t sparse_dim, int64_t dense_dim); // aten::sparse_resize_and_clear_(Tensor(a!) self, int[] size, int sparse_dim, int dense_dim) -> Tensor(a!)
Tensor sparse_mask(const Tensor & self, const Tensor & mask); // aten::sparse_mask(Tensor self, Tensor mask) -> Tensor
Tensor to_dense(const Tensor & self); // aten::to_dense(Tensor self) -> Tensor
Tensor to_dense_backward(const Tensor & grad, const Tensor & input); // aten::to_dense_backward(Tensor grad, Tensor input) -> Tensor
int64_t sparse_dim(const Tensor & self); // aten::sparse_dim(Tensor self) -> int
int64_t _dimI(const Tensor & self); // aten::_dimI(Tensor self) -> int
int64_t dense_dim(const Tensor & self); // aten::dense_dim(Tensor self) -> int
int64_t _dimV(const Tensor & self); // aten::_dimV(Tensor self) -> int
int64_t _nnz(const Tensor & self); // aten::_nnz(Tensor self) -> int
Tensor coalesce(const Tensor & self); // aten::coalesce(Tensor self) -> Tensor
bool is_coalesced(const Tensor & self); // aten::is_coalesced(Tensor self) -> bool
Tensor _indices(const Tensor & self); // aten::_indices(Tensor(a) self) -> Tensor(a)
Tensor _values(const Tensor & self); // aten::_values(Tensor(a) self) -> Tensor(a)
Tensor & _coalesced_(Tensor & self, bool coalesced); // aten::_coalesced_(Tensor(a!) self, bool coalesced) -> Tensor(a!)
Tensor indices(const Tensor & self); // aten::indices(Tensor(a) self) -> Tensor(a)
Tensor values(const Tensor & self); // aten::values(Tensor(a) self) -> Tensor(a)
Tensor & hspmm_out(Tensor & out, const Tensor & mat1, const Tensor & mat2); // aten::hspmm(Tensor mat1, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)
Tensor hspmm(const Tensor & mat1, const Tensor & mat2); // aten::hspmm(Tensor mat1, Tensor mat2) -> Tensor
Tensor & copy_sparse_to_sparse_(Tensor & self, const Tensor & src, bool non_blocking); // aten::copy_sparse_to_sparse_(Tensor(a!) self, Tensor src, bool non_blocking=False) -> Tensor(a!)
int64_t numel(const Tensor & self); // aten::numel(Tensor self) -> int
std::vector<Tensor> unbind(const Tensor & self, int64_t dim); // aten::unbind(Tensor(a) self, int dim=0) -> Tensor(a)[]
Tensor to_sparse(const Tensor & self, int64_t sparse_dim); // aten::to_sparse(Tensor self, int sparse_dim) -> Tensor
Tensor to_sparse(const Tensor & self); // aten::to_sparse(Tensor self) -> Tensor
Tensor to_mkldnn(const Tensor & self); // aten::to_mkldnn(Tensor self) -> Tensor
Tensor mkldnn_reorder_conv2d_weight(const Tensor & self, IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups); // aten::mkldnn_reorder_conv2d_weight(Tensor self, int[2] padding=0, int[2] stride=1, int[2] dilation=1, int groups=1) -> Tensor
Tensor to_mkldnn_backward(const Tensor & grad, const Tensor & input); // aten::to_mkldnn_backward(Tensor grad, Tensor input) -> Tensor
Tensor quantize_linear(const Tensor & self, double scale, int64_t zero_point, ScalarType dtype); // aten::quantize_linear(Tensor self, float scale, int zero_point, ScalarType dtype) -> Tensor
Tensor quantize_linear_per_channel(const Tensor & self, const Tensor & scales, const Tensor & zero_points, IntArrayRef axis, ScalarType dtype); // aten::quantize_linear_per_channel(Tensor self, Tensor scales, Tensor zero_points, int[] axis, ScalarType dtype) -> Tensor
Tensor dequantize(const Tensor & self); // aten::dequantize(Tensor self) -> Tensor
Tensor _dequantize_linear(const Tensor & self, double scale, int64_t zero_point, ScalarType dtype); // aten::_dequantize_linear(Tensor self, float scale, int zero_point, ScalarType dtype) -> Tensor
double q_scale(const Tensor & self); // aten::q_scale(Tensor self) -> float
int64_t q_zero_point(const Tensor & self); // aten::q_zero_point(Tensor self) -> int
Tensor int_repr(const Tensor & self); // aten::int_repr(Tensor self) -> Tensor
Tensor _per_tensor_affine_qtensor(const Tensor & self, double scale, int64_t zero_point); // aten::_per_tensor_affine_qtensor(Tensor self, float scale, int zero_point) -> Tensor
QScheme qscheme(const Tensor & self); // aten::qscheme(Tensor self) -> QScheme
Tensor fake_quantize_per_tensor_affine(const Tensor & self, double scale, int64_t zero_point, int64_t quant_min, int64_t quant_max); // aten::fake_quantize_per_tensor_affine(Tensor self, float scale, int zero_point, int quant_min, int quant_max) -> Tensor
Tensor fake_quantize_per_tensor_affine_backward(const Tensor & grad, const Tensor & self, double scale, int64_t zero_point, int64_t quant_min, int64_t quant_max); // aten::fake_quantize_per_tensor_affine_backward(Tensor grad, Tensor self, float scale, int zero_point, int quant_min, int quant_max) -> Tensor
Tensor to(const Tensor & self, const TensorOptions & options, bool non_blocking, bool copy); // aten::to(Tensor self, *, ScalarType dtype, Layout layout, Device device, bool pin_memory=False, bool non_blocking=False, bool copy=False) -> Tensor
Tensor to(const Tensor & self, Device device, ScalarType dtype, bool non_blocking, bool copy); // aten::to(Tensor self, Device device, ScalarType dtype, bool non_blocking=False, bool copy=False) -> Tensor
Tensor to(const Tensor & self, ScalarType dtype, bool non_blocking, bool copy); // aten::to(Tensor self, ScalarType dtype, bool non_blocking=False, bool copy=False) -> Tensor
Tensor to(const Tensor & self, const Tensor & other, bool non_blocking, bool copy); // aten::to(Tensor self, Tensor other, bool non_blocking=False, bool copy=False) -> Tensor
std::vector<Tensor> meshgrid(TensorList tensors); // aten::meshgrid(Tensor[] tensors) -> Tensor[]
Tensor cartesian_prod(TensorList tensors); // aten::cartesian_prod(Tensor[] tensors) -> Tensor
Tensor combinations(const Tensor & self, int64_t r, bool with_replacement); // aten::combinations(Tensor self, int r=2, bool with_replacement=False) -> Tensor
Scalar item(const Tensor & self); // aten::item(Tensor self) -> Scalar
Scalar _local_scalar_dense(const Tensor & self); // aten::_local_scalar_dense(Tensor self) -> Scalar
std::tuple<Tensor,Tensor,Tensor> _thnn_fused_lstm_cell(const Tensor & input_gates, const Tensor & hidden_gates, const Tensor & cx, const Tensor & input_bias, const Tensor & hidden_bias); // aten::_thnn_fused_lstm_cell(Tensor input_gates, Tensor hidden_gates, Tensor cx, Tensor? input_bias=None, Tensor? hidden_bias=None) -> (Tensor, Tensor, Tensor)
std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor> _thnn_fused_lstm_cell_backward(const Tensor & grad_hy, const Tensor & grad_cy, const Tensor & cx, const Tensor & cy, const Tensor & workspace, bool has_bias); // aten::_thnn_fused_lstm_cell_backward(Tensor? grad_hy, Tensor? grad_cy, Tensor cx, Tensor cy, Tensor workspace, bool has_bias) -> (Tensor, Tensor, Tensor, Tensor, Tensor)
std::tuple<Tensor,Tensor> _thnn_fused_gru_cell(const Tensor & input_gates, const Tensor & hidden_gates, const Tensor & hx, const Tensor & input_bias, const Tensor & hidden_bias); // aten::_thnn_fused_gru_cell(Tensor input_gates, Tensor hidden_gates, Tensor hx, Tensor? input_bias=None, Tensor? hidden_bias=None) -> (Tensor, Tensor)
std::tuple<Tensor,Tensor,Tensor,Tensor,Tensor> _thnn_fused_gru_cell_backward(const Tensor & grad_hy, const Tensor & workspace, bool has_bias); // aten::_thnn_fused_gru_cell_backward(Tensor grad_hy, Tensor workspace, bool has_bias) -> (Tensor, Tensor, Tensor, Tensor, Tensor)
std::tuple<Tensor,Tensor,Tensor> lstm(const Tensor & input, TensorList hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first); // aten::lstm(Tensor input, Tensor[] hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor, Tensor)
std::tuple<Tensor,Tensor,Tensor> lstm(const Tensor & data, const Tensor & batch_sizes, TensorList hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional); // aten::lstm(Tensor data, Tensor batch_sizes, Tensor[] hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional) -> (Tensor, Tensor, Tensor)
std::tuple<Tensor,Tensor> gru(const Tensor & input, const Tensor & hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first); // aten::gru(Tensor input, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor)
std::tuple<Tensor,Tensor> gru(const Tensor & data, const Tensor & batch_sizes, const Tensor & hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional); // aten::gru(Tensor data, Tensor batch_sizes, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional) -> (Tensor, Tensor)
std::tuple<Tensor,Tensor> rnn_tanh(const Tensor & input, const Tensor & hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first); // aten::rnn_tanh(Tensor input, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor)
std::tuple<Tensor,Tensor> rnn_tanh(const Tensor & data, const Tensor & batch_sizes, const Tensor & hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional); // aten::rnn_tanh(Tensor data, Tensor batch_sizes, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional) -> (Tensor, Tensor)
std::tuple<Tensor,Tensor> rnn_relu(const Tensor & input, const Tensor & hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first); // aten::rnn_relu(Tensor input, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor)
std::tuple<Tensor,Tensor> rnn_relu(const Tensor & data, const Tensor & batch_sizes, const Tensor & hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional); // aten::rnn_relu(Tensor data, Tensor batch_sizes, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional) -> (Tensor, Tensor)
std::tuple<Tensor,Tensor> lstm_cell(const Tensor & input, TensorList hx, const Tensor & w_ih, const Tensor & w_hh, const Tensor & b_ih, const Tensor & b_hh); // aten::lstm_cell(Tensor input, Tensor[] hx, Tensor w_ih, Tensor w_hh, Tensor? b_ih=None, Tensor? b_hh=None) -> (Tensor, Tensor)
Tensor gru_cell(const Tensor & input, const Tensor & hx, const Tensor & w_ih, const Tensor & w_hh, const Tensor & b_ih, const Tensor & b_hh); // aten::gru_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor? b_ih=None, Tensor? b_hh=None) -> Tensor
Tensor rnn_tanh_cell(const Tensor & input, const Tensor & hx, const Tensor & w_ih, const Tensor & w_hh, const Tensor & b_ih, const Tensor & b_hh); // aten::rnn_tanh_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor? b_ih=None, Tensor? b_hh=None) -> Tensor
Tensor rnn_relu_cell(const Tensor & input, const Tensor & hx, const Tensor & w_ih, const Tensor & w_hh, const Tensor & b_ih, const Tensor & b_hh); // aten::rnn_relu_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor? b_ih=None, Tensor? b_hh=None) -> Tensor
std::tuple<Tensor,Tensor,Tensor> quantized_lstm(const Tensor & input, TensorList hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first, c10::optional<ScalarType> dtype); // aten::quantized_lstm(Tensor input, Tensor[] hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first, *, ScalarType? dtype=None) -> (Tensor, Tensor, Tensor)
std::tuple<Tensor,Tensor> quantized_gru(const Tensor & input, const Tensor & hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional, bool batch_first); // aten::quantized_gru(Tensor input, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor)
std::tuple<Tensor,Tensor> quantized_gru(const Tensor & data, const Tensor & batch_sizes, const Tensor & hx, TensorList params, bool has_biases, int64_t num_layers, double dropout, bool train, bool bidirectional); // aten::quantized_gru(Tensor data, Tensor batch_sizes, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional) -> (Tensor, Tensor)
std::tuple<Tensor,Tensor> quantized_lstm_cell(const Tensor & input, TensorList hx, const Tensor & w_ih, const Tensor & w_hh, const Tensor & b_ih, const Tensor & b_hh, const Tensor & packed_ih, const Tensor & packed_hh, const Tensor & col_offsets_ih, const Tensor & col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh); // aten::quantized_lstm_cell(Tensor input, Tensor[] hx, Tensor w_ih, Tensor w_hh, Tensor b_ih, Tensor b_hh, Tensor packed_ih, Tensor packed_hh, Tensor col_offsets_ih, Tensor col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh) -> (Tensor, Tensor)
Tensor quantized_gru_cell(const Tensor & input, const Tensor & hx, const Tensor & w_ih, const Tensor & w_hh, const Tensor & b_ih, const Tensor & b_hh, const Tensor & packed_ih, const Tensor & packed_hh, const Tensor & col_offsets_ih, const Tensor & col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh); // aten::quantized_gru_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor b_ih, Tensor b_hh, Tensor packed_ih, Tensor packed_hh, Tensor col_offsets_ih, Tensor col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh) -> Tensor
Tensor quantized_rnn_relu_cell(const Tensor & input, const Tensor & hx, const Tensor & w_ih, const Tensor & w_hh, const Tensor & b_ih, const Tensor & b_hh, const Tensor & packed_ih, const Tensor & packed_hh, const Tensor & col_offsets_ih, const Tensor & col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh); // aten::quantized_rnn_relu_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor b_ih, Tensor b_hh, Tensor packed_ih, Tensor packed_hh, Tensor col_offsets_ih, Tensor col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh) -> Tensor
Tensor quantized_rnn_tanh_cell(const Tensor & input, const Tensor & hx, const Tensor & w_ih, const Tensor & w_hh, const Tensor & b_ih, const Tensor & b_hh, const Tensor & packed_ih, const Tensor & packed_hh, const Tensor & col_offsets_ih, const Tensor & col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh); // aten::quantized_rnn_tanh_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor b_ih, Tensor b_hh, Tensor packed_ih, Tensor packed_hh, Tensor col_offsets_ih, Tensor col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh) -> Tensor
std::tuple<Tensor,Tensor> _pack_padded_sequence(const Tensor & input, const Tensor & lengths, bool batch_first); // aten::_pack_padded_sequence(Tensor input, Tensor lengths, bool batch_first) -> (Tensor, Tensor)
Tensor _pack_padded_sequence_backward(const Tensor & grad, IntArrayRef input_size, const Tensor & batch_sizes, bool batch_first); // aten::_pack_padded_sequence_backward(Tensor grad, int[] input_size, Tensor batch_sizes, bool batch_first) -> Tensor
std::tuple<Tensor,Tensor> _pad_packed_sequence(const Tensor & data, const Tensor & batch_sizes, bool batch_first, Scalar padding_value, int64_t total_length); // aten::_pad_packed_sequence(Tensor data, Tensor batch_sizes, bool batch_first, Scalar padding_value, int total_length) -> (Tensor, Tensor)
Tensor & set_(Tensor & self, Storage source); // aten::set_(Tensor(a!) self, Storage source) -> Tensor(a!)
Tensor & set_(Tensor & self, Storage source, int64_t storage_offset, IntArrayRef size, IntArrayRef stride); // aten::set_(Tensor(a!) self, Storage source, int storage_offset, int[] size, int[] stride=[]) -> Tensor(a!)
Tensor & set_(Tensor & self, const Tensor & source); // aten::set_(Tensor(a!) self, Tensor source) -> Tensor(a!)
Tensor & set_(Tensor & self); // aten::set_(Tensor(a!) self) -> Tensor(a!)
Tensor & set_quantizer_(Tensor & self, ConstQuantizerPtr quantizer); // aten::set_quantizer_(Tensor(a!) self, ConstQuantizerPtr quantizer) -> Tensor(a!)
bool is_set_to(const Tensor & self, const Tensor & tensor); // aten::is_set_to(Tensor self, Tensor tensor) -> bool
Tensor & masked_fill_(Tensor & self, const Tensor & mask, Scalar value); // aten::masked_fill_(Tensor(a!) self, Tensor mask, Scalar value) -> Tensor(a!)
Tensor masked_fill(const Tensor & self, const Tensor & mask, Scalar value); // aten::masked_fill(Tensor self, Tensor mask, Scalar value) -> Tensor
Tensor & masked_fill_(Tensor & self, const Tensor & mask, const Tensor & value); // aten::masked_fill_(Tensor(a!) self, Tensor mask, Tensor value) -> Tensor(a!)
Tensor masked_fill(const Tensor & self, const Tensor & mask, const Tensor & value); // aten::masked_fill(Tensor self, Tensor mask, Tensor value) -> Tensor
Tensor & masked_scatter_(Tensor & self, const Tensor & mask, const Tensor & source); // aten::masked_scatter_(Tensor(a!) self, Tensor mask, Tensor source) -> Tensor(a!)
Tensor masked_scatter(const Tensor & self, const Tensor & mask, const Tensor & source); // aten::masked_scatter(Tensor self, Tensor mask, Tensor source) -> Tensor
Tensor view(const Tensor & self, IntArrayRef size); // aten::view(Tensor(a) self, int[] size) -> Tensor(a)
Tensor & put_(Tensor & self, const Tensor & index, const Tensor & source, bool accumulate); // aten::put_(Tensor(a!) self, Tensor index, Tensor source, bool accumulate=False) -> Tensor(a!)
Tensor & index_add_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source); // aten::index_add_(Tensor(a!) self, int dim, Tensor index, Tensor source) -> Tensor(a!)
Tensor index_add(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & source); // aten::index_add(Tensor self, int dim, Tensor index, Tensor source) -> Tensor
Tensor & index_fill_(Tensor & self, int64_t dim, const Tensor & index, Scalar value); // aten::index_fill_(Tensor(a!) self, int dim, Tensor index, Scalar value) -> Tensor(a!)
Tensor index_fill(const Tensor & self, int64_t dim, const Tensor & index, Scalar value); // aten::index_fill(Tensor self, int dim, Tensor index, Scalar value) -> Tensor
Tensor & index_fill_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & value); // aten::index_fill_(Tensor(a!) self, int dim, Tensor index, Tensor value) -> Tensor(a!)
Tensor index_fill(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & value); // aten::index_fill(Tensor self, int dim, Tensor index, Tensor value) -> Tensor
Tensor & scatter_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & src); // aten::scatter_(Tensor(a!) self, int dim, Tensor index, Tensor src) -> Tensor(a!)
Tensor scatter(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & src); // aten::scatter(Tensor self, int dim, Tensor index, Tensor src) -> Tensor
Tensor & scatter_(Tensor & self, int64_t dim, const Tensor & index, Scalar value); // aten::scatter_(Tensor(a!) self, int dim, Tensor index, Scalar value) -> Tensor(a!)
Tensor scatter(const Tensor & self, int64_t dim, const Tensor & index, Scalar value); // aten::scatter(Tensor self, int dim, Tensor index, Scalar value) -> Tensor
Tensor & scatter_add_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & src); // aten::scatter_add_(Tensor(a!) self, int dim, Tensor index, Tensor src) -> Tensor(a!)
Tensor scatter_add(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & src); // aten::scatter_add(Tensor self, int dim, Tensor index, Tensor src) -> Tensor
Tensor & lt_(Tensor & self, Scalar other); // aten::lt_(Tensor(a!) self, Scalar other) -> Tensor(a!)
Tensor & lt_(Tensor & self, const Tensor & other); // aten::lt_(Tensor(a!) self, Tensor other) -> Tensor(a!)
Tensor & gt_(Tensor & self, Scalar other); // aten::gt_(Tensor(a!) self, Scalar other) -> Tensor(a!)
Tensor & gt_(Tensor & self, const Tensor & other); // aten::gt_(Tensor(a!) self, Tensor other) -> Tensor(a!)
Tensor & le_(Tensor & self, Scalar other); // aten::le_(Tensor(a!) self, Scalar other) -> Tensor(a!)
Tensor & le_(Tensor & self, const Tensor & other); // aten::le_(Tensor(a!) self, Tensor other) -> Tensor(a!)
Tensor & ge_(Tensor & self, Scalar other); // aten::ge_(Tensor(a!) self, Scalar other) -> Tensor(a!)
Tensor & ge_(Tensor & self, const Tensor & other); // aten::ge_(Tensor(a!) self, Tensor other) -> Tensor(a!)
Tensor & eq_(Tensor & self, Scalar other); // aten::eq_(Tensor(a!) self, Scalar other) -> Tensor(a!)
Tensor & eq_(Tensor & self, const Tensor & other); // aten::eq_(Tensor(a!) self, Tensor other) -> Tensor(a!)
Tensor & ne_(Tensor & self, Scalar other); // aten::ne_(Tensor(a!) self, Scalar other) -> Tensor(a!)
Tensor & ne_(Tensor & self, const Tensor & other); // aten::ne_(Tensor(a!) self, Tensor other) -> Tensor(a!)
Tensor __and__(const Tensor & self, Scalar other); // aten::__and__(Tensor self, Scalar other) -> Tensor
Tensor __and__(const Tensor & self, const Tensor & other); // aten::__and__(Tensor self, Tensor other) -> Tensor
Tensor & __iand__(Tensor & self, Scalar other); // aten::__iand__(Tensor(a!) self, Scalar other) -> Tensor(a!)
Tensor & __iand__(Tensor & self, const Tensor & other); // aten::__iand__(Tensor(a!) self, Tensor other) -> Tensor(a!)
Tensor __or__(const Tensor & self, Scalar other); // aten::__or__(Tensor self, Scalar other) -> Tensor
Tensor __or__(const Tensor & self, const Tensor & other); // aten::__or__(Tensor self, Tensor other) -> Tensor
Tensor & __ior__(Tensor & self, Scalar other); // aten::__ior__(Tensor(a!) self, Scalar other) -> Tensor(a!)
Tensor & __ior__(Tensor & self, const Tensor & other); // aten::__ior__(Tensor(a!) self, Tensor other) -> Tensor(a!)
Tensor __xor__(const Tensor & self, Scalar other); // aten::__xor__(Tensor self, Scalar other) -> Tensor
Tensor __xor__(const Tensor & self, const Tensor & other); // aten::__xor__(Tensor self, Tensor other) -> Tensor
Tensor & __ixor__(Tensor & self, Scalar other); // aten::__ixor__(Tensor(a!) self, Scalar other) -> Tensor(a!)
Tensor & __ixor__(Tensor & self, const Tensor & other); // aten::__ixor__(Tensor(a!) self, Tensor other) -> Tensor(a!)
Tensor __lshift__(const Tensor & self, Scalar other); // aten::__lshift__(Tensor self, Scalar other) -> Tensor
Tensor __lshift__(const Tensor & self, const Tensor & other); // aten::__lshift__(Tensor self, Tensor other) -> Tensor
Tensor & __ilshift__(Tensor & self, Scalar other); // aten::__ilshift__(Tensor(a!) self, Scalar other) -> Tensor(a!)
Tensor & __ilshift__(Tensor & self, const Tensor & other); // aten::__ilshift__(Tensor(a!) self, Tensor other) -> Tensor(a!)
Tensor __rshift__(const Tensor & self, Scalar other); // aten::__rshift__(Tensor self, Scalar other) -> Tensor
Tensor __rshift__(const Tensor & self, const Tensor & other); // aten::__rshift__(Tensor self, Tensor other) -> Tensor
Tensor & __irshift__(Tensor & self, Scalar other); // aten::__irshift__(Tensor(a!) self, Scalar other) -> Tensor(a!)
Tensor & __irshift__(Tensor & self, const Tensor & other); // aten::__irshift__(Tensor(a!) self, Tensor other) -> Tensor(a!)
Tensor & lgamma_(Tensor & self); // aten::lgamma_(Tensor(a!) self) -> Tensor(a!)
Tensor & atan2_(Tensor & self, const Tensor & other); // aten::atan2_(Tensor(a!) self, Tensor other) -> Tensor(a!)
Tensor & tril_(Tensor & self, int64_t diagonal); // aten::tril_(Tensor(a!) self, int diagonal=0) -> Tensor(a!)
Tensor & triu_(Tensor & self, int64_t diagonal); // aten::triu_(Tensor(a!) self, int diagonal=0) -> Tensor(a!)
Tensor & digamma_(Tensor & self); // aten::digamma_(Tensor(a!) self) -> Tensor(a!)
Tensor & polygamma_(Tensor & self, int64_t n); // aten::polygamma_(Tensor(a!) self, int n) -> Tensor(a!)
Tensor & erfinv_(Tensor & self); // aten::erfinv_(Tensor(a!) self) -> Tensor(a!)
Tensor & renorm_(Tensor & self, Scalar p, int64_t dim, Scalar maxnorm); // aten::renorm_(Tensor(a!) self, Scalar p, int dim, Scalar maxnorm) -> Tensor(a!)
Tensor & pow_(Tensor & self, Scalar exponent); // aten::pow_(Tensor(a!) self, Scalar exponent) -> Tensor(a!)
Tensor & pow_(Tensor & self, const Tensor & exponent); // aten::pow_(Tensor(a!) self, Tensor exponent) -> Tensor(a!)
Tensor & lerp_(Tensor & self, const Tensor & end, Scalar weight); // aten::lerp_(Tensor(a!) self, Tensor end, Scalar weight) -> Tensor(a!)
Tensor & lerp_(Tensor & self, const Tensor & end, const Tensor & weight); // aten::lerp_(Tensor(a!) self, Tensor end, Tensor weight) -> Tensor(a!)
Tensor & sign_(Tensor & self); // aten::sign_(Tensor(a!) self) -> Tensor(a!)
Tensor & fmod_(Tensor & self, Scalar other); // aten::fmod_(Tensor(a!) self, Scalar other) -> Tensor(a!)
Tensor & fmod_(Tensor & self, const Tensor & other); // aten::fmod_(Tensor(a!) self, Tensor other) -> Tensor(a!)
Tensor & remainder_(Tensor & self, Scalar other); // aten::remainder_(Tensor(a!) self, Scalar other) -> Tensor(a!)
Tensor & remainder_(Tensor & self, const Tensor & other); // aten::remainder_(Tensor(a!) self, Tensor other) -> Tensor(a!)
Tensor & addbmm_(Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha); // aten::addbmm_(Tensor(a!) self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)
Tensor & addbmm_out(Tensor & out, const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha); // aten::addbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
Tensor addbmm(const Tensor & self, const Tensor & batch1, const Tensor & batch2, Scalar beta, Scalar alpha); // aten::addbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
Tensor & addcmul_(Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value); // aten::addcmul_(Tensor(a!) self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor(a!)
Tensor & addcdiv_(Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value); // aten::addcdiv_(Tensor(a!) self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor(a!)
Tensor & random_(Tensor & self, int64_t from, int64_t to, Generator * generator); // aten::random_(Tensor(a!) self, int from, int to, *, Generator? generator=None) -> Tensor(a!)
Tensor & random_(Tensor & self, int64_t to, Generator * generator); // aten::random_(Tensor(a!) self, int to, *, Generator? generator=None) -> Tensor(a!)
Tensor & random_(Tensor & self, Generator * generator); // aten::random_(Tensor(a!) self, *, Generator? generator=None) -> Tensor(a!)
Tensor & uniform_(Tensor & self, double from, double to, Generator * generator); // aten::uniform_(Tensor(a!) self, float from=0, float to=1, *, Generator? generator=None) -> Tensor(a!)
Tensor & normal_(Tensor & self, double mean, double std, Generator * generator); // aten::normal_(Tensor(a!) self, float mean=0, float std=1, *, Generator? generator=None) -> Tensor(a!)
Tensor & cauchy_(Tensor & self, double median, double sigma, Generator * generator); // aten::cauchy_(Tensor(a!) self, float median=0, float sigma=1, *, Generator? generator=None) -> Tensor(a!)
Tensor & log_normal_(Tensor & self, double mean, double std, Generator * generator); // aten::log_normal_(Tensor(a!) self, float mean=1, float std=2, *, Generator? generator=None) -> Tensor(a!)
Tensor & exponential_(Tensor & self, double lambd, Generator * generator); // aten::exponential_(Tensor(a!) self, float lambd=1, *, Generator? generator=None) -> Tensor(a!)
Tensor & geometric_(Tensor & self, double p, Generator * generator); // aten::geometric_(Tensor(a!) self, float p, *, Generator? generator=None) -> Tensor(a!)
Tensor & diag_out(Tensor & out, const Tensor & self, int64_t diagonal); // aten::diag(Tensor self, int diagonal=0, *, Tensor(a!) out) -> Tensor(a!)
Tensor diag(const Tensor & self, int64_t diagonal); // aten::diag(Tensor self, int diagonal=0) -> Tensor
Tensor & cross_out(Tensor & out, const Tensor & self, const Tensor & other, c10::optional<int64_t> dim); // aten::cross(Tensor self, Tensor other, int? dim=None, *, Tensor(a!) out) -> Tensor(a!)
Tensor cross(const Tensor & self, const Tensor & other, c10::optional<int64_t> dim); // aten::cross(Tensor self, Tensor other, int? dim=None) -> Tensor
Tensor & triu_out(Tensor & out, const Tensor & self, int64_t diagonal); // aten::triu(Tensor self, int diagonal=0, *, Tensor(a!) out) -> Tensor(a!)
Tensor triu(const Tensor & self, int64_t diagonal); // aten::triu(Tensor self, int diagonal=0) -> Tensor
Tensor & tril_out(Tensor & out, const Tensor & self, int64_t diagonal); // aten::tril(Tensor self, int diagonal=0, *, Tensor(a!) out) -> Tensor(a!)
Tensor tril(const Tensor & self, int64_t diagonal); // aten::tril(Tensor self, int diagonal=0) -> Tensor
Tensor tril_indices(int64_t row, int64_t col, int64_t offset, const TensorOptions & options); // aten::tril_indices(int row, int col, int offset=0, *, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
Tensor triu_indices(int64_t row, int64_t col, int64_t offset, const TensorOptions & options); // aten::triu_indices(int row, int col, int offset=0, *, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
Tensor trace(const Tensor & self); // aten::trace(Tensor self) -> Tensor
Tensor & ne_out(Tensor & out, const Tensor & self, Scalar other); // aten::ne(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
Tensor ne(const Tensor & self, Scalar other); // aten::ne(Tensor self, Scalar other) -> Tensor
Tensor & ne_out(Tensor & out, const Tensor & self, const Tensor & other); // aten::ne(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
Tensor ne(const Tensor & self, const Tensor & other); // aten::ne(Tensor self, Tensor other) -> Tensor
Tensor & eq_out(Tensor & out, const Tensor & self, Scalar other); // aten::eq(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
Tensor eq(const Tensor & self, Scalar other); // aten::eq(Tensor self, Scalar other) -> Tensor
Tensor & eq_out(Tensor & out, const Tensor & self, const Tensor & other); // aten::eq(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
Tensor eq(const Tensor & self, const Tensor & other); // aten::eq(Tensor self, Tensor other) -> Tensor
Tensor & ge_out(Tensor & out, const Tensor & self, Scalar other); // aten::ge(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
Tensor ge(const Tensor & self, Scalar other); // aten::ge(Tensor self, Scalar other) -> Tensor
Tensor & ge_out(Tensor & out, const Tensor & self, const Tensor & other); // aten::ge(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
Tensor ge(const Tensor & self, const Tensor & other); // aten::ge(Tensor self, Tensor other) -> Tensor
Tensor & le_out(Tensor & out, const Tensor & self, Scalar other); // aten::le(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
Tensor le(const Tensor & self, Scalar other); // aten::le(Tensor self, Scalar other) -> Tensor
Tensor & le_out(Tensor & out, const Tensor & self, const Tensor & other); // aten::le(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
Tensor le(const Tensor & self, const Tensor & other); // aten::le(Tensor self, Tensor other) -> Tensor
Tensor & gt_out(Tensor & out, const Tensor & self, Scalar other); // aten::gt(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
Tensor gt(const Tensor & self, Scalar other); // aten::gt(Tensor self, Scalar other) -> Tensor
Tensor & gt_out(Tensor & out, const Tensor & self, const Tensor & other); // aten::gt(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
Tensor gt(const Tensor & self, const Tensor & other); // aten::gt(Tensor self, Tensor other) -> Tensor
Tensor & lt_out(Tensor & out, const Tensor & self, Scalar other); // aten::lt(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
Tensor lt(const Tensor & self, Scalar other); // aten::lt(Tensor self, Scalar other) -> Tensor
Tensor & lt_out(Tensor & out, const Tensor & self, const Tensor & other); // aten::lt(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
Tensor lt(const Tensor & self, const Tensor & other); // aten::lt(Tensor self, Tensor other) -> Tensor
Tensor & take_out(Tensor & out, const Tensor & self, const Tensor & index); // aten::take(Tensor self, Tensor index, *, Tensor(a!) out) -> Tensor(a!)
Tensor take(const Tensor & self, const Tensor & index); // aten::take(Tensor self, Tensor index) -> Tensor
Tensor & index_select_out(Tensor & out, const Tensor & self, int64_t dim, const Tensor & index); // aten::index_select(Tensor self, int dim, Tensor index, *, Tensor(a!) out) -> Tensor(a!)
Tensor index_select(const Tensor & self, int64_t dim, const Tensor & index); // aten::index_select(Tensor self, int dim, Tensor index) -> Tensor
Tensor & masked_select_out(Tensor & out, const Tensor & self, const Tensor & mask); // aten::masked_select(Tensor self, Tensor mask, *, Tensor(a!) out) -> Tensor(a!)
Tensor masked_select(const Tensor & self, const Tensor & mask); // aten::masked_select(Tensor self, Tensor mask) -> Tensor
Tensor & nonzero_out(Tensor & out, const Tensor & self); // aten::nonzero(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
Tensor nonzero(const Tensor & self); // aten::nonzero(Tensor self) -> Tensor
std::vector<Tensor> nonzero_numpy(const Tensor & self); // aten::nonzero_numpy(Tensor self) -> Tensor[]
Tensor & gather_out(Tensor & out, const Tensor & self, int64_t dim, const Tensor & index, bool sparse_grad); // aten::gather(Tensor self, int dim, Tensor index, *, bool sparse_grad=False, Tensor(a!) out) -> Tensor(a!)
Tensor gather(const Tensor & self, int64_t dim, const Tensor & index, bool sparse_grad); // aten::gather(Tensor self, int dim, Tensor index, *, bool sparse_grad=False) -> Tensor
Tensor _gather_sparse_backward(const Tensor & self, int64_t dim, const Tensor & index, const Tensor & grad); // aten::_gather_sparse_backward(Tensor self, int dim, Tensor index, Tensor grad) -> Tensor
Tensor & addcmul_out(Tensor & out, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value); // aten::addcmul(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1, Tensor(a!) out) -> Tensor(a!)
Tensor addcmul(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value); // aten::addcmul(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor
Tensor & addcdiv_out(Tensor & out, const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value); // aten::addcdiv(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1, Tensor(a!) out) -> Tensor(a!)
Tensor addcdiv(const Tensor & self, const Tensor & tensor1, const Tensor & tensor2, Scalar value); // aten::addcdiv(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor
std::tuple<Tensor &,Tensor &> lstsq_out(Tensor & X, Tensor & qr, const Tensor & self, const Tensor & A); // aten::lstsq(Tensor self, Tensor A, *, Tensor(a!) X, Tensor(b!) qr) -> (Tensor(a!) solution, Tensor(b!) QR)
std::tuple<Tensor,Tensor> lstsq(const Tensor & self, const Tensor & A); // aten::lstsq(Tensor self, Tensor A) -> (Tensor solution, Tensor QR)
std::tuple<Tensor &,Tensor &> triangular_solve_out(Tensor & X, Tensor & M, const Tensor & self, const Tensor & A, bool upper, bool transpose, bool unitriangular); // aten::triangular_solve(Tensor self, Tensor A, bool upper=True, bool transpose=False, bool unitriangular=False, *, Tensor(a!) X, Tensor(b!) M) -> (Tensor(a!) solution, Tensor(b!) cloned_coefficient)
std::tuple<Tensor,Tensor> triangular_solve(const Tensor & self, const Tensor & A, bool upper, bool transpose, bool unitriangular); // aten::triangular_solve(Tensor self, Tensor A, bool upper=True, bool transpose=False, bool unitriangular=False) -> (Tensor solution, Tensor cloned_coefficient)
std::tuple<Tensor,Tensor> _triangular_solve_helper(const Tensor & self, const Tensor & A, bool upper, bool transpose, bool unitriangular); // aten::_triangular_solve_helper(Tensor self, Tensor A, bool upper, bool transpose, bool unitriangular) -> (Tensor, Tensor)
std::tuple<Tensor &,Tensor &> symeig_out(Tensor & e, Tensor & V, const Tensor & self, bool eigenvectors, bool upper); // aten::symeig(Tensor self, bool eigenvectors=False, bool upper=True, *, Tensor(a!) e, Tensor(b!) V) -> (Tensor(a!) eigenvalues, Tensor(b!) eigenvectors)
std::tuple<Tensor,Tensor> symeig(const Tensor & self, bool eigenvectors, bool upper); // aten::symeig(Tensor self, bool eigenvectors=False, bool upper=True) -> (Tensor eigenvalues, Tensor eigenvectors)
std::tuple<Tensor,Tensor> _symeig_helper(const Tensor & self, bool eigenvectors, bool upper); // aten::_symeig_helper(Tensor self, bool eigenvectors, bool upper) -> (Tensor, Tensor)
std::tuple<Tensor &,Tensor &> eig_out(Tensor & e, Tensor & v, const Tensor & self, bool eigenvectors); // aten::eig(Tensor self, bool eigenvectors=False, *, Tensor(a!) e, Tensor(b!) v) -> (Tensor(a!) eigenvalues, Tensor(b!) eigenvectors)
std::tuple<Tensor,Tensor> eig(const Tensor & self, bool eigenvectors); // aten::eig(Tensor self, bool eigenvectors=False) -> (Tensor eigenvalues, Tensor eigenvectors)
std::tuple<Tensor &,Tensor &,Tensor &> svd_out(Tensor & U, Tensor & S, Tensor & V, const Tensor & self, bool some, bool compute_uv); // aten::svd(Tensor self, bool some=True, bool compute_uv=True, *, Tensor(a!) U, Tensor(b!) S, Tensor(c!) V) -> (Tensor(a!) U, Tensor(b!) S, Tensor(c!) V)
std::tuple<Tensor,Tensor,Tensor> svd(const Tensor & self, bool some, bool compute_uv); // aten::svd(Tensor self, bool some=True, bool compute_uv=True) -> (Tensor U, Tensor S, Tensor V)
std::tuple<Tensor,Tensor,Tensor> _svd_helper(const Tensor & self, bool some, bool compute_uv); // aten::_svd_helper(Tensor self, bool some, bool compute_uv) -> (Tensor, Tensor, Tensor)
Tensor & cholesky_out(Tensor & out, const Tensor & self, bool upper); // aten::cholesky(Tensor self, bool upper=False, *, Tensor(a!) out) -> Tensor(a!)
Tensor cholesky(const Tensor & self, bool upper); // aten::cholesky(Tensor self, bool upper=False) -> Tensor
Tensor _cholesky_helper(const Tensor & self, bool upper); // aten::_cholesky_helper(Tensor self, bool upper) -> Tensor
Tensor & cholesky_solve_out(Tensor & out, const Tensor & self, const Tensor & input2, bool upper); // aten::cholesky_solve(Tensor self, Tensor input2, bool upper=False, *, Tensor(a!) out) -> Tensor(a!)
Tensor cholesky_solve(const Tensor & self, const Tensor & input2, bool upper); // aten::cholesky_solve(Tensor self, Tensor input2, bool upper=False) -> Tensor
Tensor _cholesky_solve_helper(const Tensor & self, const Tensor & A, bool upper); // aten::_cholesky_solve_helper(Tensor self, Tensor A, bool upper) -> Tensor
std::tuple<Tensor,Tensor> solve(const Tensor & self, const Tensor & A); // aten::solve(Tensor self, Tensor A) -> (Tensor solution, Tensor LU)
std::tuple<Tensor &,Tensor &> solve_out(Tensor & solution, Tensor & lu, const Tensor & self, const Tensor & A); // aten::solve(Tensor self, Tensor A, *, Tensor(a!) solution, Tensor(b!) lu) -> (Tensor(a!) solution, Tensor(b!) LU)
std::tuple<Tensor,Tensor> _solve_helper(const Tensor & self, const Tensor & A); // aten::_solve_helper(Tensor self, Tensor A) -> (Tensor, Tensor)
Tensor & cholesky_inverse_out(Tensor & out, const Tensor & self, bool upper); // aten::cholesky_inverse(Tensor self, bool upper=False, *, Tensor(a!) out) -> Tensor(a!)
Tensor cholesky_inverse(const Tensor & self, bool upper); // aten::cholesky_inverse(Tensor self, bool upper=False) -> Tensor
std::tuple<Tensor &,Tensor &> qr_out(Tensor & Q, Tensor & R, const Tensor & self, bool some); // aten::qr(Tensor self, bool some=True, *, Tensor(a!) Q, Tensor(b!) R) -> (Tensor(a!) Q, Tensor(b!) R)
std::tuple<Tensor,Tensor> qr(const Tensor & self, bool some); // aten::qr(Tensor self, bool some=True) -> (Tensor Q, Tensor R)
std::tuple<Tensor,Tensor> _qr_helper(const Tensor & self, bool some); // aten::_qr_helper(Tensor self, bool some) -> (Tensor, Tensor)
std::tuple<Tensor &,Tensor &> geqrf_out(Tensor & a, Tensor & tau, const Tensor & self); // aten::geqrf(Tensor self, *, Tensor(a!) a, Tensor(b!) tau) -> (Tensor(a!) a, Tensor(b!) tau)
std::tuple<Tensor,Tensor> geqrf(const Tensor & self); // aten::geqrf(Tensor self) -> (Tensor a, Tensor tau)
Tensor & orgqr_out(Tensor & out, const Tensor & self, const Tensor & input2); // aten::orgqr(Tensor self, Tensor input2, *, Tensor(a!) out) -> Tensor(a!)
Tensor orgqr(const Tensor & self, const Tensor & input2); // aten::orgqr(Tensor self, Tensor input2) -> Tensor
Tensor & ormqr_out(Tensor & out, const Tensor & self, const Tensor & input2, const Tensor & input3, bool left, bool transpose); // aten::ormqr(Tensor self, Tensor input2, Tensor input3, bool left=True, bool transpose=False, *, Tensor(a!) out) -> Tensor(a!)
Tensor ormqr(const Tensor & self, const Tensor & input2, const Tensor & input3, bool left, bool transpose); // aten::ormqr(Tensor self, Tensor input2, Tensor input3, bool left=True, bool transpose=False) -> Tensor
std::tuple<Tensor,Tensor,Tensor> _lu_with_info(const Tensor & self, bool pivot, bool check_errors); // aten::_lu_with_info(Tensor self, bool pivot=True, bool check_errors=True) -> (Tensor, Tensor, Tensor)
Tensor & lu_solve_out(Tensor & out, const Tensor & self, const Tensor & LU_data, const Tensor & LU_pivots); // aten::lu_solve(Tensor self, Tensor LU_data, Tensor LU_pivots, *, Tensor(a!) out) -> Tensor(a!)
Tensor lu_solve(const Tensor & self, const Tensor & LU_data, const Tensor & LU_pivots); // aten::lu_solve(Tensor self, Tensor LU_data, Tensor LU_pivots) -> Tensor
Tensor _lu_solve_helper(const Tensor & self, const Tensor & LU_data, const Tensor & LU_pivots); // aten::_lu_solve_helper(Tensor self, Tensor LU_data, Tensor LU_pivots) -> Tensor
Tensor & multinomial_out(Tensor & out, const Tensor & self, int64_t num_samples, bool replacement, Generator * generator); // aten::multinomial(Tensor self, int num_samples, bool replacement=False, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)
Tensor multinomial(const Tensor & self, int64_t num_samples, bool replacement, Generator * generator); // aten::multinomial(Tensor self, int num_samples, bool replacement=False, *, Generator? generator=None) -> Tensor
std::tuple<Tensor,Tensor> _multinomial_alias_setup(const Tensor & probs); // aten::_multinomial_alias_setup(Tensor probs) -> (Tensor, Tensor)
Tensor _multinomial_alias_draw(const Tensor & J, const Tensor & q, int64_t num_samples, Generator * generator); // aten::_multinomial_alias_draw(Tensor J, Tensor q, int num_samples, *, Generator? generator=None) -> Tensor
Tensor & lgamma_out(Tensor & out, const Tensor & self); // aten::lgamma(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
Tensor lgamma(const Tensor & self); // aten::lgamma(Tensor self) -> Tensor
Tensor & digamma_out(Tensor & out, const Tensor & self); // aten::digamma(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
Tensor digamma(const Tensor & self); // aten::digamma(Tensor self) -> Tensor
Tensor & polygamma_out(Tensor & out, int64_t n, const Tensor & self); // aten::polygamma(int n, Tensor self, *, Tensor(a!) out) -> Tensor(a!)
Tensor polygamma(int64_t n, const Tensor & self); // aten::polygamma(int n, Tensor self) -> Tensor
Tensor & erfinv_out(Tensor & out, const Tensor & self); // aten::erfinv(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
Tensor erfinv(const Tensor & self); // aten::erfinv(Tensor self) -> Tensor
Tensor dist(const Tensor & self, const Tensor & other, Scalar p); // aten::dist(Tensor self, Tensor other, Scalar p=2) -> Tensor
Tensor & atan2_out(Tensor & out, const Tensor & self, const Tensor & other); // aten::atan2(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
Tensor atan2(const Tensor & self, const Tensor & other); // aten::atan2(Tensor self, Tensor other) -> Tensor
Tensor & lerp_out(Tensor & out, const Tensor & self, const Tensor & end, Scalar weight); // aten::lerp(Tensor self, Tensor end, Scalar weight, *, Tensor(a!) out) -> Tensor(a!)
Tensor & lerp_out(Tensor & out, const Tensor & self, const Tensor & end, const Tensor & weight); // aten::lerp(Tensor self, Tensor end, Tensor weight, *, Tensor(a!) out) -> Tensor(a!)
Tensor lerp(const Tensor & self, const Tensor & end, Scalar weight); // aten::lerp(Tensor self, Tensor end, Scalar weight) -> Tensor
Tensor lerp(const Tensor & self, const Tensor & end, const Tensor & weight); // aten::lerp(Tensor self, Tensor end, Tensor weight) -> Tensor
Tensor & histc_out(Tensor & out, const Tensor & self, int64_t bins, Scalar min, Scalar max); // aten::histc(Tensor self, int bins=100, Scalar min=0, Scalar max=0, *, Tensor(a!) out) -> Tensor(a!)
Tensor histc(const Tensor & self, int64_t bins, Scalar min, Scalar max); // aten::histc(Tensor self, int bins=100, Scalar min=0, Scalar max=0) -> Tensor
Tensor & sign_out(Tensor & out, const Tensor & self); // aten::sign(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
Tensor sign(const Tensor & self); // aten::sign(Tensor self) -> Tensor
Tensor & fmod_out(Tensor & out, const Tensor & self, Scalar other); // aten::fmod(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
Tensor fmod(const Tensor & self, Scalar other); // aten::fmod(Tensor self, Scalar other) -> Tensor
Tensor & fmod_out(Tensor & out, const Tensor & self, const Tensor & other); // aten::fmod(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
Tensor fmod(const Tensor & self, const Tensor & other); // aten::fmod(Tensor self, Tensor other) -> Tensor
Tensor & remainder_out(Tensor & out, const Tensor & self, Scalar other); // aten::remainder(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
Tensor remainder(const Tensor & self, Scalar other); // aten::remainder(Tensor self, Scalar other) -> Tensor
Tensor & remainder_out(Tensor & out, const Tensor & self, const Tensor & other); // aten::remainder(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
Tensor remainder(const Tensor & self, const Tensor & other); // aten::remainder(Tensor self, Tensor other) -> Tensor
Tensor & min_out(Tensor & out, const Tensor & self, const Tensor & other); // aten::min(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
Tensor min(const Tensor & self, const Tensor & other); // aten::min(Tensor self, Tensor other) -> Tensor
Tensor min(const Tensor & self); // aten::min(Tensor self) -> Tensor
Tensor & max_out(Tensor & out, const Tensor & self, const Tensor & other); // aten::max(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
Tensor max(const Tensor & self, const Tensor & other); // aten::max(Tensor self, Tensor other) -> Tensor
Tensor max(const Tensor & self); // aten::max(Tensor self) -> Tensor
Tensor median(const Tensor & self); // aten::median(Tensor self) -> Tensor
std::tuple<Tensor &,Tensor &> sort_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool descending); // aten::sort(Tensor self, int dim=-1, bool descending=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
std::tuple<Tensor,Tensor> sort(const Tensor & self, int64_t dim, bool descending); // aten::sort(Tensor self, int dim=-1, bool descending=False) -> (Tensor values, Tensor indices)
Tensor argsort(const Tensor & self, int64_t dim, bool descending); // aten::argsort(Tensor self, int dim=-1, bool descending=False) -> Tensor
std::tuple<Tensor &,Tensor &> topk_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted); // aten::topk(Tensor self, int k, int dim=-1, bool largest=True, bool sorted=True, *, Tensor(a!) values, Tensor(b!) indices) ->(Tensor(a!) values, Tensor(b!) indices)
std::tuple<Tensor,Tensor> topk(const Tensor & self, int64_t k, int64_t dim, bool largest, bool sorted); // aten::topk(Tensor self, int k, int dim=-1, bool largest=True, bool sorted=True) -> (Tensor values, Tensor indices)
Tensor all(const Tensor & self); // aten::all(Tensor self) -> Tensor
Tensor any(const Tensor & self); // aten::any(Tensor self) -> Tensor
Tensor & renorm_out(Tensor & out, const Tensor & self, Scalar p, int64_t dim, Scalar maxnorm); // aten::renorm(Tensor self, Scalar p, int dim, Scalar maxnorm, *, Tensor(a!) out) -> Tensor(a!)
Tensor renorm(const Tensor & self, Scalar p, int64_t dim, Scalar maxnorm); // aten::renorm(Tensor self, Scalar p, int dim, Scalar maxnorm) -> Tensor
Tensor unfold(const Tensor & self, int64_t dimension, int64_t size, int64_t step); // aten::unfold(Tensor(a) self, int dimension, int size, int step) -> Tensor(a)
bool equal(const Tensor & self, const Tensor & other); // aten::equal(Tensor self, Tensor other) -> bool
Tensor & pow_out(Tensor & out, const Tensor & self, const Tensor & exponent); // aten::pow(Tensor self, Tensor exponent, *, Tensor(a!) out) -> Tensor(a!)
Tensor pow(const Tensor & self, const Tensor & exponent); // aten::pow(Tensor self, Tensor exponent) -> Tensor
Tensor & pow_out(Tensor & out, Scalar self, const Tensor & exponent); // aten::pow(Scalar self, Tensor exponent, *, Tensor(a!) out) -> Tensor(a!)
Tensor pow(Scalar self, const Tensor & exponent); // aten::pow(Scalar self, Tensor exponent) -> Tensor
Tensor & normal_out(Tensor & out, const Tensor & mean, double std, Generator * generator); // aten::normal(Tensor mean, float std=1, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)
Tensor normal(const Tensor & mean, double std, Generator * generator); // aten::normal(Tensor mean, float std=1, *, Generator? generator=None) -> Tensor
Tensor & normal_out(Tensor & out, double mean, const Tensor & std, Generator * generator); // aten::normal(float mean, Tensor std, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)
Tensor normal(double mean, const Tensor & std, Generator * generator); // aten::normal(float mean, Tensor std, *, Generator? generator=None) -> Tensor
Tensor & normal_out(Tensor & out, const Tensor & mean, const Tensor & std, Generator * generator); // aten::normal(Tensor mean, Tensor std, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)
Tensor normal(const Tensor & mean, const Tensor & std, Generator * generator); // aten::normal(Tensor mean, Tensor std, *, Generator? generator=None) -> Tensor
Tensor normal(double mean, double std, IntArrayRef size, Generator * generator, const TensorOptions & options); // aten::normal(float mean, float std, int[] size, *, Generator? generator=None, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
Tensor & normal_out(Tensor & out, double mean, double std, IntArrayRef size, Generator * generator); // aten::normal(float mean, float std, int[] size, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)
Tensor alias(const Tensor & self); // aten::alias(Tensor(a) self) -> Tensor(a)
Tensor _addr(const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha); // aten::_addr(Tensor self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
Tensor & _addr_(Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha); // aten::_addr_(Tensor(a!) self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)
Tensor & _addr_out(Tensor & out, const Tensor & self, const Tensor & vec1, const Tensor & vec2, Scalar beta, Scalar alpha); // aten::_addr(Tensor self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
Tensor & _index_copy_(Tensor & self, int64_t dim, const Tensor & index, const Tensor & source); // aten::_index_copy_(Tensor(a!) self, int dim, Tensor index, Tensor source) -> Tensor(a!)
Tensor _cumsum(const Tensor & self, int64_t dim); // aten::_cumsum(Tensor self, int dim) -> Tensor
Tensor & _cumsum_out(Tensor & out, const Tensor & self, int64_t dim); // aten::_cumsum(Tensor self, int dim, *, Tensor(a!) out) -> Tensor(a!)
Tensor _cumprod(const Tensor & self, int64_t dim); // aten::_cumprod(Tensor self, int dim) -> Tensor
Tensor & _cumprod_out(Tensor & out, const Tensor & self, int64_t dim); // aten::_cumprod(Tensor self, int dim, *, Tensor(a!) out) -> Tensor(a!)
Tensor _var(const Tensor & self, bool unbiased); // aten::_var(Tensor self, bool unbiased=True) -> Tensor
Tensor _std(const Tensor & self, bool unbiased); // aten::_std(Tensor self, bool unbiased=True) -> Tensor
Tensor & _addmm_out(Tensor & out, const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha); // aten::_addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
Tensor _addmm(const Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha); // aten::_addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
Tensor & _addmm_(Tensor & self, const Tensor & mat1, const Tensor & mat2, Scalar beta, Scalar alpha); // aten::_addmm_(Tensor(a!) self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)
Tensor _cat(TensorList tensors, int64_t dim); // aten::_cat(Tensor[] tensors, int dim=0) -> Tensor
Tensor & _cat_out(Tensor & out, TensorList tensors, int64_t dim); // aten::_cat(Tensor[] tensors, int dim=0, *, Tensor(a!) out) -> Tensor(a!)
std::tuple<Tensor,Tensor> _mode(const Tensor & self, int64_t dim, bool keepdim); // aten::_mode(Tensor self, int dim=-1, bool keepdim=False) -> (Tensor, Tensor)
std::tuple<Tensor &,Tensor &> _mode_out(Tensor & values, Tensor & indices, const Tensor & self, int64_t dim, bool keepdim); // aten::_mode(Tensor self, int dim=-1, bool keepdim=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!), Tensor(b!))
std::tuple<Tensor,Tensor> _max(const Tensor & self, int64_t dim, bool keepdim); // aten::_max(Tensor self, int dim, bool keepdim=False) -> (Tensor, Tensor)
std::tuple<Tensor &,Tensor &> _max_out(Tensor & max, Tensor & max_indices, const Tensor & self, int64_t dim, bool keepdim); // aten::_max(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) max, Tensor(b!) max_indices) -> (Tensor(a!), Tensor(b!))
std::tuple<Tensor,Tensor> _min(const Tensor & self, int64_t dim, bool keepdim); // aten::_min(Tensor self, int dim, bool keepdim=False) -> (Tensor, Tensor)
std::tuple<Tensor &,Tensor &> _min_out(Tensor & min, Tensor & min_indices, const Tensor & self, int64_t dim, bool keepdim); // aten::_min(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) min, Tensor(b!) min_indices) -> (Tensor(a!), Tensor(b!))
Tensor & binary_cross_entropy_out(Tensor & out, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction); // aten::binary_cross_entropy(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, *, Tensor(a!) out) -> Tensor(a!)
Tensor binary_cross_entropy(const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction); // aten::binary_cross_entropy(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean) -> Tensor
Tensor & binary_cross_entropy_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction); // aten::binary_cross_entropy_backward(Tensor grad_output, Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, *, Tensor(a!) grad_input) -> Tensor(a!)
Tensor binary_cross_entropy_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction); // aten::binary_cross_entropy_backward(Tensor grad_output, Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean) -> Tensor
Tensor & mse_loss_out(Tensor & out, const Tensor & self, const Tensor & target, int64_t reduction); // aten::mse_loss(Tensor self, Tensor target, int reduction=Mean, *, Tensor(a!) out) -> Tensor(a!)
Tensor mse_loss(const Tensor & self, const Tensor & target, int64_t reduction); // aten::mse_loss(Tensor self, Tensor target, int reduction=Mean) -> Tensor
Tensor & mse_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction); // aten::mse_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction, *, Tensor(a!) grad_input) -> Tensor(a!)
Tensor mse_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction); // aten::mse_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction) -> Tensor
Tensor & l1_loss_out(Tensor & out, const Tensor & self, const Tensor & target, int64_t reduction); // aten::l1_loss(Tensor self, Tensor target, int reduction=Mean, *, Tensor(a!) out) -> Tensor(a!)
Tensor l1_loss(const Tensor & self, const Tensor & target, int64_t reduction); // aten::l1_loss(Tensor self, Tensor target, int reduction=Mean) -> Tensor
Tensor & l1_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction); // aten::l1_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction, *, Tensor(a!) grad_input) -> Tensor(a!)
Tensor l1_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction); // aten::l1_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction) -> Tensor
Tensor & multi_margin_loss_out(Tensor & out, const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, int64_t reduction); // aten::multi_margin_loss(Tensor self, Tensor target, Scalar p=1, Scalar margin=1, Tensor? weight=None, int reduction=Mean, *, Tensor(a!) out) -> Tensor(a!)
Tensor multi_margin_loss(const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, int64_t reduction); // aten::multi_margin_loss(Tensor self, Tensor target, Scalar p=1, Scalar margin=1, Tensor? weight=None, int reduction=Mean) -> Tensor
Tensor & multi_margin_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, int64_t reduction); // aten::multi_margin_loss_backward(Tensor grad_output, Tensor self, Tensor target, Scalar p, Scalar margin, Tensor? weight=None, int reduction=Mean, *, Tensor(a!) grad_input) -> Tensor(a!)
Tensor multi_margin_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, Scalar p, Scalar margin, const Tensor & weight, int64_t reduction); // aten::multi_margin_loss_backward(Tensor grad_output, Tensor self, Tensor target, Scalar p, Scalar margin, Tensor? weight=None, int reduction=Mean) -> Tensor
Tensor & multilabel_margin_loss_out(Tensor & out, const Tensor & self, const Tensor & target, int64_t reduction); // aten::multilabel_margin_loss(Tensor self, Tensor target, int reduction=Mean, *, Tensor(a!) out) -> Tensor(a!)
Tensor multilabel_margin_loss(const Tensor & self, const Tensor & target, int64_t reduction); // aten::multilabel_margin_loss(Tensor self, Tensor target, int reduction=Mean) -> Tensor
std::tuple<Tensor &,Tensor &> multilabel_margin_loss_forward_out(Tensor & output, Tensor & is_target, const Tensor & self, const Tensor & target, int64_t reduction); // aten::multilabel_margin_loss_forward(Tensor self, Tensor target, int reduction, *, Tensor(a!) output, Tensor(b!) is_target) -> (Tensor(a!), Tensor(b!))
std::tuple<Tensor,Tensor> multilabel_margin_loss_forward(const Tensor & self, const Tensor & target, int64_t reduction); // aten::multilabel_margin_loss_forward(Tensor self, Tensor target, int reduction) -> (Tensor output, Tensor is_target)
Tensor & multilabel_margin_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction, const Tensor & is_target); // aten::multilabel_margin_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction, Tensor is_target, *, Tensor(a!) grad_input) -> Tensor(a!)
Tensor multilabel_margin_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction, const Tensor & is_target); // aten::multilabel_margin_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction, Tensor is_target) -> Tensor
Tensor & nll_loss_out(Tensor & out, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index); // aten::nll_loss(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, int ignore_index=-100, *, Tensor(a!) out) -> Tensor(a!)
Tensor nll_loss(const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index); // aten::nll_loss(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, int ignore_index=-100) -> Tensor
std::tuple<Tensor &,Tensor &> nll_loss_forward_out(Tensor & output, Tensor & total_weight, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index); // aten::nll_loss_forward(Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index, *, Tensor(a!) output, Tensor(b!) total_weight) -> (Tensor(a!), Tensor(b!))
std::tuple<Tensor,Tensor> nll_loss_forward(const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index); // aten::nll_loss_forward(Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index) -> (Tensor output, Tensor total_weight)
Tensor & nll_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index, const Tensor & total_weight); // aten::nll_loss_backward(Tensor grad_output, Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index, Tensor total_weight, *, Tensor(a!) grad_input) -> Tensor(a!)
Tensor nll_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index, const Tensor & total_weight); // aten::nll_loss_backward(Tensor grad_output, Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index, Tensor total_weight) -> Tensor
Tensor & nll_loss2d_out(Tensor & out, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index); // aten::nll_loss2d(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, int ignore_index=-100, *, Tensor(a!) out) -> Tensor(a!)
Tensor nll_loss2d(const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index); // aten::nll_loss2d(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, int ignore_index=-100) -> Tensor
std::tuple<Tensor &,Tensor &> nll_loss2d_forward_out(Tensor & output, Tensor & total_weight, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index); // aten::nll_loss2d_forward(Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index, *, Tensor(a!) output, Tensor(b!) total_weight) -> (Tensor(a!), Tensor(b!))
std::tuple<Tensor,Tensor> nll_loss2d_forward(const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index); // aten::nll_loss2d_forward(Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index) -> (Tensor output, Tensor total_weight)
Tensor & nll_loss2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index, const Tensor & total_weight); // aten::nll_loss2d_backward(Tensor grad_output, Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index, Tensor total_weight, *, Tensor(a!) grad_input) -> Tensor(a!)
Tensor nll_loss2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, const Tensor & weight, int64_t reduction, int64_t ignore_index, const Tensor & total_weight); // aten::nll_loss2d_backward(Tensor grad_output, Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index, Tensor total_weight) -> Tensor
Tensor & smooth_l1_loss_out(Tensor & out, const Tensor & self, const Tensor & target, int64_t reduction); // aten::smooth_l1_loss(Tensor self, Tensor target, int reduction=Mean, *, Tensor(a!) out) -> Tensor(a!)
Tensor smooth_l1_loss(const Tensor & self, const Tensor & target, int64_t reduction); // aten::smooth_l1_loss(Tensor self, Tensor target, int reduction=Mean) -> Tensor
Tensor & smooth_l1_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction); // aten::smooth_l1_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction, *, Tensor(a!) grad_input) -> Tensor(a!)
Tensor smooth_l1_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction); // aten::smooth_l1_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction) -> Tensor
Tensor & soft_margin_loss_out(Tensor & out, const Tensor & self, const Tensor & target, int64_t reduction); // aten::soft_margin_loss(Tensor self, Tensor target, int reduction=Mean, *, Tensor(a!) out) -> Tensor(a!)
Tensor soft_margin_loss(const Tensor & self, const Tensor & target, int64_t reduction); // aten::soft_margin_loss(Tensor self, Tensor target, int reduction=Mean) -> Tensor
Tensor & soft_margin_loss_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction); // aten::soft_margin_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction, *, Tensor(a!) grad_input) -> Tensor(a!)
Tensor soft_margin_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction); // aten::soft_margin_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction) -> Tensor
Tensor & elu_out(Tensor & out, const Tensor & self, Scalar alpha, Scalar scale, Scalar input_scale); // aten::elu(Tensor self, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1, *, Tensor(a!) out) -> Tensor(a!)
Tensor elu(const Tensor & self, Scalar alpha, Scalar scale, Scalar input_scale); // aten::elu(Tensor self, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1) -> Tensor
Tensor & elu_backward_out(Tensor & grad_input, const Tensor & grad_output, Scalar alpha, Scalar scale, Scalar input_scale, const Tensor & output); // aten::elu_backward(Tensor grad_output, Scalar alpha, Scalar scale, Scalar input_scale, Tensor output, *, Tensor(a!) grad_input) -> Tensor(a!)
Tensor elu_backward(const Tensor & grad_output, Scalar alpha, Scalar scale, Scalar input_scale, const Tensor & output); // aten::elu_backward(Tensor grad_output, Scalar alpha, Scalar scale, Scalar input_scale, Tensor output) -> Tensor
Tensor & elu_(Tensor & self, Scalar alpha, Scalar scale, Scalar input_scale); // aten::elu_(Tensor(a!) self, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1) -> Tensor(a!)
Tensor & glu_out(Tensor & out, const Tensor & self, int64_t dim); // aten::glu(Tensor self, int dim=-1, *, Tensor(a!) out) -> Tensor(a!)
Tensor glu(const Tensor & self, int64_t dim); // aten::glu(Tensor self, int dim=-1) -> Tensor
Tensor & glu_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, int64_t dim); // aten::glu_backward(Tensor grad_output, Tensor self, int dim, *, Tensor(a!) grad_input) -> Tensor(a!)
Tensor glu_backward(const Tensor & grad_output, const Tensor & self, int64_t dim); // aten::glu_backward(Tensor grad_output, Tensor self, int dim) -> Tensor
Tensor & hardtanh_out(Tensor & out, const Tensor & self, Scalar min_val, Scalar max_val); // aten::hardtanh(Tensor self, Scalar min_val=-1, Scalar max_val=1, *, Tensor(a!) out) -> Tensor(a!)
Tensor hardtanh(const Tensor & self, Scalar min_val, Scalar max_val); // aten::hardtanh(Tensor self, Scalar min_val=-1, Scalar max_val=1) -> Tensor
Tensor & hardtanh_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar min_val, Scalar max_val); // aten::hardtanh_backward(Tensor grad_output, Tensor self, Scalar min_val, Scalar max_val, *, Tensor(a!) grad_input) -> Tensor(a!)
Tensor hardtanh_backward(const Tensor & grad_output, const Tensor & self, Scalar min_val, Scalar max_val); // aten::hardtanh_backward(Tensor grad_output, Tensor self, Scalar min_val, Scalar max_val) -> Tensor
Tensor & hardtanh_(Tensor & self, Scalar min_val, Scalar max_val); // aten::hardtanh_(Tensor(a!) self, Scalar min_val=-1, Scalar max_val=1) -> Tensor(a!)
Tensor & leaky_relu_out(Tensor & out, const Tensor & self, Scalar negative_slope); // aten::leaky_relu(Tensor self, Scalar negative_slope=0.01, *, Tensor(a!) out) -> Tensor(a!)
Tensor leaky_relu(const Tensor & self, Scalar negative_slope); // aten::leaky_relu(Tensor self, Scalar negative_slope=0.01) -> Tensor
Tensor & leaky_relu_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar negative_slope); // aten::leaky_relu_backward(Tensor grad_output, Tensor self, Scalar negative_slope, *, Tensor(a!) grad_input) -> Tensor(a!)
Tensor leaky_relu_backward(const Tensor & grad_output, const Tensor & self, Scalar negative_slope); // aten::leaky_relu_backward(Tensor grad_output, Tensor self, Scalar negative_slope) -> Tensor
Tensor & leaky_relu_(Tensor & self, Scalar negative_slope); // aten::leaky_relu_(Tensor(a!) self, Scalar negative_slope=0.01) -> Tensor(a!)
Tensor & log_sigmoid_out(Tensor & out, const Tensor & self); // aten::log_sigmoid(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
Tensor log_sigmoid(const Tensor & self); // aten::log_sigmoid(Tensor self) -> Tensor
std::tuple<Tensor &,Tensor &> log_sigmoid_forward_out(Tensor & output, Tensor & buffer, const Tensor & self); // aten::log_sigmoid_forward(Tensor self, *, Tensor(a!) output, Tensor(b!) buffer) -> (Tensor(a!), Tensor(b!))
std::tuple<Tensor,Tensor> log_sigmoid_forward(const Tensor & self); // aten::log_sigmoid_forward(Tensor self) -> (Tensor output, Tensor buffer)
Tensor & log_sigmoid_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & buffer); // aten::log_sigmoid_backward(Tensor grad_output, Tensor self, Tensor buffer, *, Tensor(a!) grad_input) -> Tensor(a!)
Tensor log_sigmoid_backward(const Tensor & grad_output, const Tensor & self, const Tensor & buffer); // aten::log_sigmoid_backward(Tensor grad_output, Tensor self, Tensor buffer) -> Tensor
Tensor & rrelu_with_noise_out(Tensor & out, const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator * generator); // aten::rrelu_with_noise(Tensor self, Tensor noise, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=False, Generator? generator=None, *, Tensor(a!) out) -> Tensor(a!)
Tensor rrelu_with_noise(const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator * generator); // aten::rrelu_with_noise(Tensor self, Tensor noise, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=False, Generator? generator=None) -> Tensor
Tensor & rrelu_with_noise_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training); // aten::rrelu_with_noise_backward(Tensor grad_output, Tensor self, Tensor noise, Scalar lower, Scalar upper, bool training, *, Tensor(a!) grad_input) -> Tensor(a!)
Tensor rrelu_with_noise_backward(const Tensor & grad_output, const Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training); // aten::rrelu_with_noise_backward(Tensor grad_output, Tensor self, Tensor noise, Scalar lower, Scalar upper, bool training) -> Tensor
Tensor & rrelu_with_noise_(Tensor & self, const Tensor & noise, Scalar lower, Scalar upper, bool training, Generator * generator); // aten::rrelu_with_noise_(Tensor(a!) self, Tensor noise, Scalar lower=0.125, Scalar upper=0.3333333333333333, bool training=False, Generator? generator=None) -> Tensor(a!)
Tensor & softplus_out(Tensor & out, const Tensor & self, Scalar beta, Scalar threshold); // aten::softplus(Tensor self, Scalar beta=1, Scalar threshold=20, *, Tensor(a!) out) -> Tensor(a!)
Tensor softplus(const Tensor & self, Scalar beta, Scalar threshold); // aten::softplus(Tensor self, Scalar beta=1, Scalar threshold=20) -> Tensor
Tensor & softplus_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar beta, Scalar threshold, const Tensor & output); // aten::softplus_backward(Tensor grad_output, Tensor self, Scalar beta, Scalar threshold, Tensor output, *, Tensor(a!) grad_input) -> Tensor(a!)
Tensor softplus_backward(const Tensor & grad_output, const Tensor & self, Scalar beta, Scalar threshold, const Tensor & output); // aten::softplus_backward(Tensor grad_output, Tensor self, Scalar beta, Scalar threshold, Tensor output) -> Tensor
Tensor & softshrink_out(Tensor & out, const Tensor & self, Scalar lambd); // aten::softshrink(Tensor self, Scalar lambd=0.5, *, Tensor(a!) out) -> Tensor(a!)
Tensor softshrink(const Tensor & self, Scalar lambd); // aten::softshrink(Tensor self, Scalar lambd=0.5) -> Tensor
Tensor & softshrink_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, Scalar lambd); // aten::softshrink_backward(Tensor grad_output, Tensor self, Scalar lambd, *, Tensor(a!) grad_input) -> Tensor(a!)
Tensor softshrink_backward(const Tensor & grad_output, const Tensor & self, Scalar lambd); // aten::softshrink_backward(Tensor grad_output, Tensor self, Scalar lambd) -> Tensor
Tensor & adaptive_avg_pool2d_out(Tensor & out, const Tensor & self, IntArrayRef output_size); // aten::adaptive_avg_pool2d(Tensor self, int[2] output_size, *, Tensor(a!) out) -> Tensor(a!)
Tensor adaptive_avg_pool2d(const Tensor & self, IntArrayRef output_size); // aten::adaptive_avg_pool2d(Tensor self, int[2] output_size) -> Tensor
Tensor mkldnn_adaptive_avg_pool2d(const Tensor & self, IntArrayRef output_size); // aten::mkldnn_adaptive_avg_pool2d(Tensor self, int[2] output_size) -> Tensor
Tensor _adaptive_avg_pool2d(const Tensor & self, IntArrayRef output_size); // aten::_adaptive_avg_pool2d(Tensor self, int[2] output_size) -> Tensor
Tensor _adaptive_avg_pool2d_backward(const Tensor & grad_output, const Tensor & self); // aten::_adaptive_avg_pool2d_backward(Tensor grad_output, Tensor self) -> Tensor
Tensor & adaptive_avg_pool3d_out(Tensor & out, const Tensor & self, IntArrayRef output_size); // aten::adaptive_avg_pool3d(Tensor self, int[3] output_size, *, Tensor(a!) out) -> Tensor(a!)
Tensor adaptive_avg_pool3d(const Tensor & self, IntArrayRef output_size); // aten::adaptive_avg_pool3d(Tensor self, int[3] output_size) -> Tensor
Tensor & adaptive_avg_pool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self); // aten::adaptive_avg_pool3d_backward(Tensor grad_output, Tensor self, *, Tensor(a!) grad_input) -> Tensor(a!)
Tensor adaptive_avg_pool3d_backward(const Tensor & grad_output, const Tensor & self); // aten::adaptive_avg_pool3d_backward(Tensor grad_output, Tensor self) -> Tensor
std::tuple<Tensor &,Tensor &> adaptive_max_pool2d_out(Tensor & out, Tensor & indices, const Tensor & self, IntArrayRef output_size); // aten::adaptive_max_pool2d(Tensor self, int[2] output_size, *, Tensor(a!) out, Tensor(b!) indices) -> (Tensor(a!), Tensor(b!))
std::tuple<Tensor,Tensor> adaptive_max_pool2d(const Tensor & self, IntArrayRef output_size); // aten::adaptive_max_pool2d(Tensor self, int[2] output_size) -> (Tensor, Tensor)
Tensor & adaptive_max_pool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices); // aten::adaptive_max_pool2d_backward(Tensor grad_output, Tensor self, Tensor indices, *, Tensor(a!) grad_input) -> Tensor(a!)
Tensor adaptive_max_pool2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & indices); // aten::adaptive_max_pool2d_backward(Tensor grad_output, Tensor self, Tensor indices) -> Tensor
std::tuple<Tensor &,Tensor &> adaptive_max_pool3d_out(Tensor & out, Tensor & indices, const Tensor & self, IntArrayRef output_size); // aten::adaptive_max_pool3d(Tensor self, int[3] output_size, *, Tensor(a!) out, Tensor(b!) indices) -> (Tensor(a!), Tensor(b!))
std::tuple<Tensor,Tensor> adaptive_max_pool3d(const Tensor & self, IntArrayRef output_size); // aten::adaptive_max_pool3d(Tensor self, int[3] output_size) -> (Tensor, Tensor)
Tensor & adaptive_max_pool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices); // aten::adaptive_max_pool3d_backward(Tensor grad_output, Tensor self, Tensor indices, *, Tensor(a!) grad_input) -> Tensor(a!)
Tensor adaptive_max_pool3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & indices); // aten::adaptive_max_pool3d_backward(Tensor grad_output, Tensor self, Tensor indices) -> Tensor
Tensor & avg_pool2d_out(Tensor & out, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override); // aten::avg_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None, *, Tensor(a!) out) -> Tensor(a!)
Tensor avg_pool2d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override); // aten::avg_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None) -> Tensor
Tensor & avg_pool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override); // aten::avg_pool2d_backward(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, bool ceil_mode, bool count_include_pad, int? divisor_override, *, Tensor(a!) grad_input) -> Tensor(a!)
Tensor avg_pool2d_backward(const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override); // aten::avg_pool2d_backward(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, bool ceil_mode, bool count_include_pad, int? divisor_override) -> Tensor
Tensor & avg_pool3d_out(Tensor & out, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override); // aten::avg_pool3d(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None, *, Tensor(a!) out) -> Tensor(a!)
Tensor avg_pool3d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override); // aten::avg_pool3d(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None) -> Tensor
Tensor & avg_pool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override); // aten::avg_pool3d_backward(Tensor grad_output, Tensor self, int[3] kernel_size, int[3] stride, int[3] padding, bool ceil_mode, bool count_include_pad, int? divisor_override, *, Tensor(a!) grad_input) -> Tensor(a!)
Tensor avg_pool3d_backward(const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, bool ceil_mode, bool count_include_pad, c10::optional<int64_t> divisor_override); // aten::avg_pool3d_backward(Tensor grad_output, Tensor self, int[3] kernel_size, int[3] stride, int[3] padding, bool ceil_mode, bool count_include_pad, int? divisor_override) -> Tensor
std::tuple<Tensor &,Tensor &> fractional_max_pool2d_out(Tensor & output, Tensor & indices, const Tensor & self, IntArrayRef kernel_size, IntArrayRef output_size, const Tensor & random_samples); // aten::fractional_max_pool2d(Tensor self, int[2] kernel_size, int[2] output_size, Tensor random_samples, *, Tensor(a!) output, Tensor(b!) indices) -> (Tensor(a!), Tensor(b!))
std::tuple<Tensor,Tensor> fractional_max_pool2d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef output_size, const Tensor & random_samples); // aten::fractional_max_pool2d(Tensor self, int[2] kernel_size, int[2] output_size, Tensor random_samples) -> (Tensor, Tensor)
Tensor & fractional_max_pool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef output_size, const Tensor & indices); // aten::fractional_max_pool2d_backward(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] output_size, Tensor indices, *, Tensor(a!) grad_input) -> Tensor(a!)
Tensor fractional_max_pool2d_backward(const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef output_size, const Tensor & indices); // aten::fractional_max_pool2d_backward(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] output_size, Tensor indices) -> Tensor
std::tuple<Tensor &,Tensor &> fractional_max_pool3d_out(Tensor & output, Tensor & indices, const Tensor & self, IntArrayRef kernel_size, IntArrayRef output_size, const Tensor & random_samples); // aten::fractional_max_pool3d(Tensor self, int[3] kernel_size, int[3] output_size, Tensor random_samples, *, Tensor(a!) output, Tensor(b!) indices) -> (Tensor(a!), Tensor(b!))
std::tuple<Tensor,Tensor> fractional_max_pool3d(const Tensor & self, IntArrayRef kernel_size, IntArrayRef output_size, const Tensor & random_samples); // aten::fractional_max_pool3d(Tensor self, int[3] kernel_size, int[3] output_size, Tensor random_samples) -> (Tensor, Tensor)
Tensor & fractional_max_pool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef output_size, const Tensor & indices); // aten::fractional_max_pool3d_backward(Tensor grad_output, Tensor self, int[3] kernel_size, int[3] output_size, Tensor indices, *, Tensor(a!) grad_input) -> Tensor(a!)
Tensor fractional_max_pool3d_backward(const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef output_size, const Tensor & indices); // aten::fractional_max_pool3d_backward(Tensor grad_output, Tensor self, int[3] kernel_size, int[3] output_size, Tensor indices) -> Tensor
std::tuple<Tensor &,Tensor &> max_pool2d_with_indices_out(Tensor & out, Tensor & indices, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode); // aten::max_pool2d_with_indices(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False, *, Tensor(a!) out, Tensor(b!) indices) -> (Tensor(a!), Tensor(b!))
std::tuple<Tensor,Tensor> max_pool2d_with_indices(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode); // aten::max_pool2d_with_indices(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> (Tensor, Tensor)
Tensor & max_pool2d_with_indices_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode, const Tensor & indices); // aten::max_pool2d_with_indices_backward(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, int[2] dilation, bool ceil_mode, Tensor indices, *, Tensor(a!) grad_input) -> Tensor(a!)
Tensor max_pool2d_with_indices_backward(const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode, const Tensor & indices); // aten::max_pool2d_with_indices_backward(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, int[2] dilation, bool ceil_mode, Tensor indices) -> Tensor
std::tuple<Tensor &,Tensor &> max_pool3d_with_indices_out(Tensor & out, Tensor & indices, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode); // aten::max_pool3d_with_indices(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, int[3] dilation=1, bool ceil_mode=False, *, Tensor(a!) out, Tensor(b!) indices) -> (Tensor(a!), Tensor(b!))
std::tuple<Tensor,Tensor> max_pool3d_with_indices(const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode); // aten::max_pool3d_with_indices(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, int[3] dilation=1, bool ceil_mode=False) -> (Tensor, Tensor)
Tensor & max_pool3d_with_indices_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode, const Tensor & indices); // aten::max_pool3d_with_indices_backward(Tensor grad_output, Tensor self, int[3] kernel_size, int[3] stride, int[3] padding, int[3] dilation, bool ceil_mode, Tensor indices, *, Tensor(a!) grad_input) -> Tensor(a!)
Tensor max_pool3d_with_indices_backward(const Tensor & grad_output, const Tensor & self, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, bool ceil_mode, const Tensor & indices); // aten::max_pool3d_with_indices_backward(Tensor grad_output, Tensor self, int[3] kernel_size, int[3] stride, int[3] padding, int[3] dilation, bool ceil_mode, Tensor indices) -> Tensor
Tensor & max_unpool2d_out(Tensor & out, const Tensor & self, const Tensor & indices, IntArrayRef output_size); // aten::max_unpool2d(Tensor self, Tensor indices, int[2] output_size, *, Tensor(a!) out) -> Tensor(a!)
Tensor max_unpool2d(const Tensor & self, const Tensor & indices, IntArrayRef output_size); // aten::max_unpool2d(Tensor self, Tensor indices, int[2] output_size) -> Tensor
Tensor & max_unpool2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntArrayRef output_size); // aten::max_unpool2d_backward(Tensor grad_output, Tensor self, Tensor indices, int[2] output_size, *, Tensor(a!) grad_input) -> Tensor(a!)
Tensor max_unpool2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntArrayRef output_size); // aten::max_unpool2d_backward(Tensor grad_output, Tensor self, Tensor indices, int[2] output_size) -> Tensor
Tensor & max_unpool3d_out(Tensor & out, const Tensor & self, const Tensor & indices, IntArrayRef output_size, IntArrayRef stride, IntArrayRef padding); // aten::max_unpool3d(Tensor self, Tensor indices, int[3] output_size, int[3] stride, int[3] padding, *, Tensor(a!) out) -> Tensor(a!)
Tensor max_unpool3d(const Tensor & self, const Tensor & indices, IntArrayRef output_size, IntArrayRef stride, IntArrayRef padding); // aten::max_unpool3d(Tensor self, Tensor indices, int[3] output_size, int[3] stride, int[3] padding) -> Tensor
Tensor & max_unpool3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntArrayRef output_size, IntArrayRef stride, IntArrayRef padding); // aten::max_unpool3d_backward(Tensor grad_output, Tensor self, Tensor indices, int[3] output_size, int[3] stride, int[3] padding, *, Tensor(a!) grad_input) -> Tensor(a!)
Tensor max_unpool3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & indices, IntArrayRef output_size, IntArrayRef stride, IntArrayRef padding); // aten::max_unpool3d_backward(Tensor grad_output, Tensor self, Tensor indices, int[3] output_size, int[3] stride, int[3] padding) -> Tensor
Tensor & reflection_pad1d_out(Tensor & out, const Tensor & self, IntArrayRef padding); // aten::reflection_pad1d(Tensor self, int[2] padding, *, Tensor(a!) out) -> Tensor(a!)
Tensor reflection_pad1d(const Tensor & self, IntArrayRef padding); // aten::reflection_pad1d(Tensor self, int[2] padding) -> Tensor
Tensor & reflection_pad1d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntArrayRef padding); // aten::reflection_pad1d_backward(Tensor grad_output, Tensor self, int[2] padding, *, Tensor(a!) grad_input) -> Tensor(a!)
Tensor reflection_pad1d_backward(const Tensor & grad_output, const Tensor & self, IntArrayRef padding); // aten::reflection_pad1d_backward(Tensor grad_output, Tensor self, int[2] padding) -> Tensor
Tensor & reflection_pad2d_out(Tensor & out, const Tensor & self, IntArrayRef padding); // aten::reflection_pad2d(Tensor self, int[4] padding, *, Tensor(a!) out) -> Tensor(a!)
Tensor reflection_pad2d(const Tensor & self, IntArrayRef padding); // aten::reflection_pad2d(Tensor self, int[4] padding) -> Tensor
Tensor & reflection_pad2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntArrayRef padding); // aten::reflection_pad2d_backward(Tensor grad_output, Tensor self, int[4] padding, *, Tensor(a!) grad_input) -> Tensor(a!)
Tensor reflection_pad2d_backward(const Tensor & grad_output, const Tensor & self, IntArrayRef padding); // aten::reflection_pad2d_backward(Tensor grad_output, Tensor self, int[4] padding) -> Tensor
Tensor & replication_pad1d_out(Tensor & out, const Tensor & self, IntArrayRef padding); // aten::replication_pad1d(Tensor self, int[2] padding, *, Tensor(a!) out) -> Tensor(a!)
Tensor replication_pad1d(const Tensor & self, IntArrayRef padding); // aten::replication_pad1d(Tensor self, int[2] padding) -> Tensor
Tensor & replication_pad1d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntArrayRef padding); // aten::replication_pad1d_backward(Tensor grad_output, Tensor self, int[2] padding, *, Tensor(a!) grad_input) -> Tensor(a!)
Tensor replication_pad1d_backward(const Tensor & grad_output, const Tensor & self, IntArrayRef padding); // aten::replication_pad1d_backward(Tensor grad_output, Tensor self, int[2] padding) -> Tensor
Tensor & replication_pad2d_out(Tensor & out, const Tensor & self, IntArrayRef padding); // aten::replication_pad2d(Tensor self, int[4] padding, *, Tensor(a!) out) -> Tensor(a!)
Tensor replication_pad2d(const Tensor & self, IntArrayRef padding); // aten::replication_pad2d(Tensor self, int[4] padding) -> Tensor
Tensor & replication_pad2d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntArrayRef padding); // aten::replication_pad2d_backward(Tensor grad_output, Tensor self, int[4] padding, *, Tensor(a!) grad_input) -> Tensor(a!)
Tensor replication_pad2d_backward(const Tensor & grad_output, const Tensor & self, IntArrayRef padding); // aten::replication_pad2d_backward(Tensor grad_output, Tensor self, int[4] padding) -> Tensor
Tensor & replication_pad3d_out(Tensor & out, const Tensor & self, IntArrayRef padding); // aten::replication_pad3d(Tensor self, int[6] padding, *, Tensor(a!) out) -> Tensor(a!)
Tensor replication_pad3d(const Tensor & self, IntArrayRef padding); // aten::replication_pad3d(Tensor self, int[6] padding) -> Tensor
Tensor & replication_pad3d_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & self, IntArrayRef padding); // aten::replication_pad3d_backward(Tensor grad_output, Tensor self, int[6] padding, *, Tensor(a!) grad_input) -> Tensor(a!)
Tensor replication_pad3d_backward(const Tensor & grad_output, const Tensor & self, IntArrayRef padding); // aten::replication_pad3d_backward(Tensor grad_output, Tensor self, int[6] padding) -> Tensor
Tensor & upsample_linear1d_out(Tensor & out, const Tensor & self, IntArrayRef output_size, bool align_corners); // aten::upsample_linear1d(Tensor self, int[1] output_size, bool align_corners, *, Tensor(a!) out) -> Tensor(a!)
Tensor upsample_linear1d(const Tensor & self, IntArrayRef output_size, bool align_corners); // aten::upsample_linear1d(Tensor self, int[1] output_size, bool align_corners) -> Tensor
Tensor & upsample_linear1d_backward_out(Tensor & grad_input, const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners); // aten::upsample_linear1d_backward(Tensor grad_output, int[1] output_size, int[3] input_size, bool align_corners, *, Tensor(a!) grad_input) -> Tensor(a!)
Tensor upsample_linear1d_backward(const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners); // aten::upsample_linear1d_backward(Tensor grad_output, int[1] output_size, int[3] input_size, bool align_corners) -> Tensor
Tensor & upsample_bilinear2d_out(Tensor & out, const Tensor & self, IntArrayRef output_size, bool align_corners); // aten::upsample_bilinear2d(Tensor self, int[2] output_size, bool align_corners, *, Tensor(a!) out) -> Tensor(a!)
Tensor upsample_bilinear2d(const Tensor & self, IntArrayRef output_size, bool align_corners); // aten::upsample_bilinear2d(Tensor self, int[2] output_size, bool align_corners) -> Tensor
Tensor & upsample_bilinear2d_backward_out(Tensor & grad_input, const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners); // aten::upsample_bilinear2d_backward(Tensor grad_output, int[2] output_size, int[4] input_size, bool align_corners, *, Tensor(a!) grad_input) -> Tensor(a!)
Tensor upsample_bilinear2d_backward(const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners); // aten::upsample_bilinear2d_backward(Tensor grad_output, int[2] output_size, int[4] input_size, bool align_corners) -> Tensor
Tensor & upsample_bicubic2d_out(Tensor & out, const Tensor & self, IntArrayRef output_size, bool align_corners); // aten::upsample_bicubic2d(Tensor self, int[2] output_size, bool align_corners, *, Tensor(a!) out) -> Tensor(a!)
Tensor upsample_bicubic2d(const Tensor & self, IntArrayRef output_size, bool align_corners); // aten::upsample_bicubic2d(Tensor self, int[2] output_size, bool align_corners) -> Tensor
Tensor & upsample_bicubic2d_backward_out(Tensor & grad_input, const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners); // aten::upsample_bicubic2d_backward(Tensor grad_output, int[2] output_size, int[4] input_size, bool align_corners, *, Tensor(a!) grad_input) -> Tensor(a!)
Tensor upsample_bicubic2d_backward(const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners); // aten::upsample_bicubic2d_backward(Tensor grad_output, int[2] output_size, int[4] input_size, bool align_corners) -> Tensor
Tensor & upsample_trilinear3d_out(Tensor & out, const Tensor & self, IntArrayRef output_size, bool align_corners); // aten::upsample_trilinear3d(Tensor self, int[3] output_size, bool align_corners, *, Tensor(a!) out) -> Tensor(a!)
Tensor upsample_trilinear3d(const Tensor & self, IntArrayRef output_size, bool align_corners); // aten::upsample_trilinear3d(Tensor self, int[3] output_size, bool align_corners) -> Tensor
Tensor & upsample_trilinear3d_backward_out(Tensor & grad_input, const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners); // aten::upsample_trilinear3d_backward(Tensor grad_output, int[3] output_size, int[5] input_size, bool align_corners, *, Tensor(a!) grad_input) -> Tensor(a!)
Tensor upsample_trilinear3d_backward(const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size, bool align_corners); // aten::upsample_trilinear3d_backward(Tensor grad_output, int[3] output_size, int[5] input_size, bool align_corners) -> Tensor
Tensor & upsample_nearest1d_out(Tensor & out, const Tensor & self, IntArrayRef output_size); // aten::upsample_nearest1d(Tensor self, int[1] output_size, *, Tensor(a!) out) -> Tensor(a!)
Tensor upsample_nearest1d(const Tensor & self, IntArrayRef output_size); // aten::upsample_nearest1d(Tensor self, int[1] output_size) -> Tensor
Tensor & upsample_nearest1d_backward_out(Tensor & grad_input, const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size); // aten::upsample_nearest1d_backward(Tensor grad_output, int[1] output_size, int[3] input_size, *, Tensor(a!) grad_input) -> Tensor(a!)
Tensor upsample_nearest1d_backward(const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size); // aten::upsample_nearest1d_backward(Tensor grad_output, int[1] output_size, int[3] input_size) -> Tensor
Tensor & upsample_nearest2d_out(Tensor & out, const Tensor & self, IntArrayRef output_size); // aten::upsample_nearest2d(Tensor self, int[2] output_size, *, Tensor(a!) out) -> Tensor(a!)
Tensor upsample_nearest2d(const Tensor & self, IntArrayRef output_size); // aten::upsample_nearest2d(Tensor self, int[2] output_size) -> Tensor
Tensor & upsample_nearest2d_backward_out(Tensor & grad_input, const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size); // aten::upsample_nearest2d_backward(Tensor grad_output, int[2] output_size, int[4] input_size, *, Tensor(a!) grad_input) -> Tensor(a!)
Tensor upsample_nearest2d_backward(const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size); // aten::upsample_nearest2d_backward(Tensor grad_output, int[2] output_size, int[4] input_size) -> Tensor
Tensor & upsample_nearest3d_out(Tensor & out, const Tensor & self, IntArrayRef output_size); // aten::upsample_nearest3d(Tensor self, int[3] output_size, *, Tensor(a!) out) -> Tensor(a!)
Tensor upsample_nearest3d(const Tensor & self, IntArrayRef output_size); // aten::upsample_nearest3d(Tensor self, int[3] output_size) -> Tensor
Tensor & upsample_nearest3d_backward_out(Tensor & grad_input, const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size); // aten::upsample_nearest3d_backward(Tensor grad_output, int[3] output_size, int[5] input_size, *, Tensor(a!) grad_input) -> Tensor(a!)
Tensor upsample_nearest3d_backward(const Tensor & grad_output, IntArrayRef output_size, IntArrayRef input_size); // aten::upsample_nearest3d_backward(Tensor grad_output, int[3] output_size, int[5] input_size) -> Tensor
Tensor & sigmoid_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & output); // aten::sigmoid_backward(Tensor grad_output, Tensor output, *, Tensor(a!) grad_input) -> Tensor(a!)
Tensor sigmoid_backward(const Tensor & grad_output, const Tensor & output); // aten::sigmoid_backward(Tensor grad_output, Tensor output) -> Tensor
Tensor & tanh_backward_out(Tensor & grad_input, const Tensor & grad_output, const Tensor & output); // aten::tanh_backward(Tensor grad_output, Tensor output, *, Tensor(a!) grad_input) -> Tensor(a!)
Tensor tanh_backward(const Tensor & grad_output, const Tensor & output); // aten::tanh_backward(Tensor grad_output, Tensor output) -> Tensor
Tensor & conv_transpose2d_out(Tensor & out, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation); // aten::conv_transpose2d(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] output_padding=0, int[2] dilation=1, *, Tensor(a!) out) -> Tensor(a!)
Tensor conv_transpose2d(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation); // aten::conv_transpose2d(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] output_padding=0, int[2] dilation=1) -> Tensor
std::tuple<Tensor &,Tensor &,Tensor &> conv_transpose2d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation, const Tensor & columns, const Tensor & ones); // aten::conv_transpose2d_backward(Tensor grad_output, Tensor self, Tensor weight, int[2] kernel_size, int[2] stride, int[2] padding, int[2] output_padding, int[2] dilation, Tensor columns, Tensor ones, *, Tensor?(a!) grad_input, Tensor?(b!) grad_weight, Tensor?(c!) grad_bias) -> (Tensor(a!), Tensor(b!), Tensor(c!))
std::tuple<Tensor,Tensor,Tensor> conv_transpose2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation, const Tensor & columns, const Tensor & ones, std::array<bool,3> output_mask); // aten::conv_transpose2d_backward(Tensor grad_output, Tensor self, Tensor weight, int[2] kernel_size, int[2] stride, int[2] padding, int[2] output_padding, int[2] dilation, Tensor columns, Tensor ones, bool[3] output_mask) -> (Tensor grad_input, Tensor grad_weight, Tensor grad_bias)
Tensor & conv_transpose3d_out(Tensor & out, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation); // aten::conv_transpose3d(Tensor self, Tensor weight, int[3] kernel_size, Tensor? bias=None, int[3] stride=1, int[3] padding=0, int[3] output_padding=0, int[3] dilation=1, *, Tensor(a!) out) -> Tensor(a!)
Tensor conv_transpose3d(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation); // aten::conv_transpose3d(Tensor self, Tensor weight, int[3] kernel_size, Tensor? bias=None, int[3] stride=1, int[3] padding=0, int[3] output_padding=0, int[3] dilation=1) -> Tensor
std::tuple<Tensor &,Tensor &,Tensor &> conv_transpose3d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation, const Tensor & finput, const Tensor & fgrad_input); // aten::conv_transpose3d_backward(Tensor grad_output, Tensor self, Tensor weight, int[3] kernel_size, int[3] stride, int[3] padding, int[3] output_padding, int[3] dilation, Tensor finput, Tensor fgrad_input, *, Tensor?(a!) grad_input, Tensor?(b!) grad_weight, Tensor?(c!) grad_bias) -> (Tensor(a!), Tensor(b!), Tensor(c!))
std::tuple<Tensor,Tensor,Tensor> conv_transpose3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef output_padding, IntArrayRef dilation, const Tensor & finput, const Tensor & fgrad_input, std::array<bool,3> output_mask); // aten::conv_transpose3d_backward(Tensor grad_output, Tensor self, Tensor weight, int[3] kernel_size, int[3] stride, int[3] padding, int[3] output_padding, int[3] dilation, Tensor finput, Tensor fgrad_input, bool[3] output_mask) -> (Tensor grad_input, Tensor grad_weight, Tensor grad_bias)
Tensor & thnn_conv2d_out(Tensor & out, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding); // aten::thnn_conv2d(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias=None, int[2] stride=1, int[2] padding=0, *, Tensor(a!) out) -> Tensor(a!)
Tensor thnn_conv2d(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding); // aten::thnn_conv2d(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias=None, int[2] stride=1, int[2] padding=0) -> Tensor
std::tuple<Tensor &,Tensor &,Tensor &> thnn_conv2d_forward_out(Tensor & output, Tensor & finput, Tensor & fgrad_input, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding); // aten::thnn_conv2d_forward(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias, int[2] stride, int[2] padding, *, Tensor(a!) output, Tensor(b!) finput, Tensor(c!) fgrad_input) -> (Tensor(a!), Tensor(b!), Tensor(c!))
std::tuple<Tensor,Tensor,Tensor> thnn_conv2d_forward(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding); // aten::thnn_conv2d_forward(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias, int[2] stride, int[2] padding) -> (Tensor output, Tensor finput, Tensor fgrad_input)
std::tuple<Tensor &,Tensor &,Tensor &> thnn_conv2d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, const Tensor & finput, const Tensor & fgrad_input); // aten::thnn_conv2d_backward(Tensor grad_output, Tensor self, Tensor weight, int[2] kernel_size, int[2] stride, int[2] padding, Tensor finput, Tensor fgrad_input, *, Tensor?(a!) grad_input, Tensor?(b!) grad_weight, Tensor?(c!) grad_bias) -> (Tensor(a!), Tensor(b!), Tensor(c!))
std::tuple<Tensor,Tensor,Tensor> thnn_conv2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, const Tensor & finput, const Tensor & fgrad_input, std::array<bool,3> output_mask); // aten::thnn_conv2d_backward(Tensor grad_output, Tensor self, Tensor weight, int[2] kernel_size, int[2] stride, int[2] padding, Tensor finput, Tensor fgrad_input, bool[3] output_mask) -> (Tensor grad_input, Tensor grad_weight, Tensor grad_bias)
Tensor & thnn_conv_depthwise2d_out(Tensor & out, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation); // aten::thnn_conv_depthwise2d(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1, *, Tensor(a!) out) -> Tensor(a!)
Tensor thnn_conv_depthwise2d(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation); // aten::thnn_conv_depthwise2d(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1) -> Tensor
Tensor & thnn_conv_depthwise2d_forward_out(Tensor & out, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation); // aten::thnn_conv_depthwise2d_forward(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias, int[2] stride, int[2] padding, int[2] dilation, *, Tensor(a!) out) -> Tensor(a!)
Tensor thnn_conv_depthwise2d_forward(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation); // aten::thnn_conv_depthwise2d_forward(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias, int[2] stride, int[2] padding, int[2] dilation) -> Tensor
std::tuple<Tensor &,Tensor &> thnn_conv_depthwise2d_backward_out(Tensor & grad_input, Tensor & grad_weight, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation); // aten::thnn_conv_depthwise2d_backward(Tensor grad_output, Tensor self, Tensor weight, int[2] kernel_size, int[2] stride, int[2] padding, int[2] dilation, *, Tensor?(a!) grad_input, Tensor?(b!) grad_weight) -> (Tensor(a!), Tensor(b!))
std::tuple<Tensor,Tensor> thnn_conv_depthwise2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, std::array<bool,2> output_mask); // aten::thnn_conv_depthwise2d_backward(Tensor grad_output, Tensor self, Tensor weight, int[2] kernel_size, int[2] stride, int[2] padding, int[2] dilation, bool[2] output_mask) -> (Tensor grad_input, Tensor grad_weight)
Tensor & thnn_conv3d_out(Tensor & out, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding); // aten::thnn_conv3d(Tensor self, Tensor weight, int[3] kernel_size, Tensor? bias=None, int[3] stride=1, int[3] padding=0, *, Tensor(a!) out) -> Tensor(a!)
Tensor thnn_conv3d(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding); // aten::thnn_conv3d(Tensor self, Tensor weight, int[3] kernel_size, Tensor? bias=None, int[3] stride=1, int[3] padding=0) -> Tensor
std::tuple<Tensor &,Tensor &,Tensor &> thnn_conv3d_forward_out(Tensor & output, Tensor & finput, Tensor & fgrad_input, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding); // aten::thnn_conv3d_forward(Tensor self, Tensor weight, int[3] kernel_size, Tensor? bias, int[3] stride, int[3] padding, *, Tensor(a!) output, Tensor(b!) finput, Tensor(c!) fgrad_input) -> (Tensor(a!), Tensor(b!), Tensor(c!))
std::tuple<Tensor,Tensor,Tensor> thnn_conv3d_forward(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding); // aten::thnn_conv3d_forward(Tensor self, Tensor weight, int[3] kernel_size, Tensor? bias, int[3] stride, int[3] padding) -> (Tensor output, Tensor finput, Tensor fgrad_input)
std::tuple<Tensor &,Tensor &,Tensor &> thnn_conv3d_backward_out(Tensor & grad_input, Tensor & grad_weight, Tensor & grad_bias, const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, const Tensor & finput, const Tensor & fgrad_input); // aten::thnn_conv3d_backward(Tensor grad_output, Tensor self, Tensor weight, int[3] kernel_size, int[3] stride, int[3] padding, Tensor finput, Tensor fgrad_input, *, Tensor?(a!) grad_input, Tensor?(b!) grad_weight, Tensor?(c!) grad_bias) -> (Tensor(a!), Tensor(b!), Tensor(c!))
std::tuple<Tensor,Tensor,Tensor> thnn_conv3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, const Tensor & finput, const Tensor & fgrad_input, std::array<bool,3> output_mask); // aten::thnn_conv3d_backward(Tensor grad_output, Tensor self, Tensor weight, int[3] kernel_size, int[3] stride, int[3] padding, Tensor finput, Tensor fgrad_input, bool[3] output_mask) -> (Tensor grad_input, Tensor grad_weight, Tensor grad_bias)
Tensor conv_dilated2d(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation); // aten::conv_dilated2d(Tensor self, Tensor weight, int[2] kernel_size, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1) -> Tensor
std::tuple<Tensor,Tensor,Tensor> conv_dilated2d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, std::array<bool,3> output_mask); // aten::conv_dilated2d_backward(Tensor grad_output, Tensor self, Tensor weight, int[2] kernel_size, int[2] stride, int[2] padding, int[2] dilation, bool[3] output_mask) -> (Tensor grad_input, Tensor grad_weight, Tensor grad_bias)
Tensor conv_dilated3d(const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, const Tensor & bias, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation); // aten::conv_dilated3d(Tensor self, Tensor weight, int[3] kernel_size, Tensor? bias=None, int[3] stride=1, int[3] padding=0, int[3] dilation=1) -> Tensor
std::tuple<Tensor,Tensor,Tensor> conv_dilated3d_backward(const Tensor & grad_output, const Tensor & self, const Tensor & weight, IntArrayRef kernel_size, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation, std::array<bool,3> output_mask); // aten::conv_dilated3d_backward(Tensor grad_output, Tensor self, Tensor weight, int[3] kernel_size, int[3] stride, int[3] padding, int[3] dilation, bool[3] output_mask) -> (Tensor grad_input, Tensor grad_weight, Tensor grad_bias)
Tensor & col2im_out(Tensor & out, const Tensor & self, IntArrayRef output_size, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride); // aten::col2im(Tensor self, int[2] output_size, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride, *, Tensor(a!) out) -> Tensor(a!)
Tensor col2im(const Tensor & self, IntArrayRef output_size, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride); // aten::col2im(Tensor self, int[2] output_size, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride) -> Tensor
Tensor & col2im_backward_out(Tensor & grad_input, const Tensor & grad_output, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride); // aten::col2im_backward(Tensor grad_output, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride, *, Tensor(a!) grad_input) -> Tensor(a!)
Tensor col2im_backward(const Tensor & grad_output, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride); // aten::col2im_backward(Tensor grad_output, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride) -> Tensor
Tensor & im2col_out(Tensor & out, const Tensor & self, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride); // aten::im2col(Tensor self, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride, *, Tensor(a!) out) -> Tensor(a!)
Tensor im2col(const Tensor & self, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride); // aten::im2col(Tensor self, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride) -> Tensor
Tensor & im2col_backward_out(Tensor & grad_input, const Tensor & grad_output, IntArrayRef input_size, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride); // aten::im2col_backward(Tensor grad_output, int[2] input_size, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride, *, Tensor(a!) grad_input) -> Tensor(a!)
Tensor im2col_backward(const Tensor & grad_output, IntArrayRef input_size, IntArrayRef kernel_size, IntArrayRef dilation, IntArrayRef padding, IntArrayRef stride); // aten::im2col_backward(Tensor grad_output, int[2] input_size, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride) -> Tensor
