#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <cstdint>
#include <cfloat>
#include <cmath>

// FP8 support for Hopper architecture (H100/H200)
#if __CUDA_ARCH__ >= 900
#include <cuda_fp8.h>
#endif

namespace whisper_turbo {
namespace cuda {
namespace quantization {

// Constants for quantization
constexpr float FP8_E4M3_MAX = 448.0f;
constexpr float FP8_E5M2_MAX = 57344.0f;
constexpr int INT8_MAX = 127;
constexpr int INT8_MIN = -128;
constexpr int INT4_MAX = 7;
constexpr int INT4_MIN = -8;

// Quantization scale and zero-point structure
struct QuantizationParams {
    float scale;
    float zero_point;
    float min_val;
    float max_val;
    int bits;
};

// =================================================================
// FP32 to INT8 Quantization with Per-Channel Scaling
// =================================================================

__global__ void quantize_fp32_to_int8_per_channel(
    const float* __restrict__ input,
    int8_t* __restrict__ output,
    float* __restrict__ scales,
    float* __restrict__ zero_points,
    const int batch_size,
    const int channels,
    const int spatial_size
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * channels * spatial_size;

    if (tid >= total_elements) return;

    const int c = (tid / spatial_size) % channels;
    const float scale = scales[c];
    const float zero_point = zero_points[c];

    // Quantize with rounding
    float value = input[tid];
    int quantized = __float2int_rn(value / scale + zero_point);

    // Clamp to INT8 range
    quantized = min(max(quantized, INT8_MIN), INT8_MAX);
    output[tid] = static_cast<int8_t>(quantized);
}

// Dequantization kernel
__global__ void dequantize_int8_to_fp32_per_channel(
    const int8_t* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ scales,
    const float* __restrict__ zero_points,
    const int batch_size,
    const int channels,
    const int spatial_size
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * channels * spatial_size;

    if (tid >= total_elements) return;

    const int c = (tid / spatial_size) % channels;
    const float scale = scales[c];
    const float zero_point = zero_points[c];

    // Dequantize
    float value = static_cast<float>(input[tid]);
    output[tid] = (value - zero_point) * scale;
}

// =================================================================
// FP32 to FP8 Quantization (H100/H200 only)
// =================================================================

#if __CUDA_ARCH__ >= 900

template<typename FP8Type>
__global__ void quantize_fp32_to_fp8_kernel(
    const float* __restrict__ input,
    FP8Type* __restrict__ output,
    const float scale,
    const int num_elements
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < num_elements; i += stride) {
        float value = input[i] * scale;

        // Clamp to FP8 range
        if constexpr (std::is_same_v<FP8Type, __nv_fp8_e4m3>) {
            value = fminf(fmaxf(value, -FP8_E4M3_MAX), FP8_E4M3_MAX);
        } else {
            value = fminf(fmaxf(value, -FP8_E5M2_MAX), FP8_E5M2_MAX);
        }

        output[i] = __float2fp8(value);
    }
}

// FP8 to FP32 dequantization
template<typename FP8Type>
__global__ void dequantize_fp8_to_fp32_kernel(
    const FP8Type* __restrict__ input,
    float* __restrict__ output,
    const float scale,
    const int num_elements
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < num_elements; i += stride) {
        output[i] = __fp82float(input[i]) / scale;
    }
}

#endif // __CUDA_ARCH__ >= 900

// =================================================================
// INT4 Quantization (4-bit for extreme compression)
// =================================================================

__global__ void quantize_fp32_to_int4_packed(
    const float* __restrict__ input,
    uint8_t* __restrict__ output,  // Pack two INT4 values per byte
    const float* __restrict__ scales,
    const float* __restrict__ zero_points,
    const int num_elements
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int output_idx = tid;
    const int input_idx = tid * 2;

    if (input_idx >= num_elements) return;

    // Quantize two values
    float scale = scales[0];  // Simplified: using single scale
    float zero_point = zero_points[0];

    int val1 = 0, val2 = 0;

    // First value
    if (input_idx < num_elements) {
        float value = input[input_idx];
        val1 = __float2int_rn(value / scale + zero_point);
        val1 = min(max(val1, INT4_MIN), INT4_MAX);
    }

    // Second value
    if (input_idx + 1 < num_elements) {
        float value = input[input_idx + 1];
        val2 = __float2int_rn(value / scale + zero_point);
        val2 = min(max(val2, INT4_MIN), INT4_MAX);
    }

    // Pack two INT4 values into one byte
    output[output_idx] = ((val1 & 0xF) << 4) | (val2 & 0xF);
}

// =================================================================
// Dynamic Quantization with Outlier Detection
// =================================================================

__global__ void compute_dynamic_quantization_params(
    const float* __restrict__ input,
    QuantizationParams* __restrict__ params,
    const int num_elements,
    const float percentile_min = 0.01f,
    const float percentile_max = 0.99f
) {
    // Shared memory for block-wise min/max reduction
    extern __shared__ float shared_mem[];
    float* s_min = shared_mem;
    float* s_max = &shared_mem[blockDim.x];

    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + tid;

    // Initialize with extreme values
    float local_min = FLT_MAX;
    float local_max = -FLT_MAX;

    // Find local min/max
    if (gid < num_elements) {
        float value = input[gid];
        local_min = value;
        local_max = value;
    }

    // Load to shared memory
    s_min[tid] = local_min;
    s_max[tid] = local_max;
    __syncthreads();

    // Parallel reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_min[tid] = fminf(s_min[tid], s_min[tid + stride]);
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + stride]);
        }
        __syncthreads();
    }

    // Write result
    if (tid == 0) {
        atomicMin((int*)&params->min_val, __float_as_int(s_min[0]));
        atomicMax((int*)&params->max_val, __float_as_int(s_max[0]));

        // Calculate scale and zero point for symmetric quantization
        float range = s_max[0] - s_min[0];
        params->scale = range / 254.0f;  // INT8 range
        params->zero_point = -s_min[0] / params->scale - 127.0f;
    }
}

// =================================================================
// Mixed Precision GEMM with INT8 and FP16 Accumulation
// =================================================================

__global__ void mixed_precision_gemm_int8_fp16(
    const int8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    half* __restrict__ C,
    const float* __restrict__ scale_A,
    const float* __restrict__ scale_B,
    const int M,
    const int N,
    const int K,
    const float alpha = 1.0f,
    const float beta = 0.0f
) {
    // Use Tensor Cores for INT8 GEMM on Turing+ architectures
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int block_row = blockIdx.y;
    const int block_col = blockIdx.x;

    // Tile sizes for Tensor Core operations
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;

#if __CUDA_ARCH__ >= 750
    // Use WMMA (Warp Matrix Multiply Accumulate) for Tensor Cores
    using namespace nvcuda::wmma;

    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, int8_t, row_major> a_frag;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, int8_t, col_major> b_frag;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, int32_t> c_frag;

    fill_fragment(c_frag, 0);

    // Loop over K dimension
    for (int k = 0; k < K; k += WMMA_K) {
        int a_row = block_row * WMMA_M;
        int a_col = k;
        int b_row = k;
        int b_col = block_col * WMMA_N;

        // Bounds checking
        if (a_row < M && a_col < K && b_row < K && b_col < N) {
            // Load tiles
            load_matrix_sync(a_frag, A + a_row * K + a_col, K);
            load_matrix_sync(b_frag, B + b_row * N + b_col, N);

            // Perform matrix multiply
            mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
    }

    // Store result with scaling
    int c_row = block_row * WMMA_M;
    int c_col = block_col * WMMA_N;

    if (c_row < M && c_col < N) {
        // Apply scales and convert to FP16
        for (int i = 0; i < c_frag.num_elements; i++) {
            float scaled = c_frag.x[i] * (*scale_A) * (*scale_B) * alpha;
            c_frag.x[i] = __float2half(scaled);
        }

        store_matrix_sync(C + c_row * N + c_col, c_frag, N, mem_row_major);
    }
#else
    // Fallback for older architectures
    const int row = block_row * blockDim.y + threadIdx.y;
    const int col = block_col * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += static_cast<float>(A[row * K + k]) * static_cast<float>(B[k * N + col]);
        }
        C[row * N + col] = __float2half(sum * (*scale_A) * (*scale_B) * alpha);
    }
#endif
}

// =================================================================
// Activation Quantization with Smoothing
// =================================================================

__global__ void smooth_quantize_activations(
    const float* __restrict__ input,
    int8_t* __restrict__ output,
    float* __restrict__ smoothing_factors,
    const int batch_size,
    const int channels,
    const int spatial_size,
    const float smoothing_alpha = 0.1f
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * channels * spatial_size;

    if (tid >= total_elements) return;

    const int c = (tid / spatial_size) % channels;

    // Apply smoothing to reduce quantization error
    float value = input[tid];
    float smooth_factor = smoothing_factors[c];

    // Exponential moving average for smoothing
    smooth_factor = smooth_factor * (1.0f - smoothing_alpha) + fabsf(value) * smoothing_alpha;
    smoothing_factors[c] = smooth_factor;

    // Quantize with smoothed scale
    float scale = smooth_factor / 127.0f;
    int quantized = __float2int_rn(value / scale);
    quantized = min(max(quantized, INT8_MIN), INT8_MAX);

    output[tid] = static_cast<int8_t>(quantized);
}

// =================================================================
// Calibration for Quantization (collect statistics)
// =================================================================

__global__ void collect_calibration_statistics(
    const float* __restrict__ input,
    float* __restrict__ min_values,
    float* __restrict__ max_values,
    float* __restrict__ abs_max_values,
    int* __restrict__ histogram,
    const int num_elements,
    const int num_bins = 2048
) {
    extern __shared__ float shared_stats[];
    float* s_min = shared_stats;
    float* s_max = &shared_stats[blockDim.x];
    float* s_abs_max = &shared_stats[2 * blockDim.x];

    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + tid;

    float local_min = FLT_MAX;
    float local_max = -FLT_MAX;
    float local_abs_max = 0.0f;

    // Process elements
    if (gid < num_elements) {
        float value = input[gid];
        local_min = value;
        local_max = value;
        local_abs_max = fabsf(value);

        // Update histogram
        float normalized = (value - min_values[0]) / (max_values[0] - min_values[0]);
        int bin = min(max(int(normalized * num_bins), 0), num_bins - 1);
        atomicAdd(&histogram[bin], 1);
    }

    // Store in shared memory
    s_min[tid] = local_min;
    s_max[tid] = local_max;
    s_abs_max[tid] = local_abs_max;
    __syncthreads();

    // Parallel reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_min[tid] = fminf(s_min[tid], s_min[tid + stride]);
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + stride]);
            s_abs_max[tid] = fmaxf(s_abs_max[tid], s_abs_max[tid + stride]);
        }
        __syncthreads();
    }

    // Update global statistics
    if (tid == 0) {
        atomicMin((int*)min_values, __float_as_int(s_min[0]));
        atomicMax((int*)max_values, __float_as_int(s_max[0]));
        atomicMax((int*)abs_max_values, __float_as_int(s_abs_max[0]));
    }
}

// =================================================================
// KL Divergence-based Calibration for INT8
// =================================================================

__global__ void compute_kl_divergence_threshold(
    const int* __restrict__ histogram,
    float* __restrict__ kl_divergences,
    const int num_bins,
    const int target_bins = 128  // INT8 range
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= num_bins - target_bins) return;

    // Compute KL divergence for this threshold
    float kl_div = 0.0f;
    int threshold = tid + target_bins;

    // Reference distribution (original)
    int total_ref = 0;
    for (int i = 0; i < threshold; i++) {
        total_ref += histogram[i];
    }

    // Quantized distribution
    int total_quant = 0;
    for (int i = 0; i < target_bins; i++) {
        int sum = 0;
        int start = i * threshold / target_bins;
        int end = (i + 1) * threshold / target_bins;
        for (int j = start; j < end; j++) {
            sum += histogram[j];
        }
        total_quant += sum;

        if (sum > 0 && total_ref > 0) {
            float p = float(sum) / total_ref;
            float q = 1.0f / target_bins;
            kl_div += p * logf(p / q);
        }
    }

    kl_divergences[tid] = kl_div;
}

// =================================================================
// Weight Quantization with Group-wise Scaling
// =================================================================

__global__ void quantize_weights_groupwise(
    const float* __restrict__ weights,
    int8_t* __restrict__ quantized_weights,
    float* __restrict__ scales,
    const int num_groups,
    const int group_size
) {
    const int group_id = blockIdx.x;
    const int tid = threadIdx.x;

    if (group_id >= num_groups) return;

    extern __shared__ float s_max[];

    // Find max absolute value in group
    float local_max = 0.0f;
    int group_start = group_id * group_size;
    int group_end = min(group_start + group_size, group_start + group_size);

    for (int i = group_start + tid; i < group_end; i += blockDim.x) {
        local_max = fmaxf(local_max, fabsf(weights[i]));
    }

    // Reduce to find group max
    s_max[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + stride]);
        }
        __syncthreads();
    }

    // Calculate scale
    float scale = s_max[0] / 127.0f;
    if (tid == 0) {
        scales[group_id] = scale;
    }
    __syncthreads();

    // Quantize group
    for (int i = group_start + tid; i < group_end; i += blockDim.x) {
        int quantized = __float2int_rn(weights[i] / scale);
        quantized = min(max(quantized, INT8_MIN), INT8_MAX);
        quantized_weights[i] = static_cast<int8_t>(quantized);
    }
}

// =================================================================
// Launch wrapper functions
// =================================================================

void quantize_tensor_int8(
    const float* input,
    int8_t* output,
    float* scales,
    float* zero_points,
    int batch_size,
    int channels,
    int spatial_size,
    cudaStream_t stream
) {
    int total_elements = batch_size * channels * spatial_size;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;

    quantize_fp32_to_int8_per_channel<<<grid_size, block_size, 0, stream>>>(
        input, output, scales, zero_points,
        batch_size, channels, spatial_size
    );
}

void dequantize_tensor_int8(
    const int8_t* input,
    float* output,
    const float* scales,
    const float* zero_points,
    int batch_size,
    int channels,
    int spatial_size,
    cudaStream_t stream
) {
    int total_elements = batch_size * channels * spatial_size;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;

    dequantize_int8_to_fp32_per_channel<<<grid_size, block_size, 0, stream>>>(
        input, output, scales, zero_points,
        batch_size, channels, spatial_size
    );
}

#if __CUDA_ARCH__ >= 900
void quantize_tensor_fp8(
    const float* input,
    void* output,
    float scale,
    int num_elements,
    bool use_e4m3,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (num_elements + block_size - 1) / block_size;

    if (use_e4m3) {
        quantize_fp32_to_fp8_kernel<__nv_fp8_e4m3><<<grid_size, block_size, 0, stream>>>(
            input, reinterpret_cast<__nv_fp8_e4m3*>(output), scale, num_elements
        );
    } else {
        quantize_fp32_to_fp8_kernel<__nv_fp8_e5m2><<<grid_size, block_size, 0, stream>>>(
            input, reinterpret_cast<__nv_fp8_e5m2*>(output), scale, num_elements
        );
    }
}
#endif

void calibrate_for_quantization(
    const float* input,
    float* min_val,
    float* max_val,
    float* abs_max,
    int* histogram,
    int num_elements,
    int num_bins,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (num_elements + block_size - 1) / block_size;
    size_t shared_size = 3 * block_size * sizeof(float);

    collect_calibration_statistics<<<grid_size, block_size, shared_size, stream>>>(
        input, min_val, max_val, abs_max, histogram,
        num_elements, num_bins
    );
}

} // namespace quantization
} // namespace cuda
} // namespace whisper_turbo