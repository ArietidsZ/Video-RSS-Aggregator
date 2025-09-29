#include <cuda_runtime.h>
#include <cufft.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

namespace audio_processor {
namespace cuda {

// Constants
constexpr int WARP_SIZE = 32;
constexpr int MAX_THREADS = 1024;
constexpr int SHARED_MEM_SIZE = 48 * 1024; // 48KB shared memory

/**
 * High-performance audio normalization kernel
 * Uses warp-level primitives for maximum throughput
 */
__global__ void normalize_audio_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int num_samples,
    const float target_level,
    const float max_gain
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;

    // Cooperative group for warp-level operations
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());

    // First pass: Find max value using warp reduction
    float local_max = 0.0f;
    for (int i = tid; i < num_samples; i += stride) {
        local_max = fmaxf(local_max, fabsf(input[i]));
    }

    // Warp-level reduction
    for (int offset = warp.size() / 2; offset > 0; offset /= 2) {
        local_max = fmaxf(local_max, warp.shfl_down(local_max, offset));
    }

    // Broadcast max value to all threads
    __shared__ float block_max[32];
    if (warp.thread_rank() == 0) {
        block_max[warp.meta_group_rank()] = local_max;
    }
    __syncthreads();

    // Final reduction in first warp
    if (warp.meta_group_rank() == 0) {
        local_max = (warp.thread_rank() < warp.meta_group_size())
            ? block_max[warp.thread_rank()] : 0.0f;

        for (int offset = warp.size() / 2; offset > 0; offset /= 2) {
            local_max = fmaxf(local_max, warp.shfl_down(local_max, offset));
        }

        if (warp.thread_rank() == 0) {
            block_max[0] = local_max;
        }
    }
    __syncthreads();

    // Calculate gain
    const float max_val = block_max[0];
    const float gain = (max_val > 0.0f)
        ? fminf(target_level / max_val, max_gain) : 1.0f;

    // Apply normalization
    for (int i = tid; i < num_samples; i += stride) {
        output[i] = input[i] * gain;
    }
}

/**
 * GPU-accelerated resampling using cubic interpolation
 * Optimized for audio quality and performance
 */
__global__ void resample_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int input_size,
    const int output_size,
    const float resample_ratio
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= output_size) return;

    // Calculate source position with sub-sample precision
    const float src_pos = tid * resample_ratio;
    const int src_idx = __float2int_rd(src_pos);
    const float frac = src_pos - src_idx;

    // Bounds checking
    if (src_idx < 1 || src_idx >= input_size - 2) {
        output[tid] = (src_idx < input_size) ? input[src_idx] : 0.0f;
        return;
    }

    // Load samples for cubic interpolation
    const float y0 = input[src_idx - 1];
    const float y1 = input[src_idx];
    const float y2 = input[src_idx + 1];
    const float y3 = input[src_idx + 2];

    // Cubic interpolation coefficients
    const float a0 = y3 - y2 - y0 + y1;
    const float a1 = y0 - y1 - a0;
    const float a2 = y2 - y0;
    const float a3 = y1;

    // Compute interpolated value
    output[tid] = ((a0 * frac + a1) * frac + a2) * frac + a3;
}

/**
 * Voice Activity Detection (VAD) kernel
 * Uses energy-based detection with spectral features
 */
__global__ void vad_kernel(
    const float* __restrict__ input,
    bool* __restrict__ output,
    const int num_frames,
    const int frame_size,
    const float energy_threshold,
    const float zero_crossing_threshold
) {
    const int frame_idx = blockIdx.x;
    const int tid = threadIdx.x;

    if (frame_idx >= num_frames) return;

    extern __shared__ float shared_data[];
    float* frame_data = shared_data;

    // Load frame data to shared memory
    const int frame_offset = frame_idx * frame_size;
    for (int i = tid; i < frame_size; i += blockDim.x) {
        frame_data[i] = (frame_offset + i < num_frames * frame_size)
            ? input[frame_offset + i] : 0.0f;
    }
    __syncthreads();

    // Calculate frame energy
    float local_energy = 0.0f;
    int local_zero_crossings = 0;

    for (int i = tid; i < frame_size; i += blockDim.x) {
        local_energy += frame_data[i] * frame_data[i];

        // Count zero crossings
        if (i > 0 && i < frame_size) {
            if ((frame_data[i-1] >= 0) != (frame_data[i] >= 0)) {
                local_zero_crossings++;
            }
        }
    }

    // Reduce energy and zero crossings
    __shared__ float block_energy;
    __shared__ int block_zero_crossings;

    // Warp reduction
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    for (int offset = warp.size() / 2; offset > 0; offset /= 2) {
        local_energy += warp.shfl_down(local_energy, offset);
        local_zero_crossings += warp.shfl_down(local_zero_crossings, offset);
    }

    if (tid == 0) {
        block_energy = local_energy / frame_size;
        block_zero_crossings = local_zero_crossings;
    }
    __syncthreads();

    // Make VAD decision
    if (tid == 0) {
        const float energy_db = 10.0f * log10f(block_energy + 1e-10f);
        const float zcr = (float)block_zero_crossings / frame_size;

        // Combined decision based on energy and zero-crossing rate
        output[frame_idx] = (energy_db > energy_threshold) &&
                           (zcr < zero_crossing_threshold);
    }
}

/**
 * High-performance FFT preparation kernel
 * Applies window function and prepares data for cuFFT
 */
__global__ void prepare_fft_kernel(
    const float* __restrict__ input,
    cufftComplex* __restrict__ output,
    const int size,
    const int window_type // 0: Hamming, 1: Hanning, 2: Blackman
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= size) return;

    // Calculate window coefficient
    float window_coef = 1.0f;
    const float n = (float)tid / (size - 1);

    switch (window_type) {
        case 0: // Hamming
            window_coef = 0.54f - 0.46f * cosf(2.0f * M_PI * n);
            break;
        case 1: // Hanning
            window_coef = 0.5f * (1.0f - cosf(2.0f * M_PI * n));
            break;
        case 2: // Blackman
            window_coef = 0.42f - 0.5f * cosf(2.0f * M_PI * n) +
                         0.08f * cosf(4.0f * M_PI * n);
            break;
    }

    // Apply window and convert to complex
    output[tid].x = input[tid] * window_coef;
    output[tid].y = 0.0f;
}

/**
 * Audio filtering kernel with biquad filters
 * Supports multiple filter types in cascade
 */
__global__ void biquad_filter_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int num_samples,
    const float* __restrict__ coeffs, // b0, b1, b2, a1, a2 for each filter
    const int num_filters,
    float* __restrict__ state // z1, z2 for each filter per block
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int bid = blockIdx.x;

    // Load filter state
    extern __shared__ float shared_state[];
    float* z1 = shared_state;
    float* z2 = shared_state + num_filters;

    if (threadIdx.x < num_filters * 2) {
        const int filter_idx = threadIdx.x % num_filters;
        const int state_type = threadIdx.x / num_filters;
        const int state_idx = bid * num_filters * 2 + filter_idx * 2 + state_type;

        if (state_type == 0) {
            z1[filter_idx] = state[state_idx];
        } else {
            z2[filter_idx] = state[state_idx];
        }
    }
    __syncthreads();

    // Process samples
    for (int i = tid; i < num_samples; i += gridDim.x * blockDim.x) {
        float sample = input[i];

        // Apply cascade of biquad filters
        for (int f = 0; f < num_filters; f++) {
            const float b0 = coeffs[f * 5 + 0];
            const float b1 = coeffs[f * 5 + 1];
            const float b2 = coeffs[f * 5 + 2];
            const float a1 = coeffs[f * 5 + 3];
            const float a2 = coeffs[f * 5 + 4];

            const float output_sample = b0 * sample + z1[f];
            z1[f] = b1 * sample - a1 * output_sample + z2[f];
            z2[f] = b2 * sample - a2 * output_sample;

            sample = output_sample;
        }

        output[i] = sample;
    }

    // Save filter state
    __syncthreads();
    if (threadIdx.x < num_filters * 2) {
        const int filter_idx = threadIdx.x % num_filters;
        const int state_type = threadIdx.x / num_filters;
        const int state_idx = bid * num_filters * 2 + filter_idx * 2 + state_type;

        if (state_type == 0) {
            state[state_idx] = z1[filter_idx];
        } else {
            state[state_idx] = z2[filter_idx];
        }
    }
}

/**
 * Optimized audio mixing kernel
 * Mixes multiple audio streams with gain control
 */
__global__ void mix_audio_kernel(
    const float** __restrict__ inputs,
    float* __restrict__ output,
    const float* __restrict__ gains,
    const int num_streams,
    const int num_samples
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= num_samples) return;

    float mixed_sample = 0.0f;

    // Mix all streams
    for (int s = 0; s < num_streams; s++) {
        mixed_sample += inputs[s][tid] * gains[s];
    }

    // Apply soft clipping to prevent distortion
    const float threshold = 0.95f;
    if (fabsf(mixed_sample) > threshold) {
        const float sign = (mixed_sample > 0) ? 1.0f : -1.0f;
        mixed_sample = sign * (threshold + (1.0f - threshold) *
                      tanhf((fabsf(mixed_sample) - threshold) / (1.0f - threshold)));
    }

    output[tid] = mixed_sample;
}

/**
 * Launch configuration helper
 */
inline void get_launch_config(int num_elements, dim3& grid, dim3& block) {
    block = dim3(256);
    grid = dim3((num_elements + block.x - 1) / block.x);

    // Limit grid size for better occupancy
    if (grid.x > 65535) {
        grid.x = 65535;
    }
}

} // namespace cuda
} // namespace audio_processor