#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cub/cub.cuh>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

namespace whisper_turbo {
namespace cuda {
namespace batch {

// Batch audio preprocessing kernel with per-sample normalization
template<int BLOCK_SIZE = 256>
__global__ void batch_preprocess_audio_kernel(
    const float** __restrict__ audio_batch,
    float* __restrict__ preprocessed,
    const int* __restrict__ sample_counts,
    int batch_size,
    int max_samples) {

    const int batch_idx = blockIdx.y;
    const int sample_idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    if (batch_idx >= batch_size || sample_idx >= max_samples) return;

    const float* audio = audio_batch[batch_idx];
    const int num_samples = sample_counts ? sample_counts[batch_idx] : max_samples;

    // Output location
    float* output = preprocessed + batch_idx * max_samples + sample_idx;

    if (sample_idx < num_samples) {
        float value = audio[sample_idx];

        // Apply pre-emphasis filter
        float pre_emphasis = 0.97f;
        if (sample_idx > 0) {
            value = value - pre_emphasis * audio[sample_idx - 1];
        }

        // Normalize to [-1, 1]
        value = fmaxf(-1.0f, fminf(1.0f, value));

        *output = value;
    } else {
        // Pad with zeros
        *output = 0.0f;
    }
}

// Parallel log-mel spectrogram extraction for batch
template<int FFT_SIZE = 400, int HOP_SIZE = 160, int N_MELS = 80>
__global__ void batch_logmel_spectrogram_kernel(
    const float* __restrict__ preprocessed,
    float* __restrict__ mel_features,
    int batch_size,
    int num_frames,
    int frame_size) {

    const int batch_idx = blockIdx.z;
    const int frame_idx = blockIdx.y;
    const int mel_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx >= batch_size || frame_idx >= num_frames || mel_idx >= N_MELS) {
        return;
    }

    // Shared memory for FFT window
    extern __shared__ float s_fft_window[];

    // Load audio frame into shared memory
    const int tid = threadIdx.x;
    const float* frame_start = preprocessed + batch_idx * (num_frames * HOP_SIZE + FFT_SIZE) +
                               frame_idx * HOP_SIZE;

    if (tid < FFT_SIZE) {
        // Apply Hanning window
        float window = 0.5f * (1.0f - cosf(2.0f * M_PI * tid / (FFT_SIZE - 1)));
        s_fft_window[tid] = frame_start[tid] * window;
    }
    __syncthreads();

    // Compute mel filter bank coefficient for this mel channel
    float mel_coeff = 0.0f;

    // Simplified mel filter bank computation
    const float mel_low = 0.0f;
    const float mel_high = 2595.0f * log10f(1.0f + 8000.0f / 700.0f);
    const float mel_step = (mel_high - mel_low) / (N_MELS + 1);

    const float mel_center = mel_low + (mel_idx + 1) * mel_step;
    const float freq_center = 700.0f * (powf(10.0f, mel_center / 2595.0f) - 1.0f);

    // Apply triangular filter
    for (int freq_bin = tid; freq_bin < FFT_SIZE / 2; freq_bin += blockDim.x) {
        float freq = freq_bin * 16000.0f / FFT_SIZE;
        float filter_val = 0.0f;

        if (freq >= freq_center - 200.0f && freq <= freq_center + 200.0f) {
            filter_val = 1.0f - fabsf(freq - freq_center) / 200.0f;
        }

        // Accumulate filtered energy
        mel_coeff += filter_val * s_fft_window[freq_bin] * s_fft_window[freq_bin];
    }

    // Reduce across warp
    for (int offset = 16; offset > 0; offset >>= 1) {
        mel_coeff += __shfl_down_sync(0xFFFFFFFF, mel_coeff, offset);
    }

    // Write result
    if (tid == 0) {
        int output_idx = batch_idx * num_frames * N_MELS + frame_idx * N_MELS + mel_idx;
        mel_features[output_idx] = log10f(mel_coeff + 1e-10f);
    }
}

// Batch transformer encoder with variable sequence lengths
template<int HIDDEN_DIM = 1024, int NUM_HEADS = 16, int HEAD_DIM = 64>
__global__ void batch_transformer_encoder_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int* __restrict__ seq_lengths,
    const float* __restrict__ weights,
    int batch_size,
    int max_seq_len) {

    const int batch_idx = blockIdx.z;
    const int seq_idx = blockIdx.y;
    const int hidden_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx >= batch_size || seq_idx >= max_seq_len || hidden_idx >= HIDDEN_DIM) {
        return;
    }

    const int actual_seq_len = seq_lengths ? seq_lengths[batch_idx] : max_seq_len;

    // Skip computation for padded positions
    if (seq_idx >= actual_seq_len) {
        output[batch_idx * max_seq_len * HIDDEN_DIM + seq_idx * HIDDEN_DIM + hidden_idx] = 0.0f;
        return;
    }

    // Input offset for this batch and position
    const float* batch_input = input + batch_idx * max_seq_len * HIDDEN_DIM;
    float* batch_output = output + batch_idx * max_seq_len * HIDDEN_DIM;

    // Simplified self-attention (would be full implementation in production)
    float value = batch_input[seq_idx * HIDDEN_DIM + hidden_idx];

    // Layer norm
    float mean = 0.0f;
    float variance = 0.0f;

    for (int i = 0; i < HIDDEN_DIM; i++) {
        mean += batch_input[seq_idx * HIDDEN_DIM + i];
    }
    mean /= HIDDEN_DIM;

    for (int i = 0; i < HIDDEN_DIM; i++) {
        float diff = batch_input[seq_idx * HIDDEN_DIM + i] - mean;
        variance += diff * diff;
    }
    variance = sqrtf(variance / HIDDEN_DIM + 1e-6f);

    value = (value - mean) / variance;

    // Multi-head attention (simplified)
    const int head_idx = hidden_idx / HEAD_DIM;
    const int head_offset = hidden_idx % HEAD_DIM;

    float attention_output = 0.0f;

    // Compute attention scores for this head
    for (int i = 0; i < actual_seq_len; i++) {
        float q = value;
        float k = batch_input[i * HIDDEN_DIM + hidden_idx];
        float v = batch_input[i * HIDDEN_DIM + hidden_idx];

        float score = (q * k) / sqrtf(float(HEAD_DIM));

        // Apply causal mask if needed
        if (i > seq_idx) {
            score = -INFINITY;
        }

        score = expf(score);
        attention_output += score * v;
    }

    // Write output
    batch_output[seq_idx * HIDDEN_DIM + hidden_idx] = attention_output;
}

// Batch beam search decoder
template<int VOCAB_SIZE = 51865, int BEAM_WIDTH = 5>
__global__ void batch_beam_search_kernel(
    const float* __restrict__ encoder_output,
    int* __restrict__ output_tokens,
    float* __restrict__ log_probs,
    const float* __restrict__ lm_weights,
    int batch_size,
    int max_length,
    int encoder_seq_len) {

    const int batch_idx = blockIdx.y;
    const int beam_idx = blockIdx.x;
    const int token_idx = threadIdx.x;

    if (batch_idx >= batch_size || beam_idx >= BEAM_WIDTH) return;

    // Shared memory for beam candidates
    __shared__ float s_beam_scores[BEAM_WIDTH * 32];  // Top 32 candidates per beam
    __shared__ int s_beam_tokens[BEAM_WIDTH * 32];

    // Initialize beam search state
    if (token_idx == 0 && beam_idx == 0) {
        // Start with BOS token
        output_tokens[batch_idx * max_length * BEAM_WIDTH + 0] = 50257;  // <|startoftranscript|>
        log_probs[batch_idx * BEAM_WIDTH + 0] = 0.0f;
    }
    __syncthreads();

    // Beam search iterations
    for (int step = 1; step < max_length; step++) {
        // Get encoder hidden state for current position
        const float* hidden = encoder_output +
            batch_idx * encoder_seq_len * 1024 + (step % encoder_seq_len) * 1024;

        // Compute next token probabilities (simplified)
        float token_logit = 0.0f;

        if (token_idx < VOCAB_SIZE) {
            // Simplified logit computation
            for (int i = 0; i < 1024; i += 32) {
                token_logit += hidden[i] * lm_weights[token_idx * 1024 + i];
            }

            // Store in shared memory for reduction
            if (token_idx < 32) {
                s_beam_scores[beam_idx * 32 + token_idx] = token_logit;
                s_beam_tokens[beam_idx * 32 + token_idx] = token_idx;
            }
        }
        __syncthreads();

        // Select top tokens for beam expansion (simplified)
        if (beam_idx == 0 && token_idx < BEAM_WIDTH) {
            // Find top-k tokens
            int best_token = 50257;  // Default
            float best_score = -INFINITY;

            for (int i = 0; i < 32; i++) {
                if (s_beam_scores[token_idx * 32 + i] > best_score) {
                    best_score = s_beam_scores[token_idx * 32 + i];
                    best_token = s_beam_tokens[token_idx * 32 + i];
                }
            }

            // Update beam
            int output_idx = batch_idx * max_length * BEAM_WIDTH +
                           step * BEAM_WIDTH + token_idx;
            output_tokens[output_idx] = best_token;
            log_probs[batch_idx * BEAM_WIDTH + token_idx] += best_score;
        }
        __syncthreads();
    }
}

// Optimized batch matrix multiplication for decoder
template<typename T, int TILE_SIZE = 16>
__global__ void batch_gemm_kernel(
    const T* __restrict__ A,
    const T* __restrict__ B,
    T* __restrict__ C,
    int batch_size,
    int M, int N, int K,
    float alpha = 1.0f,
    float beta = 0.0f) {

    const int batch_idx = blockIdx.z;
    const int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    const int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    if (batch_idx >= batch_size || row >= M || col >= N) return;

    // Compute C[batch][row][col]
    T sum = 0;

    const T* A_batch = A + batch_idx * M * K;
    const T* B_batch = B + batch_idx * K * N;
    T* C_batch = C + batch_idx * M * N;

    // Tiled matrix multiplication
    __shared__ T s_A[TILE_SIZE][TILE_SIZE];
    __shared__ T s_B[TILE_SIZE][TILE_SIZE];

    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // Load tiles into shared memory
        int a_col = tile * TILE_SIZE + threadIdx.x;
        int b_row = tile * TILE_SIZE + threadIdx.y;

        if (row < M && a_col < K) {
            s_A[threadIdx.y][threadIdx.x] = A_batch[row * K + a_col];
        } else {
            s_A[threadIdx.y][threadIdx.x] = 0;
        }

        if (b_row < K && col < N) {
            s_B[threadIdx.y][threadIdx.x] = B_batch[b_row * N + col];
        } else {
            s_B[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();

        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Write result
    if (row < M && col < N) {
        C_batch[row * N + col] = alpha * sum + beta * C_batch[row * N + col];
    }
}

// Dynamic sequence packing to minimize padding
__global__ void pack_sequences_kernel(
    const float* __restrict__ padded_batch,
    float* __restrict__ packed_batch,
    const int* __restrict__ seq_lengths,
    const int* __restrict__ seq_offsets,
    int batch_size,
    int max_seq_len,
    int hidden_dim) {

    const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * max_seq_len * hidden_dim;

    if (global_idx >= total_elements) return;

    // Determine batch, sequence position, and feature
    const int batch_idx = global_idx / (max_seq_len * hidden_dim);
    const int seq_idx = (global_idx % (max_seq_len * hidden_dim)) / hidden_dim;
    const int feat_idx = global_idx % hidden_dim;

    const int actual_len = seq_lengths[batch_idx];

    // Skip padding
    if (seq_idx >= actual_len) return;

    // Pack into continuous memory
    const int packed_offset = seq_offsets[batch_idx] + seq_idx * hidden_dim + feat_idx;
    packed_batch[packed_offset] = padded_batch[global_idx];
}

// Unpack results back to padded format
__global__ void unpack_sequences_kernel(
    const float* __restrict__ packed_batch,
    float* __restrict__ padded_batch,
    const int* __restrict__ seq_lengths,
    const int* __restrict__ seq_offsets,
    int batch_size,
    int max_seq_len,
    int hidden_dim) {

    const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batch_size * max_seq_len * hidden_dim;

    if (global_idx >= total_elements) return;

    const int batch_idx = global_idx / (max_seq_len * hidden_dim);
    const int seq_idx = (global_idx % (max_seq_len * hidden_dim)) / hidden_dim;
    const int feat_idx = global_idx % hidden_dim;

    const int actual_len = seq_lengths[batch_idx];

    if (seq_idx >= actual_len) {
        // Zero out padding
        padded_batch[global_idx] = 0.0f;
    } else {
        // Unpack from continuous memory
        const int packed_offset = seq_offsets[batch_idx] + seq_idx * hidden_dim + feat_idx;
        padded_batch[global_idx] = packed_batch[packed_offset];
    }
}

// Host-side wrapper functions
void batch_preprocess_audio(const float** audio_batch, float* preprocessed,
                           int batch_size, int num_samples,
                           cudaStream_t stream) {

    const int block_size = 256;
    dim3 blocks((num_samples + block_size - 1) / block_size, batch_size);
    dim3 threads(block_size);

    batch_preprocess_audio_kernel<block_size><<<blocks, threads, 0, stream>>>(
        audio_batch, preprocessed, nullptr, batch_size, num_samples
    );
}

void batch_encode_transformer(const float* input, float* encoded,
                             int batch_size, int seq_len, int hidden_dim,
                             cudaStream_t stream) {

    dim3 blocks((hidden_dim + 31) / 32, seq_len, batch_size);
    dim3 threads(32);

    batch_transformer_encoder_kernel<1024, 16, 64><<<blocks, threads, 0, stream>>>(
        input, encoded, nullptr, nullptr, batch_size, seq_len
    );
}

void batch_decode_transformer(const float* encoded, int* tokens,
                             int batch_size, int max_length,
                             cudaStream_t stream) {

    dim3 blocks(5, batch_size);  // 5 beams
    dim3 threads(256);  // For vocabulary processing

    batch_beam_search_kernel<51865, 5><<<blocks, threads, 0, stream>>>(
        encoded, tokens, nullptr, nullptr, batch_size, max_length, 1500
    );
}

// Optimized batch GEMM for mixed precision
template void batch_gemm_kernel<float, 16>(const float*, const float*, float*,
                                           int, int, int, int, float, float);
template void batch_gemm_kernel<__half, 16>(const __half*, const __half*, __half*,
                                            int, int, int, int, float, float);

} // namespace batch
} // namespace cuda
} // namespace whisper_turbo