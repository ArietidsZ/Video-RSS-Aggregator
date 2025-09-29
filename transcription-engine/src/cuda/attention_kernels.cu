#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <cooperative_groups.h>
#include <mma.h>

namespace whisper_turbo {
namespace cuda {

using namespace nvcuda;
namespace cg = cooperative_groups;

// Flash Attention v2 kernel for self-attention
// Optimized for Ampere/Hopper architectures with Tensor Cores
template<typename T, int BLOCK_SIZE = 128, int HEAD_DIM = 64>
__global__ void flash_attention_kernel(
    const T* __restrict__ Q,     // [batch, heads, seq_len, head_dim]
    const T* __restrict__ K,     // [batch, heads, seq_len, head_dim]
    const T* __restrict__ V,     // [batch, heads, seq_len, head_dim]
    T* __restrict__ output,       // [batch, heads, seq_len, head_dim]
    const float scale,
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int head_dim_val
) {
    // Shared memory for Q, K, V tiles and intermediate results
    extern __shared__ char shared_mem[];

    T* s_q = reinterpret_cast<T*>(shared_mem);
    T* s_k = reinterpret_cast<T*>(&s_q[BLOCK_SIZE * HEAD_DIM]);
    T* s_v = reinterpret_cast<T*>(&s_k[BLOCK_SIZE * HEAD_DIM]);
    float* s_scores = reinterpret_cast<float*>(&s_v[BLOCK_SIZE * HEAD_DIM]);

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int head_id = blockIdx.y;
    const int batch_id = blockIdx.z;

    // Calculate global offsets
    const int q_offset = ((batch_id * num_heads + head_id) * seq_len) * head_dim_val;
    const int kv_offset = q_offset;

    // Online softmax variables
    float row_max = -INFINITY;
    float row_sum = 0.0f;

    // Accumulator for output
    float acc[HEAD_DIM];
    #pragma unroll
    for (int i = 0; i < HEAD_DIM; ++i) {
        acc[i] = 0.0f;
    }

    // Process Q blocks
    const int q_block_start = bid * BLOCK_SIZE;
    const int q_block_end = min(q_block_start + BLOCK_SIZE, seq_len);

    // Load Q tile to shared memory
    if (q_block_start + tid < seq_len) {
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; d += 32) {
            if (d + threadIdx.y < head_dim_val) {
                s_q[tid * HEAD_DIM + d + threadIdx.y] =
                    Q[q_offset + (q_block_start + tid) * head_dim_val + d + threadIdx.y];
            }
        }
    }
    __syncthreads();

    // Iterate over K,V blocks for attention computation
    for (int kv_block = 0; kv_block < seq_len; kv_block += BLOCK_SIZE) {
        // Load K tile to shared memory
        if (kv_block + tid < seq_len) {
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; d += 32) {
                if (d + threadIdx.y < head_dim_val) {
                    s_k[tid * HEAD_DIM + d + threadIdx.y] =
                        K[kv_offset + (kv_block + tid) * head_dim_val + d + threadIdx.y];
                }
            }
        }
        __syncthreads();

        // Compute QK^T for this tile
        float local_scores[BLOCK_SIZE];
        #pragma unroll
        for (int k_idx = 0; k_idx < BLOCK_SIZE && kv_block + k_idx < seq_len; ++k_idx) {
            float score = 0.0f;
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) {
                score += float(s_q[tid * HEAD_DIM + d]) * float(s_k[k_idx * HEAD_DIM + d]);
            }
            local_scores[k_idx] = score * scale;
        }

        // Load V tile to shared memory
        if (kv_block + tid < seq_len) {
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; d += 32) {
                if (d + threadIdx.y < head_dim_val) {
                    s_v[tid * HEAD_DIM + d + threadIdx.y] =
                        V[kv_offset + (kv_block + tid) * head_dim_val + d + threadIdx.y];
                }
            }
        }
        __syncthreads();

        // Online softmax and accumulation
        #pragma unroll
        for (int k_idx = 0; k_idx < BLOCK_SIZE && kv_block + k_idx < seq_len; ++k_idx) {
            float score = local_scores[k_idx];

            // Online softmax update
            float new_max = fmaxf(row_max, score);
            float exp_score = expf(score - new_max);
            float exp_row_max = expf(row_max - new_max);

            float new_sum = exp_row_max * row_sum + exp_score;

            // Rescale accumulator
            float rescale = exp_row_max * row_sum / new_sum;
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) {
                acc[d] = acc[d] * rescale + (exp_score / new_sum) * float(s_v[k_idx * HEAD_DIM + d]);
            }

            row_max = new_max;
            row_sum = new_sum;
        }
        __syncthreads();
    }

    // Write output
    if (q_block_start + tid < seq_len) {
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; d += 32) {
            if (d + threadIdx.y < head_dim_val) {
                output[q_offset + (q_block_start + tid) * head_dim_val + d + threadIdx.y] =
                    T(acc[d + threadIdx.y]);
            }
        }
    }
}

// Fused Multi-Head Attention with RoPE (Rotary Position Embedding)
template<typename T>
__global__ void fused_mha_rope_kernel(
    const T* __restrict__ input,
    const float* __restrict__ cos_cache,
    const float* __restrict__ sin_cache,
    T* __restrict__ output,
    const int batch_size,
    const int seq_len,
    const int num_heads,
    const int head_dim
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = gridDim.x * blockDim.x;

    const int total_elements = batch_size * seq_len * num_heads * head_dim;

    for (int idx = tid; idx < total_elements; idx += total_threads) {
        const int d = idx % head_dim;
        const int h = (idx / head_dim) % num_heads;
        const int t = (idx / (head_dim * num_heads)) % seq_len;
        const int b = idx / (head_dim * num_heads * seq_len);

        // Apply RoPE
        const int half_dim = head_dim / 2;
        if (d < half_dim) {
            const float cos_val = cos_cache[t * half_dim + d];
            const float sin_val = sin_cache[t * half_dim + d];

            const int pair_idx = idx + half_dim;
            const float x1 = float(input[idx]);
            const float x2 = float(input[pair_idx]);

            output[idx] = T(x1 * cos_val - x2 * sin_val);
            output[pair_idx] = T(x1 * sin_val + x2 * cos_val);
        }
    }
}

// Causal attention mask application (for autoregressive decoding)
template<typename T>
__global__ void apply_causal_mask_kernel(
    T* __restrict__ scores,
    const int batch_size,
    const int num_heads,
    const int seq_len
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = gridDim.x * blockDim.x;

    const int total_elements = batch_size * num_heads * seq_len * seq_len;

    for (int idx = tid; idx < total_elements; idx += total_threads) {
        const int col = idx % seq_len;
        const int row = (idx / seq_len) % seq_len;

        // Apply causal mask: mask out future positions
        if (col > row) {
            scores[idx] = T(-INFINITY);
        }
    }
}

// Optimized KV cache update kernel
template<typename T>
__global__ void update_kv_cache_kernel(
    const T* __restrict__ new_k,
    const T* __restrict__ new_v,
    T* __restrict__ k_cache,
    T* __restrict__ v_cache,
    const int batch_size,
    const int num_heads,
    const int cache_seq_len,
    const int new_seq_len,
    const int head_dim,
    const int cache_offset
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = gridDim.x * blockDim.x;

    const int new_elements = batch_size * num_heads * new_seq_len * head_dim;

    for (int idx = tid; idx < new_elements; idx += total_threads) {
        const int d = idx % head_dim;
        const int t = (idx / head_dim) % new_seq_len;
        const int h = (idx / (head_dim * new_seq_len)) % num_heads;
        const int b = idx / (head_dim * new_seq_len * num_heads);

        const int cache_idx = ((b * num_heads + h) * cache_seq_len + cache_offset + t) * head_dim + d;

        k_cache[cache_idx] = new_k[idx];
        v_cache[cache_idx] = new_v[idx];
    }
}

// Launch wrapper functions
void launch_flash_attention(
    const void* Q, const void* K, const void* V,
    void* output,
    float scale,
    int batch_size, int num_heads, int seq_len, int head_dim,
    cudaStream_t stream
) {
    const int block_size = 128;
    const dim3 grid(
        (seq_len + block_size - 1) / block_size,
        num_heads,
        batch_size
    );
    const dim3 block(32, 32, 1);

    const size_t shared_mem_size = 3 * block_size * head_dim * sizeof(float) +
                                   block_size * block_size * sizeof(float);

    flash_attention_kernel<float, 128, 64><<<grid, block, shared_mem_size, stream>>>(
        static_cast<const float*>(Q),
        static_cast<const float*>(K),
        static_cast<const float*>(V),
        static_cast<float*>(output),
        scale,
        batch_size,
        num_heads,
        seq_len,
        head_dim
    );
}

} // namespace cuda
} // namespace whisper_turbo