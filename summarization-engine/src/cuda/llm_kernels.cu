#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cub/cub.cuh>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

namespace summarization {
namespace cuda {
namespace llm {

// RoPE (Rotary Position Embeddings) kernel
template<int HEAD_DIM>
__global__ void rotary_position_embedding_kernel(
    float* __restrict__ q,
    float* __restrict__ k,
    int seq_len,
    int num_heads,
    int max_seq_len = 8192) {

    const int head_idx = blockIdx.x;
    const int pos = blockIdx.y;
    const int tid = threadIdx.x;

    if (head_idx >= num_heads || pos >= seq_len) return;

    const int half_dim = HEAD_DIM / 2;

    // Calculate rotation angle
    float theta = powf(10000.0f, -2.0f * (tid % half_dim) / float(HEAD_DIM));
    float angle = pos * theta;
    float cos_theta = cosf(angle);
    float sin_theta = sinf(angle);

    // Apply rotation to Q
    if (tid < HEAD_DIM) {
        float* q_head = q + pos * num_heads * HEAD_DIM + head_idx * HEAD_DIM;

        if (tid < half_dim) {
            float q1 = q_head[tid];
            float q2 = q_head[tid + half_dim];
            q_head[tid] = q1 * cos_theta - q2 * sin_theta;
            q_head[tid + half_dim] = q1 * sin_theta + q2 * cos_theta;
        }
    }

    // Apply rotation to K
    if (tid < HEAD_DIM) {
        float* k_head = k + pos * num_heads * HEAD_DIM + head_idx * HEAD_DIM;

        if (tid < half_dim) {
            float k1 = k_head[tid];
            float k2 = k_head[tid + half_dim];
            k_head[tid] = k1 * cos_theta - k2 * sin_theta;
            k_head[tid + half_dim] = k1 * sin_theta + k2 * cos_theta;
        }
    }
}

// Grouped Query Attention (GQA) kernel
template<int BLOCK_SIZE = 256, int HEAD_DIM = 128>
__global__ void grouped_query_attention_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ output,
    int batch_size,
    int seq_len,
    int num_q_heads,
    int num_kv_heads,
    const float* __restrict__ k_cache = nullptr,
    const float* __restrict__ v_cache = nullptr,
    int cache_len = 0) {

    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int query_idx = blockIdx.x;

    if (batch_idx >= batch_size || head_idx >= num_q_heads || query_idx >= seq_len) {
        return;
    }

    // Shared memory for K,V cache tiles
    extern __shared__ float smem[];
    float* s_k = smem;
    float* s_v = smem + BLOCK_SIZE * HEAD_DIM;

    // Map query head to KV head (for grouped attention)
    const int kv_head_idx = head_idx / (num_q_heads / num_kv_heads);

    // Load query vector
    float q_vec[HEAD_DIM];
    const float* q_ptr = q + batch_idx * seq_len * num_q_heads * HEAD_DIM +
                         query_idx * num_q_heads * HEAD_DIM +
                         head_idx * HEAD_DIM;

    #pragma unroll
    for (int i = 0; i < HEAD_DIM; i++) {
        q_vec[i] = q_ptr[i];
    }

    // Compute attention scores
    float max_score = -INFINITY;
    float sum_exp = 0.0f;

    // Process KV cache if available
    int total_kv_len = cache_len + seq_len;

    for (int kv_start = 0; kv_start < total_kv_len; kv_start += BLOCK_SIZE) {
        int kv_end = min(kv_start + BLOCK_SIZE, total_kv_len);
        int tile_size = kv_end - kv_start;

        // Load K tile into shared memory
        __syncthreads();

        for (int i = threadIdx.x; i < tile_size * HEAD_DIM; i += blockDim.x) {
            int kv_idx = kv_start + i / HEAD_DIM;
            int dim_idx = i % HEAD_DIM;

            if (kv_idx < cache_len && k_cache != nullptr) {
                // Load from cache
                s_k[i] = k_cache[batch_idx * cache_len * num_kv_heads * HEAD_DIM +
                                 kv_idx * num_kv_heads * HEAD_DIM +
                                 kv_head_idx * HEAD_DIM + dim_idx];
            } else if (kv_idx >= cache_len) {
                // Load from current sequence
                int seq_idx = kv_idx - cache_len;
                s_k[i] = k[batch_idx * seq_len * num_kv_heads * HEAD_DIM +
                          seq_idx * num_kv_heads * HEAD_DIM +
                          kv_head_idx * HEAD_DIM + dim_idx];
            }
        }

        __syncthreads();

        // Compute attention scores for this tile
        for (int kv_idx = 0; kv_idx < tile_size; kv_idx++) {
            if (kv_start + kv_idx > query_idx + cache_len) {
                // Causal mask
                continue;
            }

            float score = 0.0f;
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; d++) {
                score += q_vec[d] * s_k[kv_idx * HEAD_DIM + d];
            }
            score /= sqrtf(float(HEAD_DIM));

            max_score = fmaxf(max_score, score);
        }
    }

    // Second pass: compute softmax denominators
    for (int kv_start = 0; kv_start < total_kv_len; kv_start += BLOCK_SIZE) {
        int kv_end = min(kv_start + BLOCK_SIZE, total_kv_len);
        int tile_size = kv_end - kv_start;

        __syncthreads();

        for (int i = threadIdx.x; i < tile_size * HEAD_DIM; i += blockDim.x) {
            int kv_idx = kv_start + i / HEAD_DIM;
            int dim_idx = i % HEAD_DIM;

            if (kv_idx < cache_len && k_cache != nullptr) {
                s_k[i] = k_cache[batch_idx * cache_len * num_kv_heads * HEAD_DIM +
                                kv_idx * num_kv_heads * HEAD_DIM +
                                kv_head_idx * HEAD_DIM + dim_idx];
            } else if (kv_idx >= cache_len) {
                int seq_idx = kv_idx - cache_len;
                s_k[i] = k[batch_idx * seq_len * num_kv_heads * HEAD_DIM +
                          seq_idx * num_kv_heads * HEAD_DIM +
                          kv_head_idx * HEAD_DIM + dim_idx];
            }
        }

        __syncthreads();

        for (int kv_idx = 0; kv_idx < tile_size; kv_idx++) {
            if (kv_start + kv_idx > query_idx + cache_len) continue;

            float score = 0.0f;
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; d++) {
                score += q_vec[d] * s_k[kv_idx * HEAD_DIM + d];
            }
            score /= sqrtf(float(HEAD_DIM));

            sum_exp += expf(score - max_score);
        }
    }

    // Third pass: accumulate weighted values
    float out_vec[HEAD_DIM] = {0.0f};

    for (int kv_start = 0; kv_start < total_kv_len; kv_start += BLOCK_SIZE) {
        int kv_end = min(kv_start + BLOCK_SIZE, total_kv_len);
        int tile_size = kv_end - kv_start;

        __syncthreads();

        // Load K and V tiles
        for (int i = threadIdx.x; i < tile_size * HEAD_DIM; i += blockDim.x) {
            int kv_idx = kv_start + i / HEAD_DIM;
            int dim_idx = i % HEAD_DIM;

            if (kv_idx < cache_len) {
                if (k_cache != nullptr) {
                    s_k[i] = k_cache[batch_idx * cache_len * num_kv_heads * HEAD_DIM +
                                    kv_idx * num_kv_heads * HEAD_DIM +
                                    kv_head_idx * HEAD_DIM + dim_idx];
                }
                if (v_cache != nullptr) {
                    s_v[i] = v_cache[batch_idx * cache_len * num_kv_heads * HEAD_DIM +
                                    kv_idx * num_kv_heads * HEAD_DIM +
                                    kv_head_idx * HEAD_DIM + dim_idx];
                }
            } else {
                int seq_idx = kv_idx - cache_len;
                s_k[i] = k[batch_idx * seq_len * num_kv_heads * HEAD_DIM +
                          seq_idx * num_kv_heads * HEAD_DIM +
                          kv_head_idx * HEAD_DIM + dim_idx];
                s_v[i] = v[batch_idx * seq_len * num_kv_heads * HEAD_DIM +
                          seq_idx * num_kv_heads * HEAD_DIM +
                          kv_head_idx * HEAD_DIM + dim_idx];
            }
        }

        __syncthreads();

        for (int kv_idx = 0; kv_idx < tile_size; kv_idx++) {
            if (kv_start + kv_idx > query_idx + cache_len) continue;

            float score = 0.0f;
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; d++) {
                score += q_vec[d] * s_k[kv_idx * HEAD_DIM + d];
            }
            score /= sqrtf(float(HEAD_DIM));

            float attn_weight = expf(score - max_score) / sum_exp;

            #pragma unroll
            for (int d = 0; d < HEAD_DIM; d++) {
                out_vec[d] += attn_weight * s_v[kv_idx * HEAD_DIM + d];
            }
        }
    }

    // Write output
    float* out_ptr = output + batch_idx * seq_len * num_q_heads * HEAD_DIM +
                     query_idx * num_q_heads * HEAD_DIM +
                     head_idx * HEAD_DIM;

    #pragma unroll
    for (int i = 0; i < HEAD_DIM; i++) {
        out_ptr[i] = out_vec[i];
    }
}

// RMSNorm kernel
template<int BLOCK_SIZE = 256>
__global__ void rms_norm_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ weight,
    int batch_size,
    int seq_len,
    int hidden_dim,
    float epsilon) {

    const int batch_idx = blockIdx.z;
    const int seq_idx = blockIdx.y;
    const int tid = threadIdx.x;

    if (batch_idx >= batch_size || seq_idx >= seq_len) return;

    const float* in_ptr = input + batch_idx * seq_len * hidden_dim +
                         seq_idx * hidden_dim;
    float* out_ptr = output + batch_idx * seq_len * hidden_dim +
                     seq_idx * hidden_dim;

    // Compute RMS
    float sum_sq = 0.0f;
    for (int i = tid; i < hidden_dim; i += BLOCK_SIZE) {
        float val = in_ptr[i];
        sum_sq += val * val;
    }

    // Block reduction
    __shared__ float s_sum[BLOCK_SIZE];
    s_sum[tid] = sum_sq;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
        }
        __syncthreads();
    }

    float rms = sqrtf(s_sum[0] / hidden_dim + epsilon);

    // Normalize and scale
    for (int i = tid; i < hidden_dim; i += BLOCK_SIZE) {
        out_ptr[i] = (in_ptr[i] / rms) * weight[i];
    }
}

// SwiGLU FFN kernel
template<int BLOCK_SIZE = 256>
__global__ void swiglu_ffn_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ w_gate,
    const float* __restrict__ w_up,
    const float* __restrict__ w_down,
    int batch_size,
    int seq_len,
    int hidden_dim,
    int ffn_dim) {

    const int global_idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const int total_elements = batch_size * seq_len * hidden_dim;

    if (global_idx >= total_elements) return;

    const int batch_idx = global_idx / (seq_len * hidden_dim);
    const int seq_idx = (global_idx % (seq_len * hidden_dim)) / hidden_dim;
    const int hidden_idx = global_idx % hidden_dim;

    const float* in_ptr = input + batch_idx * seq_len * hidden_dim +
                         seq_idx * hidden_dim;

    // Allocate intermediate activations
    extern __shared__ float s_mem[];
    float* gate_out = s_mem;
    float* up_out = s_mem + ffn_dim;

    // Gate projection with SiLU activation
    float gate_sum = 0.0f;
    for (int i = 0; i < hidden_dim; i++) {
        gate_sum += in_ptr[i] * w_gate[i * ffn_dim + threadIdx.x];
    }
    float gate_act = gate_sum / (1.0f + expf(-gate_sum));  // SiLU

    // Up projection
    float up_sum = 0.0f;
    for (int i = 0; i < hidden_dim; i++) {
        up_sum += in_ptr[i] * w_up[i * ffn_dim + threadIdx.x];
    }

    // Element-wise product
    float ffn_out = gate_act * up_sum;

    // Down projection
    __syncthreads();

    if (threadIdx.x < hidden_dim) {
        float down_sum = 0.0f;
        for (int i = 0; i < ffn_dim; i++) {
            down_sum += ffn_out * w_down[i * hidden_dim + threadIdx.x];
        }

        output[global_idx] = down_sum;
    }
}

// INT4 dequantization and matmul kernel
__global__ void int4_dequantize_matmul_kernel(
    const uint8_t* __restrict__ weight_int4,
    const float* __restrict__ scales,
    const float* __restrict__ zeros,
    const float* __restrict__ input,
    float* __restrict__ output,
    int m, int n, int k,
    int group_size = 128) {

    const int row = blockIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= m || col >= n) return;

    float sum = 0.0f;

    for (int i = 0; i < k; i++) {
        // Unpack INT4 weight
        int weight_idx = row * k + i;
        int byte_idx = weight_idx / 2;
        int nibble = weight_idx % 2;

        uint8_t packed = weight_int4[byte_idx];
        int4_t weight_val = (nibble == 0) ?
            (packed & 0x0F) - 8 :
            ((packed >> 4) & 0x0F) - 8;

        // Dequantize
        int group_idx = i / group_size;
        float scale = scales[row * ((k + group_size - 1) / group_size) + group_idx];
        float zero = zeros[row * ((k + group_size - 1) / group_size) + group_idx];
        float weight_fp32 = weight_val * scale + zero;

        // Multiply-accumulate
        sum += input[col * k + i] * weight_fp32;
    }

    output[col * n + row] = sum;
}

// Update KV cache kernel
__global__ void update_kv_cache_kernel(
    float* __restrict__ cache_k,
    float* __restrict__ cache_v,
    const float* __restrict__ new_k,
    const float* __restrict__ new_v,
    int cache_size,
    int update_size,
    int hidden_dim) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= update_size * hidden_dim) return;

    // Shift existing cache
    if (idx < (cache_size - update_size) * hidden_dim) {
        cache_k[idx] = cache_k[idx + update_size * hidden_dim];
        cache_v[idx] = cache_v[idx + update_size * hidden_dim];
    }

    // Add new entries
    int new_idx = idx - (cache_size - update_size) * hidden_dim;
    if (new_idx >= 0 && new_idx < update_size * hidden_dim) {
        cache_k[(cache_size - update_size) * hidden_dim + new_idx] = new_k[new_idx];
        cache_v[(cache_size - update_size) * hidden_dim + new_idx] = new_v[new_idx];
    }
}

// Host wrapper functions
void rotary_position_embedding(float* q, float* k, int seq_len,
                              int num_heads, int head_dim,
                              cudaStream_t stream) {
    dim3 blocks(num_heads, seq_len);
    dim3 threads(head_dim);

    if (head_dim == 64) {
        rotary_position_embedding_kernel<64><<<blocks, threads, 0, stream>>>(
            q, k, seq_len, num_heads
        );
    } else if (head_dim == 128) {
        rotary_position_embedding_kernel<128><<<blocks, threads, 0, stream>>>(
            q, k, seq_len, num_heads
        );
    }
}

void grouped_query_attention(const float* q, const float* k, const float* v,
                            float* output, int batch_size, int seq_len,
                            int num_q_heads, int num_kv_heads, int head_dim,
                            cudaStream_t stream) {
    dim3 blocks(seq_len, num_q_heads, batch_size);
    dim3 threads(256);

    size_t smem_size = 2 * 256 * head_dim * sizeof(float);

    if (head_dim == 128) {
        grouped_query_attention_kernel<256, 128><<<blocks, threads, smem_size, stream>>>(
            q, k, v, output, batch_size, seq_len, num_q_heads, num_kv_heads
        );
    }
}

void rms_norm(const float* input, float* output, const float* weight,
             int batch_size, int seq_len, int hidden_dim,
             float epsilon, cudaStream_t stream) {
    dim3 blocks(1, seq_len, batch_size);
    dim3 threads(256);

    rms_norm_kernel<256><<<blocks, threads, 0, stream>>>(
        input, output, weight, batch_size, seq_len, hidden_dim, epsilon
    );
}

void swiglu_ffn(const float* input, float* output,
               const float* w1, const float* w2, const float* w3,
               int batch_size, int seq_len, int hidden_dim, int ffn_dim,
               cudaStream_t stream) {
    int total_elements = batch_size * seq_len * hidden_dim;
    int blocks = (total_elements + 255) / 256;

    size_t smem_size = 2 * ffn_dim * sizeof(float);

    swiglu_ffn_kernel<256><<<blocks, 256, smem_size, stream>>>(
        input, output, w1, w2, w3,
        batch_size, seq_len, hidden_dim, ffn_dim
    );
}

void int4_dequantize_matmul(const uint8_t* weight_int4, const float* scales,
                            const float* zeros, const float* input,
                            float* output, int m, int n, int k,
                            cudaStream_t stream) {
    dim3 blocks((n + 31) / 32, m);
    dim3 threads(32);

    int4_dequantize_matmul_kernel<<<blocks, threads, 0, stream>>>(
        weight_int4, scales, zeros, input, output, m, n, k
    );
}

void update_kv_cache(float* cache_k, float* cache_v,
                    const float* new_k, const float* new_v,
                    int cache_size, int update_size,
                    cudaStream_t stream) {
    int hidden_dim = 128;  // Would be passed as parameter
    int total_elements = cache_size * hidden_dim;
    int blocks = (total_elements + 255) / 256;

    update_kv_cache_kernel<<<blocks, 256, 0, stream>>>(
        cache_k, cache_v, new_k, new_v,
        cache_size, update_size, hidden_dim
    );
}

} // namespace llm
} // namespace cuda
} // namespace summarization