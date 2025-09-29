#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <cub/cub.cuh>
#include <cooperative_groups.h>

namespace whisper_turbo {
namespace cuda {
namespace fused {

namespace cg = cooperative_groups;

// =================================================================
// Fused LayerNorm + Attention + Residual Connection
// =================================================================

template<typename T, int HIDDEN_DIM, int NUM_HEADS>
__global__ void fused_transformer_encoder_layer(
    const T* __restrict__ input,           // [batch, seq_len, hidden_dim]
    const T* __restrict__ ln1_gamma,       // [hidden_dim]
    const T* __restrict__ ln1_beta,        // [hidden_dim]
    const T* __restrict__ q_weight,        // [hidden_dim, hidden_dim]
    const T* __restrict__ k_weight,        // [hidden_dim, hidden_dim]
    const T* __restrict__ v_weight,        // [hidden_dim, hidden_dim]
    const T* __restrict__ out_weight,      // [hidden_dim, hidden_dim]
    const T* __restrict__ ln2_gamma,       // [hidden_dim]
    const T* __restrict__ ln2_beta,        // [hidden_dim]
    const T* __restrict__ ffn_weight1,     // [hidden_dim, 4*hidden_dim]
    const T* __restrict__ ffn_weight2,     // [4*hidden_dim, hidden_dim]
    T* __restrict__ output,                // [batch, seq_len, hidden_dim]
    T* __restrict__ kv_cache_k,           // [batch, num_heads, cache_len, head_dim]
    T* __restrict__ kv_cache_v,           // [batch, num_heads, cache_len, head_dim]
    const int batch_size,
    const int seq_len,
    const int cache_offset,
    const float attention_scale
) {
    // Shared memory for intermediate results
    extern __shared__ char shared_mem[];
    T* s_ln_out = reinterpret_cast<T*>(shared_mem);
    T* s_qkv = reinterpret_cast<T*>(&s_ln_out[HIDDEN_DIM]);
    float* s_attn_scores = reinterpret_cast<float*>(&s_qkv[3 * HIDDEN_DIM]);

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int batch_id = blockIdx.z;

    const int head_dim = HIDDEN_DIM / NUM_HEADS;
    const int seq_pos = bid;

    if (seq_pos >= seq_len) return;

    // Cooperative groups for warp-level operations
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    auto block = cg::this_thread_block();

    // ========== Layer Norm 1 ==========
    // Compute mean and variance for LayerNorm
    float local_sum = 0.0f;
    float local_sq_sum = 0.0f;

    for (int i = tid; i < HIDDEN_DIM; i += blockDim.x) {
        int idx = batch_id * seq_len * HIDDEN_DIM + seq_pos * HIDDEN_DIM + i;
        float val = float(input[idx]);
        local_sum += val;
        local_sq_sum += val * val;
    }

    // Warp-level reduction for mean and variance
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
        local_sq_sum += __shfl_down_sync(0xFFFFFFFF, local_sq_sum, offset);
    }

    // Broadcast mean and variance
    __shared__ float s_mean, s_var;
    if (warp.thread_rank() == 0) {
        s_mean = local_sum / HIDDEN_DIM;
        s_var = (local_sq_sum / HIDDEN_DIM) - (s_mean * s_mean);
    }
    block.sync();

    float mean = s_mean;
    float inv_std = rsqrtf(s_var + 1e-5f);

    // Apply LayerNorm and store in shared memory
    for (int i = tid; i < HIDDEN_DIM; i += blockDim.x) {
        int idx = batch_id * seq_len * HIDDEN_DIM + seq_pos * HIDDEN_DIM + i;
        float normalized = (float(input[idx]) - mean) * inv_std;
        s_ln_out[i] = T(normalized * float(ln1_gamma[i]) + float(ln1_beta[i]));
    }
    block.sync();

    // ========== Fused QKV Projection ==========
    // Compute Q, K, V projections in parallel
    for (int head = 0; head < NUM_HEADS; head++) {
        int head_offset = head * head_dim;

        // Each thread computes one element of Q, K, or V
        if (tid < head_dim) {
            float q_val = 0.0f, k_val = 0.0f, v_val = 0.0f;

            #pragma unroll
            for (int i = 0; i < HIDDEN_DIM; i++) {
                float x = float(s_ln_out[i]);
                q_val += x * float(q_weight[i * HIDDEN_DIM + head_offset + tid]);
                k_val += x * float(k_weight[i * HIDDEN_DIM + head_offset + tid]);
                v_val += x * float(v_weight[i * HIDDEN_DIM + head_offset + tid]);
            }

            // Store in shared memory
            s_qkv[tid] = T(q_val);
            s_qkv[HIDDEN_DIM + tid] = T(k_val);
            s_qkv[2 * HIDDEN_DIM + tid] = T(v_val);

            // Update KV cache
            int cache_idx = ((batch_id * NUM_HEADS + head) * (cache_offset + seq_len) +
                           cache_offset + seq_pos) * head_dim + tid;
            kv_cache_k[cache_idx] = T(k_val);
            kv_cache_v[cache_idx] = T(v_val);
        }
    }
    block.sync();

    // ========== Fused Self-Attention ==========
    // Compute attention scores and apply softmax
    for (int head = 0; head < NUM_HEADS; head++) {
        if (tid < seq_len) {
            // Compute Q @ K^T for current position
            float score = 0.0f;
            int q_offset = head * head_dim;

            #pragma unroll
            for (int d = 0; d < head_dim; d++) {
                float q = float(s_qkv[q_offset + d]);
                int k_cache_idx = ((batch_id * NUM_HEADS + head) *
                                 (cache_offset + seq_len) + tid) * head_dim + d;
                float k = float(kv_cache_k[k_cache_idx]);
                score += q * k;
            }

            s_attn_scores[tid] = score * attention_scale;
        }
    }
    block.sync();

    // Online softmax with numerical stability
    float max_score = -INFINITY;
    if (tid < seq_len) {
        max_score = s_attn_scores[tid];
    }

    // Find max score
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        max_score = fmaxf(max_score, __shfl_down_sync(0xFFFFFFFF, max_score, offset));
    }
    max_score = __shfl_sync(0xFFFFFFFF, max_score, 0);

    // Compute exp and sum
    float exp_sum = 0.0f;
    if (tid < seq_len) {
        float exp_score = expf(s_attn_scores[tid] - max_score);
        s_attn_scores[tid] = exp_score;
        exp_sum = exp_score;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        exp_sum += __shfl_down_sync(0xFFFFFFFF, exp_sum, offset);
    }
    exp_sum = __shfl_sync(0xFFFFFFFF, exp_sum, 0);

    // Normalize
    if (tid < seq_len) {
        s_attn_scores[tid] /= exp_sum;
    }
    block.sync();

    // ========== Attention Output ==========
    // Compute attention output: scores @ V
    for (int head = 0; head < NUM_HEADS; head++) {
        int head_offset = head * head_dim;

        if (tid < head_dim) {
            float out_val = 0.0f;

            #pragma unroll
            for (int s = 0; s < seq_len; s++) {
                float score = s_attn_scores[s];
                int v_cache_idx = ((batch_id * NUM_HEADS + head) *
                                 (cache_offset + seq_len) + s) * head_dim + tid;
                float v = float(kv_cache_v[v_cache_idx]);
                out_val += score * v;
            }

            // Project output
            float projected = 0.0f;
            for (int i = 0; i < HIDDEN_DIM; i++) {
                projected += out_val * float(out_weight[head_offset * HIDDEN_DIM + i]);
            }

            // Residual connection
            int idx = batch_id * seq_len * HIDDEN_DIM + seq_pos * HIDDEN_DIM + head_offset + tid;
            s_ln_out[head_offset + tid] = T(projected + float(input[idx]));
        }
    }
    block.sync();

    // ========== Layer Norm 2 + FFN ==========
    // Second LayerNorm
    local_sum = 0.0f;
    local_sq_sum = 0.0f;

    for (int i = tid; i < HIDDEN_DIM; i += blockDim.x) {
        float val = float(s_ln_out[i]);
        local_sum += val;
        local_sq_sum += val * val;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
        local_sq_sum += __shfl_down_sync(0xFFFFFFFF, local_sq_sum, offset);
    }

    if (warp.thread_rank() == 0) {
        s_mean = local_sum / HIDDEN_DIM;
        s_var = (local_sq_sum / HIDDEN_DIM) - (s_mean * s_mean);
    }
    block.sync();

    mean = s_mean;
    inv_std = rsqrtf(s_var + 1e-5f);

    // Apply FFN with GELU activation (fused)
    for (int i = tid; i < HIDDEN_DIM; i += blockDim.x) {
        float normalized = (float(s_ln_out[i]) - mean) * inv_std;
        float ln2_out = normalized * float(ln2_gamma[i]) + float(ln2_beta[i]);

        // FFN first layer with 4x expansion
        float ffn_hidden = 0.0f;
        #pragma unroll
        for (int j = 0; j < HIDDEN_DIM; j++) {
            ffn_hidden += ln2_out * float(ffn_weight1[j * 4 * HIDDEN_DIM + i]);
        }

        // GELU activation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        const float c1 = 0.7978845608f;  // sqrt(2/pi)
        const float c2 = 0.044715f;
        float x3 = ffn_hidden * ffn_hidden * ffn_hidden;
        float gelu_out = 0.5f * ffn_hidden *
                        (1.0f + tanhf(c1 * (ffn_hidden + c2 * x3)));

        // FFN second layer
        float ffn_out = 0.0f;
        #pragma unroll
        for (int j = 0; j < 4 * HIDDEN_DIM; j++) {
            ffn_out += gelu_out * float(ffn_weight2[j * HIDDEN_DIM + i]);
        }

        // Final residual connection
        int idx = batch_id * seq_len * HIDDEN_DIM + seq_pos * HIDDEN_DIM + i;
        output[idx] = T(ffn_out + float(s_ln_out[i]));
    }
}

// =================================================================
// Fused Decoder Layer with Cross-Attention
// =================================================================

template<typename T, int HIDDEN_DIM, int NUM_HEADS>
__global__ void fused_transformer_decoder_layer(
    const T* __restrict__ input,           // [batch, tgt_len, hidden_dim]
    const T* __restrict__ encoder_output,   // [batch, src_len, hidden_dim]
    const T* __restrict__ self_attn_mask,   // [batch, tgt_len, tgt_len]
    const T* __restrict__ cross_attn_mask,  // [batch, tgt_len, src_len]
    T* __restrict__ output,                // [batch, tgt_len, hidden_dim]
    const T* __restrict__ weights,         // All layer weights concatenated
    T* __restrict__ kv_cache_self_k,
    T* __restrict__ kv_cache_self_v,
    T* __restrict__ kv_cache_cross_k,
    T* __restrict__ kv_cache_cross_v,
    const int batch_size,
    const int tgt_len,
    const int src_len,
    const int cache_offset
) {
    // Complex decoder implementation with self-attention and cross-attention
    // This would follow similar pattern as encoder but with two attention mechanisms
}

// =================================================================
// Fused Beam Search Decoding
// =================================================================

template<int VOCAB_SIZE, int BEAM_SIZE, int MAX_LENGTH>
__global__ void fused_beam_search_step(
    const float* __restrict__ logits,      // [batch, beam_size, vocab_size]
    int* __restrict__ beam_tokens,         // [batch, beam_size, max_length]
    float* __restrict__ beam_scores,       // [batch, beam_size]
    int* __restrict__ beam_indices,        // [batch, beam_size]
    bool* __restrict__ beam_finished,      // [batch, beam_size]
    const int batch_size,
    const int current_step,
    const float length_penalty,
    const float temperature,
    const int eos_token_id
) {
    // Shared memory for top-k selection
    __shared__ float s_top_scores[BEAM_SIZE * 2];
    __shared__ int s_top_indices[BEAM_SIZE * 2];

    const int tid = threadIdx.x;
    const int batch_id = blockIdx.x;
    const int beam_id = blockIdx.y;

    if (batch_id >= batch_size || beam_id >= BEAM_SIZE) return;
    if (beam_finished[batch_id * BEAM_SIZE + beam_id]) return;

    // Apply temperature
    float* beam_logits = const_cast<float*>(&logits[(batch_id * BEAM_SIZE + beam_id) * VOCAB_SIZE]);

    if (temperature != 1.0f && tid < VOCAB_SIZE) {
        beam_logits[tid] /= temperature;
    }
    __syncthreads();

    // Compute log softmax (numerically stable)
    float max_logit = -INFINITY;
    for (int i = tid; i < VOCAB_SIZE; i += blockDim.x) {
        max_logit = fmaxf(max_logit, beam_logits[i]);
    }

    // Reduce max across block
    typedef cub::BlockReduce<float, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    max_logit = BlockReduce(temp_storage).Reduce(max_logit, cub::Max());

    if (tid == 0) {
        s_top_scores[0] = max_logit;
    }
    __syncthreads();
    max_logit = s_top_scores[0];

    // Compute exp and sum
    float exp_sum = 0.0f;
    for (int i = tid; i < VOCAB_SIZE; i += blockDim.x) {
        float exp_val = expf(beam_logits[i] - max_logit);
        beam_logits[i] = exp_val;
        exp_sum += exp_val;
    }

    exp_sum = BlockReduce(temp_storage).Sum(exp_sum);

    if (tid == 0) {
        s_top_scores[1] = exp_sum;
    }
    __syncthreads();
    exp_sum = s_top_scores[1];

    // Normalize to get probabilities and convert to log probs
    for (int i = tid; i < VOCAB_SIZE; i += blockDim.x) {
        beam_logits[i] = logf(beam_logits[i] / exp_sum);
    }
    __syncthreads();

    // Find top 2*BEAM_SIZE candidates using parallel reduction
    if (tid < 2 * BEAM_SIZE) {
        float best_score = -INFINITY;
        int best_idx = -1;

        for (int i = tid; i < VOCAB_SIZE; i += 2 * BEAM_SIZE) {
            float score = beam_logits[i] + beam_scores[batch_id * BEAM_SIZE + beam_id];

            // Apply length penalty
            float length_factor = powf(5.0f + current_step + 1, length_penalty) /
                                powf(5.0f + 1, length_penalty);
            score /= length_factor;

            if (score > best_score) {
                best_score = score;
                best_idx = i;
            }
        }

        s_top_scores[tid] = best_score;
        s_top_indices[tid] = best_idx;
    }
    __syncthreads();

    // Final selection of top BEAM_SIZE beams
    if (tid == 0) {
        // Sort candidates and select top BEAM_SIZE
        for (int i = 0; i < BEAM_SIZE; i++) {
            int max_idx = i;
            for (int j = i + 1; j < 2 * BEAM_SIZE; j++) {
                if (s_top_scores[j] > s_top_scores[max_idx]) {
                    max_idx = j;
                }
            }

            if (max_idx != i) {
                // Swap
                float tmp_score = s_top_scores[i];
                int tmp_idx = s_top_indices[i];
                s_top_scores[i] = s_top_scores[max_idx];
                s_top_indices[i] = s_top_indices[max_idx];
                s_top_scores[max_idx] = tmp_score;
                s_top_indices[max_idx] = tmp_idx;
            }

            // Update beam
            int beam_offset = batch_id * BEAM_SIZE + i;
            beam_scores[beam_offset] = s_top_scores[i];
            beam_tokens[beam_offset * MAX_LENGTH + current_step] = s_top_indices[i];

            // Check for EOS
            if (s_top_indices[i] == eos_token_id) {
                beam_finished[beam_offset] = true;
            }
        }
    }
}

// =================================================================
// Fused Mel Spectrogram + Log + Normalization
// =================================================================

__global__ void fused_mel_spectrogram_processing(
    const float* __restrict__ stft_real,    // [batch, n_fft/2+1, n_frames]
    const float* __restrict__ stft_imag,    // [batch, n_fft/2+1, n_frames]
    const float* __restrict__ mel_filters,  // [n_mels, n_fft/2+1]
    float* __restrict__ mel_output,        // [batch, n_mels, n_frames]
    const float* __restrict__ mean,        // [n_mels]
    const float* __restrict__ std,         // [n_mels]
    const int batch_size,
    const int n_fft,
    const int n_mels,
    const int n_frames,
    const float log_offset = 1e-10f
) {
    const int batch_id = blockIdx.z;
    const int mel_id = blockIdx.y;
    const int frame_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_id >= batch_size || mel_id >= n_mels || frame_id >= n_frames) {
        return;
    }

    // Compute power spectrum and apply mel filter
    float mel_energy = 0.0f;
    const int n_bins = n_fft / 2 + 1;

    #pragma unroll
    for (int bin = 0; bin < n_bins; bin++) {
        int stft_idx = batch_id * n_bins * n_frames + bin * n_frames + frame_id;

        // Power spectrum: real^2 + imag^2
        float power = stft_real[stft_idx] * stft_real[stft_idx] +
                     stft_imag[stft_idx] * stft_imag[stft_idx];

        // Apply mel filter
        mel_energy += power * mel_filters[mel_id * n_bins + bin];
    }

    // Apply log
    mel_energy = logf(mel_energy + log_offset);

    // Normalize with mean and std
    mel_energy = (mel_energy - mean[mel_id]) / std[mel_id];

    // Store result
    int out_idx = batch_id * n_mels * n_frames + mel_id * n_frames + frame_id;
    mel_output[out_idx] = mel_energy;
}

// =================================================================
// CUDA Graph Capture for Inference Pipeline
// =================================================================

class CUDAGraphExecutor {
private:
    cudaGraph_t graph_;
    cudaGraphExec_t graph_exec_;
    cudaStream_t stream_;
    bool captured_;

    // Pre-allocated buffers for graph
    void* buffers_[32];
    size_t buffer_sizes_[32];
    int num_buffers_;

public:
    CUDAGraphExecutor() : graph_(nullptr), graph_exec_(nullptr), captured_(false), num_buffers_(0) {
        cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
    }

    ~CUDAGraphExecutor() {
        if (graph_exec_) cudaGraphExecDestroy(graph_exec_);
        if (graph_) cudaGraphDestroy(graph_);
        cudaStreamDestroy(stream_);

        for (int i = 0; i < num_buffers_; i++) {
            cudaFree(buffers_[i]);
        }
    }

    template<typename Func>
    void CaptureGraph(Func inference_function) {
        // Start capture
        cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal);

        // Run the inference function
        inference_function(stream_);

        // End capture
        cudaStreamEndCapture(stream_, &graph_);

        // Create executable graph
        cudaGraphInstantiate(&graph_exec_, graph_, nullptr, nullptr, 0);

        captured_ = true;
    }

    void LaunchGraph() {
        if (!captured_) {
            throw std::runtime_error("Graph not captured");
        }
        cudaGraphLaunch(graph_exec_, stream_);
    }

    void Synchronize() {
        cudaStreamSynchronize(stream_);
    }

    void* AllocateBuffer(size_t size) {
        if (num_buffers_ >= 32) {
            throw std::runtime_error("Too many buffers");
        }

        void* ptr;
        cudaMalloc(&ptr, size);
        buffers_[num_buffers_] = ptr;
        buffer_sizes_[num_buffers_] = size;
        num_buffers_++;

        return ptr;
    }

    // Update graph with new parameters without recapture
    void UpdateGraph(cudaGraphExec_t new_exec) {
        if (graph_exec_) {
            cudaGraphExecDestroy(graph_exec_);
        }
        graph_exec_ = new_exec;
    }
};

// =================================================================
// Optimized Memory Pool for KV Cache
// =================================================================

class KVCacheMemoryPool {
private:
    struct MemoryBlock {
        void* ptr;
        size_t size;
        bool in_use;
    };

    std::vector<MemoryBlock> blocks_;
    size_t total_allocated_;
    size_t peak_usage_;
    cudaStream_t stream_;

public:
    KVCacheMemoryPool() : total_allocated_(0), peak_usage_(0) {
        cudaStreamCreate(&stream_);
    }

    ~KVCacheMemoryPool() {
        for (auto& block : blocks_) {
            if (block.ptr) {
                cudaFree(block.ptr);
            }
        }
        cudaStreamDestroy(stream_);
    }

    void* Allocate(size_t size) {
        // Try to find a free block of sufficient size
        for (auto& block : blocks_) {
            if (!block.in_use && block.size >= size) {
                block.in_use = true;
                return block.ptr;
            }
        }

        // Allocate new block
        void* ptr;
        cudaMalloc(&ptr, size);
        blocks_.push_back({ptr, size, true});

        total_allocated_ += size;
        peak_usage_ = std::max(peak_usage_, total_allocated_);

        return ptr;
    }

    void Free(void* ptr) {
        for (auto& block : blocks_) {
            if (block.ptr == ptr) {
                block.in_use = false;

                // Optionally clear memory
                cudaMemsetAsync(ptr, 0, block.size, stream_);
                return;
            }
        }
    }

    void Reset() {
        for (auto& block : blocks_) {
            block.in_use = false;
        }
    }

    size_t GetPeakUsage() const { return peak_usage_; }
    size_t GetTotalAllocated() const { return total_allocated_; }
};

// =================================================================
// Launch Functions
// =================================================================

void launch_fused_transformer_encoder(
    const void* input,
    const void* weights,
    void* output,
    void* kv_cache_k,
    void* kv_cache_v,
    int batch_size,
    int seq_len,
    int hidden_dim,
    int num_heads,
    int num_layers,
    cudaStream_t stream
) {
    const int block_size = 256;
    const int grid_size = seq_len;

    // Calculate shared memory size
    size_t shared_mem_size = (3 * hidden_dim * sizeof(float)) +
                            (seq_len * seq_len * sizeof(float));

    for (int layer = 0; layer < num_layers; layer++) {
        dim3 grid(grid_size, 1, batch_size);
        dim3 block(block_size);

        // Launch fused kernel for each layer
        if (hidden_dim == 768 && num_heads == 12) {
            fused_transformer_encoder_layer<float, 768, 12><<<grid, block, shared_mem_size, stream>>>(
                static_cast<const float*>(input),
                nullptr, nullptr,  // Layer weights would be passed properly
                nullptr, nullptr, nullptr, nullptr,
                nullptr, nullptr,
                nullptr, nullptr,
                static_cast<float*>(output),
                static_cast<float*>(kv_cache_k),
                static_cast<float*>(kv_cache_v),
                batch_size, seq_len, 0, 1.0f / sqrtf(768.0f / 12)
            );
        }
    }
}

void launch_fused_beam_search(
    const float* logits,
    int* beam_tokens,
    float* beam_scores,
    int batch_size,
    int beam_size,
    int vocab_size,
    int current_step,
    cudaStream_t stream
) {
    dim3 grid(batch_size, beam_size);
    dim3 block(256);

    if (vocab_size == 51865 && beam_size == 5) {
        fused_beam_search_step<51865, 5, 448><<<grid, block, 0, stream>>>(
            logits,
            beam_tokens,
            beam_scores,
            nullptr,  // beam_indices
            nullptr,  // beam_finished
            batch_size,
            current_step,
            0.6f,     // length_penalty
            1.0f,     // temperature
            50257     // EOS token ID
        );
    }
}

} // namespace fused
} // namespace cuda
} // namespace whisper_turbo