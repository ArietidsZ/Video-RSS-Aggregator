#include <memory>
#include <string>
#include <vector>
#include <queue>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cublasLt.h>

namespace summarization {

// Model configurations for different LLMs
enum class LLMModel {
    Llama3_8B,
    Llama3_70B,
    Mistral_7B,
    Mixtral_8x7B,
    Qwen2_7B,
    Qwen2_72B,
    Yi_6B,
    Yi_34B,
    DeepSeek_7B,
    DeepSeek_67B,
    Custom
};

struct LLMConfig {
    LLMModel model;
    int max_context_length;
    int max_output_tokens;
    float temperature;
    float top_p;
    int top_k;
    float repetition_penalty;
    bool use_flash_attention;
    bool use_kv_cache;
    bool use_int8_quantization;
    bool use_int4_quantization;
    int batch_size;
    std::string model_path;
    std::string tokenizer_path;
};

// Forward declarations for CUDA kernels
namespace cuda {
namespace llm {
    void rotary_position_embedding(float* q, float* k, int seq_len,
                                  int num_heads, int head_dim,
                                  cudaStream_t stream);

    void grouped_query_attention(const float* q, const float* k, const float* v,
                                float* output, int batch_size, int seq_len,
                                int num_q_heads, int num_kv_heads, int head_dim,
                                cudaStream_t stream);

    void rms_norm(const float* input, float* output, const float* weight,
                 int batch_size, int seq_len, int hidden_dim,
                 float epsilon, cudaStream_t stream);

    void swiglu_ffn(const float* input, float* output,
                   const float* w1, const float* w2, const float* w3,
                   int batch_size, int seq_len, int hidden_dim, int ffn_dim,
                   cudaStream_t stream);

    void int4_dequantize_matmul(const uint8_t* weight_int4, const float* scales,
                                const float* zeros, const float* input,
                                float* output, int m, int n, int k,
                                cudaStream_t stream);
}
}

class LLMEngine {
private:
    struct ModelWeights {
        // Token embeddings
        float* token_embeddings;      // [vocab_size, hidden_dim]

        // Transformer layers
        struct TransformerLayer {
            float* q_proj;             // [hidden_dim, hidden_dim]
            float* k_proj;             // [hidden_dim, kv_dim]
            float* v_proj;             // [hidden_dim, kv_dim]
            float* o_proj;             // [hidden_dim, hidden_dim]
            float* gate_proj;          // [hidden_dim, ffn_dim]
            float* up_proj;            // [hidden_dim, ffn_dim]
            float* down_proj;          // [ffn_dim, hidden_dim]
            float* input_norm;         // [hidden_dim]
            float* post_attention_norm; // [hidden_dim]
        };
        std::vector<TransformerLayer> layers;

        // Output
        float* output_norm;           // [hidden_dim]
        float* lm_head;               // [hidden_dim, vocab_size]

        // Quantized weights (optional)
        struct QuantizedWeights {
            uint8_t* q_proj_int4;
            float* q_proj_scales;
            float* q_proj_zeros;
            // ... other quantized weights
        };
        std::unique_ptr<QuantizedWeights> quantized;
    };

    struct KVCache {
        float* k_cache;  // [num_layers, batch_size, max_seq_len, kv_dim]
        float* v_cache;  // [num_layers, batch_size, max_seq_len, kv_dim]
        int* seq_lengths;  // Current sequence length for each batch
        int max_seq_len;
        int num_layers;
        int batch_size;
        int kv_dim;
    };

    LLMConfig config_;
    std::unique_ptr<ModelWeights> weights_;
    std::unique_ptr<KVCache> kv_cache_;

    // CUDA resources
    cudaStream_t stream_;
    cublasLtHandle_t cublaslt_handle_;
    cudnnHandle_t cudnn_handle_;

    // Memory pools
    float* workspace_;
    size_t workspace_size_;

    // Tokenizer
    std::unique_ptr<class Tokenizer> tokenizer_;

public:
    LLMEngine(const LLMConfig& config) : config_(config) {
        InitializeCUDA();
        LoadModel();
        InitializeKVCache();
        LoadTokenizer();
    }

    ~LLMEngine() {
        FreeModelWeights();
        FreeKVCache();
        if (workspace_) cudaFree(workspace_);
        cublasLtDestroy(cublaslt_handle_);
        cudnnDestroy(cudnn_handle_);
        cudaStreamDestroy(stream_);
    }

    struct GenerationResult {
        std::string text;
        std::vector<float> token_probs;
        int num_tokens;
        float avg_logprob;
        double generation_time_ms;
    };

    GenerationResult Generate(const std::string& prompt,
                             int max_tokens = -1,
                             float temperature = -1.0f) {
        auto start = std::chrono::high_resolution_clock::now();

        if (max_tokens <= 0) max_tokens = config_.max_output_tokens;
        if (temperature < 0) temperature = config_.temperature;

        // Tokenize prompt
        std::vector<int> input_tokens = tokenizer_->Encode(prompt);

        // Prepare batch (single sequence for now)
        int batch_size = 1;
        int seq_len = input_tokens.size();

        // Allocate device memory
        int* d_tokens;
        float* d_logits;
        cudaMalloc(&d_tokens, (seq_len + max_tokens) * sizeof(int));
        cudaMalloc(&d_logits, config_.model == LLMModel::Llama3_70B ?
                   128256 * sizeof(float) : 32000 * sizeof(float));

        // Copy input tokens
        cudaMemcpyAsync(d_tokens, input_tokens.data(),
                       seq_len * sizeof(int),
                       cudaMemcpyHostToDevice, stream_);

        // Generate tokens
        std::vector<int> output_tokens;
        std::vector<float> token_probs;

        for (int i = 0; i < max_tokens; i++) {
            // Forward pass
            ForwardPass(d_tokens, seq_len + i, d_logits);

            // Sample next token
            int next_token = SampleToken(d_logits, temperature,
                                        config_.top_p, config_.top_k);

            // Check for end token
            if (next_token == tokenizer_->GetEOSToken()) {
                break;
            }

            // Add to sequence
            output_tokens.push_back(next_token);
            cudaMemcpyAsync(d_tokens + seq_len + i, &next_token,
                           sizeof(int), cudaMemcpyHostToDevice, stream_);

            // Update KV cache position
            if (kv_cache_) {
                kv_cache_->seq_lengths[0] = seq_len + i + 1;
            }
        }

        // Decode tokens to text
        GenerationResult result;
        result.text = tokenizer_->Decode(output_tokens);
        result.token_probs = token_probs;
        result.num_tokens = output_tokens.size();

        // Calculate metrics
        auto end = std::chrono::high_resolution_clock::now();
        result.generation_time_ms = std::chrono::duration<double, std::milli>(
            end - start).count();

        // Cleanup
        cudaFree(d_tokens);
        cudaFree(d_logits);

        return result;
    }

    std::string Summarize(const std::string& text,
                         const std::string& style = "concise") {
        // Build summarization prompt based on style
        std::string prompt = BuildSummarizationPrompt(text, style);

        // Generate summary
        auto result = Generate(prompt, 500, 0.7f);

        // Post-process summary
        return PostProcessSummary(result.text, style);
    }

private:
    void InitializeCUDA() {
        cudaStreamCreate(&stream_);
        cublasLtCreate(&cublaslt_handle_);
        cudnnCreate(&cudnn_handle_);

        // Allocate workspace
        workspace_size_ = 2ULL * 1024 * 1024 * 1024;  // 2GB workspace
        cudaMalloc(&workspace_, workspace_size_);
    }

    void LoadModel() {
        weights_ = std::make_unique<ModelWeights>();

        // Model dimensions based on config
        int hidden_dim, ffn_dim, num_layers, num_heads, num_kv_heads, vocab_size;

        switch (config_.model) {
            case LLMModel::Llama3_8B:
                hidden_dim = 4096;
                ffn_dim = 14336;
                num_layers = 32;
                num_heads = 32;
                num_kv_heads = 8;
                vocab_size = 128256;
                break;
            case LLMModel::Llama3_70B:
                hidden_dim = 8192;
                ffn_dim = 28672;
                num_layers = 80;
                num_heads = 64;
                num_kv_heads = 8;
                vocab_size = 128256;
                break;
            case LLMModel::Qwen2_7B:
                hidden_dim = 3584;
                ffn_dim = 18944;
                num_layers = 28;
                num_heads = 28;
                num_kv_heads = 4;
                vocab_size = 152064;
                break;
            default:
                hidden_dim = 4096;
                ffn_dim = 11008;
                num_layers = 32;
                num_heads = 32;
                num_kv_heads = 32;
                vocab_size = 32000;
        }

        // Allocate embeddings
        cudaMalloc(&weights_->token_embeddings,
                  vocab_size * hidden_dim * sizeof(float));

        // Allocate transformer layers
        weights_->layers.resize(num_layers);
        for (auto& layer : weights_->layers) {
            AllocateTransformerLayer(layer, hidden_dim, ffn_dim, num_kv_heads);
        }

        // Allocate output weights
        cudaMalloc(&weights_->output_norm, hidden_dim * sizeof(float));
        cudaMalloc(&weights_->lm_head, hidden_dim * vocab_size * sizeof(float));

        // Load weights from file
        if (config_.use_int4_quantization) {
            LoadQuantizedWeights();
        } else {
            LoadFloatWeights();
        }
    }

    void AllocateTransformerLayer(ModelWeights::TransformerLayer& layer,
                                 int hidden_dim, int ffn_dim, int num_kv_heads) {
        int kv_dim = (hidden_dim / 32) * num_kv_heads;  // Grouped query attention

        cudaMalloc(&layer.q_proj, hidden_dim * hidden_dim * sizeof(float));
        cudaMalloc(&layer.k_proj, hidden_dim * kv_dim * sizeof(float));
        cudaMalloc(&layer.v_proj, hidden_dim * kv_dim * sizeof(float));
        cudaMalloc(&layer.o_proj, hidden_dim * hidden_dim * sizeof(float));
        cudaMalloc(&layer.gate_proj, hidden_dim * ffn_dim * sizeof(float));
        cudaMalloc(&layer.up_proj, hidden_dim * ffn_dim * sizeof(float));
        cudaMalloc(&layer.down_proj, ffn_dim * hidden_dim * sizeof(float));
        cudaMalloc(&layer.input_norm, hidden_dim * sizeof(float));
        cudaMalloc(&layer.post_attention_norm, hidden_dim * sizeof(float));
    }

    void InitializeKVCache() {
        if (!config_.use_kv_cache) return;

        kv_cache_ = std::make_unique<KVCache>();
        kv_cache_->max_seq_len = config_.max_context_length;
        kv_cache_->num_layers = weights_->layers.size();
        kv_cache_->batch_size = config_.batch_size;

        // Calculate KV dimension
        int hidden_dim = GetHiddenDim();
        int num_heads = GetNumHeads();
        int num_kv_heads = GetNumKVHeads();
        kv_cache_->kv_dim = (hidden_dim / num_heads) * num_kv_heads;

        // Allocate cache
        size_t cache_size = kv_cache_->num_layers * kv_cache_->batch_size *
                           kv_cache_->max_seq_len * kv_cache_->kv_dim * sizeof(float);
        cudaMalloc(&kv_cache_->k_cache, cache_size);
        cudaMalloc(&kv_cache_->v_cache, cache_size);
        cudaMallocHost(&kv_cache_->seq_lengths, kv_cache_->batch_size * sizeof(int));

        // Initialize
        cudaMemset(kv_cache_->k_cache, 0, cache_size);
        cudaMemset(kv_cache_->v_cache, 0, cache_size);
        memset(kv_cache_->seq_lengths, 0, kv_cache_->batch_size * sizeof(int));
    }

    void ForwardPass(int* tokens, int seq_len, float* logits) {
        int hidden_dim = GetHiddenDim();
        int num_layers = weights_->layers.size();

        // Allocate activations
        float* hidden_states;
        cudaMalloc(&hidden_states, seq_len * hidden_dim * sizeof(float));

        // Token embedding
        EmbedTokens(tokens, seq_len, hidden_states);

        // Transformer layers
        for (int layer_idx = 0; layer_idx < num_layers; layer_idx++) {
            TransformerLayer(hidden_states, seq_len, layer_idx);
        }

        // Output norm
        cuda::llm::rms_norm(hidden_states, hidden_states,
                           weights_->output_norm,
                           1, seq_len, hidden_dim,
                           1e-6f, stream_);

        // LM head (only last token for generation)
        ComputeLogits(hidden_states + (seq_len - 1) * hidden_dim, logits);

        cudaFree(hidden_states);
    }

    void TransformerLayer(float* hidden_states, int seq_len, int layer_idx) {
        int hidden_dim = GetHiddenDim();
        int ffn_dim = GetFFNDim();
        int num_heads = GetNumHeads();
        int num_kv_heads = GetNumKVHeads();
        int head_dim = hidden_dim / num_heads;

        auto& layer = weights_->layers[layer_idx];

        // Allocate temporary buffers
        float *normed, *q, *k, *v, *attn_output;
        cudaMalloc(&normed, seq_len * hidden_dim * sizeof(float));
        cudaMalloc(&q, seq_len * hidden_dim * sizeof(float));
        cudaMalloc(&k, seq_len * (hidden_dim / num_heads * num_kv_heads) * sizeof(float));
        cudaMalloc(&v, seq_len * (hidden_dim / num_heads * num_kv_heads) * sizeof(float));
        cudaMalloc(&attn_output, seq_len * hidden_dim * sizeof(float));

        // Input norm
        cuda::llm::rms_norm(hidden_states, normed, layer.input_norm,
                           1, seq_len, hidden_dim, 1e-6f, stream_);

        // QKV projections
        if (config_.use_int4_quantization && weights_->quantized) {
            // Use INT4 quantized matmul
            cuda::llm::int4_dequantize_matmul(
                weights_->quantized->q_proj_int4,
                weights_->quantized->q_proj_scales,
                weights_->quantized->q_proj_zeros,
                normed, q,
                seq_len, hidden_dim, hidden_dim,
                stream_
            );
        } else {
            // Regular matmul
            MatMul(normed, layer.q_proj, q, seq_len, hidden_dim, hidden_dim);
        }

        MatMul(normed, layer.k_proj, k, seq_len,
               hidden_dim, hidden_dim / num_heads * num_kv_heads);
        MatMul(normed, layer.v_proj, v, seq_len,
               hidden_dim, hidden_dim / num_heads * num_kv_heads);

        // Apply RoPE
        cuda::llm::rotary_position_embedding(q, k, seq_len,
                                            num_heads, head_dim, stream_);

        // Update KV cache
        if (kv_cache_) {
            UpdateKVCache(k, v, layer_idx, seq_len);
        }

        // Grouped query attention
        cuda::llm::grouped_query_attention(
            q, k, v, attn_output,
            1, seq_len, num_heads, num_kv_heads, head_dim,
            stream_
        );

        // Output projection
        MatMul(attn_output, layer.o_proj, attn_output,
               seq_len, hidden_dim, hidden_dim);

        // Residual connection
        AddResidual(hidden_states, attn_output, seq_len * hidden_dim);

        // Post-attention norm
        cuda::llm::rms_norm(hidden_states, normed,
                           layer.post_attention_norm,
                           1, seq_len, hidden_dim, 1e-6f, stream_);

        // FFN with SwiGLU activation
        cuda::llm::swiglu_ffn(normed, attn_output,
                             layer.gate_proj, layer.up_proj, layer.down_proj,
                             1, seq_len, hidden_dim, ffn_dim,
                             stream_);

        // Residual connection
        AddResidual(hidden_states, attn_output, seq_len * hidden_dim);

        // Cleanup
        cudaFree(normed);
        cudaFree(q);
        cudaFree(k);
        cudaFree(v);
        cudaFree(attn_output);
    }

    void UpdateKVCache(float* k, float* v, int layer_idx, int seq_len) {
        int kv_dim = kv_cache_->kv_dim;
        int cache_offset = layer_idx * kv_cache_->batch_size *
                          kv_cache_->max_seq_len * kv_dim;

        // Copy new K,V to cache
        int start_pos = kv_cache_->seq_lengths[0];

        cudaMemcpyAsync(kv_cache_->k_cache + cache_offset + start_pos * kv_dim,
                       k, seq_len * kv_dim * sizeof(float),
                       cudaMemcpyDeviceToDevice, stream_);

        cudaMemcpyAsync(kv_cache_->v_cache + cache_offset + start_pos * kv_dim,
                       v, seq_len * kv_dim * sizeof(float),
                       cudaMemcpyDeviceToDevice, stream_);
    }

    int SampleToken(float* logits, float temperature, float top_p, int top_k) {
        // Apply temperature
        if (temperature != 1.0f) {
            ApplyTemperature(logits, temperature);
        }

        // Apply repetition penalty if needed
        if (config_.repetition_penalty > 1.0f) {
            ApplyRepetitionPenalty(logits);
        }

        // Top-K filtering
        if (top_k > 0) {
            ApplyTopK(logits, top_k);
        }

        // Top-P (nucleus) sampling
        if (top_p < 1.0f) {
            return NucleusSampling(logits, top_p);
        }

        // Greedy sampling
        return GreedySampling(logits);
    }

    std::string BuildSummarizationPrompt(const std::string& text,
                                        const std::string& style) {
        std::string prompt;

        if (style == "concise") {
            prompt = "Summarize the following text in 2-3 sentences, capturing only the most important points:\n\n";
        } else if (style == "detailed") {
            prompt = "Provide a comprehensive summary of the following text, including key points, supporting details, and conclusions:\n\n";
        } else if (style == "bullet") {
            prompt = "Summarize the following text as a bullet-point list of key takeaways:\n\n";
        } else if (style == "technical") {
            prompt = "Provide a technical summary focusing on methodologies, data, and findings:\n\n";
        } else {
            prompt = "Summarize the following text:\n\n";
        }

        prompt += text;
        prompt += "\n\nSummary:";

        return prompt;
    }

    std::string PostProcessSummary(const std::string& summary,
                                  const std::string& style) {
        std::string processed = summary;

        // Remove any instruction leakage
        size_t pos = processed.find("Summarize");
        if (pos != std::string::npos && pos < 50) {
            processed = processed.substr(processed.find('\n') + 1);
        }

        // Format based on style
        if (style == "bullet") {
            // Ensure bullet points
            if (processed[0] != '-' && processed[0] != '•') {
                processed = "• " + processed;
            }
        }

        // Trim whitespace
        processed.erase(0, processed.find_first_not_of(" \t\n"));
        processed.erase(processed.find_last_not_of(" \t\n") + 1);

        return processed;
    }

    // Helper functions
    void MatMul(const float* A, const float* B, float* C,
               int M, int N, int K) {
        float alpha = 1.0f, beta = 0.0f;

        // Use cuBLASLt for optimized GEMM
        cublasLtMatmulDesc_t matmul_desc;
        cublasLtMatmulDescCreate(&matmul_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);

        cublasLtMatrixLayout_t a_desc, b_desc, c_desc;
        cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_32F, M, K, M);
        cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_32F, K, N, K);
        cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_32F, M, N, M);

        cublasLtMatmul(cublaslt_handle_, matmul_desc,
                      &alpha, A, a_desc, B, b_desc,
                      &beta, C, c_desc, C, c_desc,
                      nullptr, workspace_, workspace_size_, stream_);

        cublasLtMatrixLayoutDestroy(a_desc);
        cublasLtMatrixLayoutDestroy(b_desc);
        cublasLtMatrixLayoutDestroy(c_desc);
        cublasLtMatmulDescDestroy(matmul_desc);
    }

    void AddResidual(float* hidden_states, const float* residual, int size) {
        // hidden_states += residual
        float alpha = 1.0f;
        cublasSaxpy(cublaslt_handle_, size, &alpha,
                   residual, 1, hidden_states, 1);
    }

    void EmbedTokens(int* tokens, int seq_len, float* embeddings) {
        int hidden_dim = GetHiddenDim();
        int vocab_size = GetVocabSize();

        // Lookup embeddings for each token
        for (int i = 0; i < seq_len; i++) {
            int token = tokens[i];
            cudaMemcpyAsync(embeddings + i * hidden_dim,
                          weights_->token_embeddings + token * hidden_dim,
                          hidden_dim * sizeof(float),
                          cudaMemcpyDeviceToDevice, stream_);
        }
    }

    void ComputeLogits(float* hidden_state, float* logits) {
        int hidden_dim = GetHiddenDim();
        int vocab_size = GetVocabSize();

        MatMul(hidden_state, weights_->lm_head, logits,
               1, vocab_size, hidden_dim);
    }

    void ApplyTemperature(float* logits, float temperature) {
        int vocab_size = GetVocabSize();
        float inv_temp = 1.0f / temperature;
        cublasSscal(cublaslt_handle_, vocab_size, &inv_temp, logits, 1);
    }

    void ApplyRepetitionPenalty(float* logits) {
        // Would track previously generated tokens and apply penalty
    }

    void ApplyTopK(float* logits, int top_k) {
        // Would sort and zero out all but top-k logits
    }

    int NucleusSampling(float* logits, float top_p) {
        // Would implement nucleus sampling
        return 0;
    }

    int GreedySampling(float* logits) {
        int vocab_size = GetVocabSize();

        // Find argmax
        int max_idx = 0;
        float max_val = -INFINITY;

        std::vector<float> h_logits(vocab_size);
        cudaMemcpy(h_logits.data(), logits, vocab_size * sizeof(float),
                  cudaMemcpyDeviceToHost);

        for (int i = 0; i < vocab_size; i++) {
            if (h_logits[i] > max_val) {
                max_val = h_logits[i];
                max_idx = i;
            }
        }

        return max_idx;
    }

    // Model dimension getters
    int GetHiddenDim() const {
        switch (config_.model) {
            case LLMModel::Llama3_8B: return 4096;
            case LLMModel::Llama3_70B: return 8192;
            case LLMModel::Qwen2_7B: return 3584;
            default: return 4096;
        }
    }

    int GetFFNDim() const {
        switch (config_.model) {
            case LLMModel::Llama3_8B: return 14336;
            case LLMModel::Llama3_70B: return 28672;
            case LLMModel::Qwen2_7B: return 18944;
            default: return 11008;
        }
    }

    int GetNumHeads() const {
        switch (config_.model) {
            case LLMModel::Llama3_8B: return 32;
            case LLMModel::Llama3_70B: return 64;
            case LLMModel::Qwen2_7B: return 28;
            default: return 32;
        }
    }

    int GetNumKVHeads() const {
        switch (config_.model) {
            case LLMModel::Llama3_8B: return 8;
            case LLMModel::Llama3_70B: return 8;
            case LLMModel::Qwen2_7B: return 4;
            default: return 32;
        }
    }

    int GetVocabSize() const {
        switch (config_.model) {
            case LLMModel::Llama3_8B:
            case LLMModel::Llama3_70B:
                return 128256;
            case LLMModel::Qwen2_7B:
                return 152064;
            default:
                return 32000;
        }
    }

    void LoadFloatWeights() {
        // Load weights from file (simplified)
        FILE* f = fopen(config_.model_path.c_str(), "rb");
        if (!f) {
            throw std::runtime_error("Failed to open model file");
        }

        // Read weights into GPU memory
        // (Implementation would read actual weight format)

        fclose(f);
    }

    void LoadQuantizedWeights() {
        weights_->quantized = std::make_unique<ModelWeights::QuantizedWeights>();
        // Load INT4 quantized weights
    }

    void LoadTokenizer() {
        // Load tokenizer (would use sentencepiece or tiktoken)
    }

    void FreeModelWeights() {
        if (!weights_) return;

        cudaFree(weights_->token_embeddings);
        for (auto& layer : weights_->layers) {
            cudaFree(layer.q_proj);
            cudaFree(layer.k_proj);
            cudaFree(layer.v_proj);
            cudaFree(layer.o_proj);
            cudaFree(layer.gate_proj);
            cudaFree(layer.up_proj);
            cudaFree(layer.down_proj);
            cudaFree(layer.input_norm);
            cudaFree(layer.post_attention_norm);
        }
        cudaFree(weights_->output_norm);
        cudaFree(weights_->lm_head);
    }

    void FreeKVCache() {
        if (!kv_cache_) return;

        cudaFree(kv_cache_->k_cache);
        cudaFree(kv_cache_->v_cache);
        cudaFreeHost(kv_cache_->seq_lengths);
    }
};

// Simple tokenizer interface
class Tokenizer {
public:
    virtual std::vector<int> Encode(const std::string& text) = 0;
    virtual std::string Decode(const std::vector<int>& tokens) = 0;
    virtual int GetEOSToken() = 0;
    virtual ~Tokenizer() = default;
};

// Global LLM engine instance
static std::unique_ptr<LLMEngine> g_llm_engine;

void InitializeLLM(const LLMConfig& config) {
    g_llm_engine = std::make_unique<LLMEngine>(config);
}

std::string GenerateSummary(const std::string& text, const std::string& style) {
    if (!g_llm_engine) {
        throw std::runtime_error("LLM engine not initialized");
    }
    return g_llm_engine->Summarize(text, style);
}

} // namespace summarization