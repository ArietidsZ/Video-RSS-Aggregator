#include "whisper_turbo.h"
#include <cuda_runtime.h>
#include <thread>
#include <atomic>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <memory>
#include <cstring>
#include <algorithm>

namespace whisper_turbo {

// Forward declarations for CUDA streaming kernels
namespace cuda {
namespace streaming {
    void process_audio_chunk(const float* input, float* features,
                            int chunk_size, int overlap_size,
                            cudaStream_t stream);

    void streaming_attention(const float* features, float* context,
                            const float* cache_k, const float* cache_v,
                            int seq_len, int cache_len,
                            cudaStream_t stream);

    void update_kv_cache(float* cache_k, float* cache_v,
                        const float* new_k, const float* new_v,
                        int cache_size, int update_size,
                        cudaStream_t stream);
}
}

class StreamingTranscriber {
public:
    struct StreamingConfig {
        int chunk_length_ms = 2000;        // 2 second chunks
        int overlap_ms = 500;              // 500ms overlap between chunks
        int lookahead_chunks = 2;          // Look ahead for context
        float silence_threshold = 0.01f;   // VAD threshold
        int min_silence_ms = 300;          // Minimum silence for segmentation
        bool enable_punctuation = true;    // Real-time punctuation
        bool enable_speaker_change = false; // Speaker change detection
        int max_context_length = 4096;     // Maximum context tokens
        float latency_target_ms = 100.0f;  // Target latency
    };

    struct StreamingState {
        std::vector<float> audio_buffer;
        std::vector<float> overlap_buffer;
        std::string partial_transcript;
        std::string confirmed_transcript;
        std::vector<int> context_tokens;
        float* kv_cache_k;
        float* kv_cache_v;
        int cache_position;
        std::chrono::steady_clock::time_point last_speech_time;
        bool is_speaking;
        int speaker_id;
    };

private:
    StreamingConfig config_;
    std::unique_ptr<StreamingState> state_;
    WhisperTurbo* model_;
    cudaStream_t stream_;

    // Ring buffer for audio chunks
    struct AudioChunk {
        std::vector<float> samples;
        std::chrono::steady_clock::time_point timestamp;
        bool contains_speech;
        float energy;
    };

    std::queue<AudioChunk> chunk_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;

    // Processing thread
    std::thread processing_thread_;
    std::atomic<bool> is_running_{true};

    // Callback for partial results
    using StreamingCallback = std::function<void(const std::string& partial,
                                                 const std::string& confirmed,
                                                 bool is_final)>;
    StreamingCallback callback_;

    // Performance monitoring
    struct StreamingMetrics {
        std::atomic<double> avg_chunk_latency_ms{0};
        std::atomic<double> avg_rtf{0};  // Real-time factor
        std::atomic<size_t> chunks_processed{0};
        std::atomic<size_t> words_transcribed{0};
    };
    StreamingMetrics metrics_;

public:
    StreamingTranscriber(WhisperTurbo* model, const StreamingConfig& config = {})
        : model_(model), config_(config) {

        InitializeState();
        cudaStreamCreate(&stream_);

        // Start processing thread
        processing_thread_ = std::thread(&StreamingTranscriber::ProcessingLoop, this);
    }

    ~StreamingTranscriber() {
        Stop();
        if (processing_thread_.joinable()) {
            processing_thread_.join();
        }
        CleanupState();
        cudaStreamDestroy(stream_);
    }

    // Feed audio samples for streaming transcription
    void FeedAudio(const float* samples, size_t num_samples, uint32_t sample_rate = 16000) {
        // Resample if necessary
        std::vector<float> resampled;
        if (sample_rate != 16000) {
            resampled = Resample(samples, num_samples, sample_rate, 16000);
            samples = resampled.data();
            num_samples = resampled.size();
        }

        // Buffer incoming audio
        state_->audio_buffer.insert(state_->audio_buffer.end(),
                                   samples, samples + num_samples);

        // Process chunks when we have enough data
        int chunk_samples = (config_.chunk_length_ms * 16000) / 1000;
        int overlap_samples = (config_.overlap_ms * 16000) / 1000;

        while (state_->audio_buffer.size() >= chunk_samples) {
            AudioChunk chunk;
            chunk.samples.assign(state_->audio_buffer.begin(),
                               state_->audio_buffer.begin() + chunk_samples);
            chunk.timestamp = std::chrono::steady_clock::now();

            // Compute energy for VAD
            chunk.energy = ComputeEnergy(chunk.samples);
            chunk.contains_speech = chunk.energy > config_.silence_threshold;

            // Add to processing queue
            {
                std::lock_guard<std::mutex> lock(queue_mutex_);
                chunk_queue_.push(std::move(chunk));
            }
            queue_cv_.notify_one();

            // Keep overlap for next chunk
            state_->audio_buffer.erase(state_->audio_buffer.begin(),
                                      state_->audio_buffer.begin() + chunk_samples - overlap_samples);
        }
    }

    // Flush any remaining audio and get final transcript
    std::string Flush() {
        // Process remaining audio in buffer
        if (!state_->audio_buffer.empty()) {
            AudioChunk chunk;
            chunk.samples = state_->audio_buffer;
            chunk.timestamp = std::chrono::steady_clock::now();
            chunk.energy = ComputeEnergy(chunk.samples);
            chunk.contains_speech = chunk.energy > config_.silence_threshold;

            {
                std::lock_guard<std::mutex> lock(queue_mutex_);
                chunk_queue_.push(std::move(chunk));
            }
            queue_cv_.notify_one();

            state_->audio_buffer.clear();
        }

        // Wait for processing to complete
        while (!chunk_queue_.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        // Finalize transcript
        FinalizeTranscript();

        return state_->confirmed_transcript;
    }

    // Set callback for streaming results
    void SetCallback(StreamingCallback callback) {
        callback_ = callback;
    }

    // Reset streaming state
    void Reset() {
        state_->audio_buffer.clear();
        state_->overlap_buffer.clear();
        state_->partial_transcript.clear();
        state_->confirmed_transcript.clear();
        state_->context_tokens.clear();
        state_->cache_position = 0;
        state_->is_speaking = false;

        // Clear cache
        cudaMemsetAsync(state_->kv_cache_k, 0,
                       config_.max_context_length * 1024 * sizeof(float), stream_);
        cudaMemsetAsync(state_->kv_cache_v, 0,
                       config_.max_context_length * 1024 * sizeof(float), stream_);
    }

    // Get current metrics
    StreamingMetrics GetMetrics() const {
        return metrics_;
    }

private:
    void InitializeState() {
        state_ = std::make_unique<StreamingState>();

        // Allocate KV cache for streaming attention
        size_t cache_size = config_.max_context_length * 1024 * sizeof(float);
        cudaMalloc(&state_->kv_cache_k, cache_size);
        cudaMalloc(&state_->kv_cache_v, cache_size);

        cudaMemset(state_->kv_cache_k, 0, cache_size);
        cudaMemset(state_->kv_cache_v, 0, cache_size);

        state_->cache_position = 0;
        state_->is_speaking = false;
        state_->speaker_id = 0;
    }

    void CleanupState() {
        if (state_) {
            cudaFree(state_->kv_cache_k);
            cudaFree(state_->kv_cache_v);
        }
    }

    void ProcessingLoop() {
        while (is_running_) {
            AudioChunk chunk;
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                queue_cv_.wait_for(lock, std::chrono::milliseconds(100),
                                  [this] { return !chunk_queue_.empty() || !is_running_; });

                if (!is_running_) break;
                if (chunk_queue_.empty()) continue;

                chunk = std::move(chunk_queue_.front());
                chunk_queue_.pop();
            }

            ProcessChunk(chunk);
        }
    }

    void ProcessChunk(const AudioChunk& chunk) {
        auto start_time = std::chrono::high_resolution_clock::now();

        // Skip silent chunks unless we were speaking
        if (!chunk.contains_speech && !state_->is_speaking) {
            return;
        }

        // Update speaking state
        if (chunk.contains_speech) {
            state_->is_speaking = true;
            state_->last_speech_time = chunk.timestamp;
        } else {
            auto silence_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                chunk.timestamp - state_->last_speech_time).count();

            if (silence_duration > config_.min_silence_ms) {
                state_->is_speaking = false;
                FinalizeSegment();
            }
        }

        // Process audio through streaming pipeline
        StreamingResult result = ProcessAudioStreaming(chunk.samples);

        // Update transcripts
        UpdateTranscripts(result);

        // Invoke callback if set
        if (callback_) {
            bool is_final = !state_->is_speaking;
            callback_(state_->partial_transcript,
                     state_->confirmed_transcript,
                     is_final);
        }

        // Update metrics
        auto end_time = std::chrono::high_resolution_clock::now();
        double latency_ms = std::chrono::duration<double, std::milli>(
            end_time - start_time).count();

        metrics_.avg_chunk_latency_ms = (metrics_.avg_chunk_latency_ms * 0.9) +
                                        (latency_ms * 0.1);
        metrics_.chunks_processed++;

        // Calculate real-time factor
        double audio_duration_ms = (chunk.samples.size() / 16.0);  // 16 samples per ms at 16kHz
        double rtf = latency_ms / audio_duration_ms;
        metrics_.avg_rtf = (metrics_.avg_rtf * 0.9) + (rtf * 0.1);
    }

    struct StreamingResult {
        std::vector<int> tokens;
        std::vector<float> probabilities;
        std::string text;
        bool is_partial;
        int start_time_ms;
        int end_time_ms;
    };

    StreamingResult ProcessAudioStreaming(const std::vector<float>& audio_chunk) {
        StreamingResult result;

        // Allocate device memory
        float* d_audio;
        float* d_features;
        float* d_context;
        int* d_tokens;

        size_t audio_size = audio_chunk.size() * sizeof(float);
        size_t feature_size = 80 * 50 * sizeof(float);  // 80 mel bins, ~50 frames
        size_t context_size = 1024 * sizeof(float);

        cudaMalloc(&d_audio, audio_size);
        cudaMalloc(&d_features, feature_size);
        cudaMalloc(&d_context, context_size);
        cudaMalloc(&d_tokens, 448 * sizeof(int));

        // Copy audio to device
        cudaMemcpyAsync(d_audio, audio_chunk.data(), audio_size,
                       cudaMemcpyHostToDevice, stream_);

        // Extract features with overlap handling
        int overlap_samples = (config_.overlap_ms * 16000) / 1000;
        cuda::streaming::process_audio_chunk(
            d_audio, d_features,
            audio_chunk.size(), overlap_samples,
            stream_
        );

        // Streaming attention with KV cache
        cuda::streaming::streaming_attention(
            d_features, d_context,
            state_->kv_cache_k, state_->kv_cache_v,
            50, state_->cache_position,
            stream_
        );

        // Decode tokens
        DecodeStreamingTokens(d_context, d_tokens, 448);

        // Copy results back
        result.tokens.resize(448);
        cudaMemcpyAsync(result.tokens.data(), d_tokens,
                       448 * sizeof(int),
                       cudaMemcpyDeviceToHost, stream_);

        cudaStreamSynchronize(stream_);

        // Convert tokens to text
        result.text = model_->DecodeTokens(result.tokens.data(), 448);
        result.is_partial = state_->is_speaking;

        // Update KV cache
        UpdateKVCache(d_features);

        // Cleanup
        cudaFree(d_audio);
        cudaFree(d_features);
        cudaFree(d_context);
        cudaFree(d_tokens);

        return result;
    }

    void DecodeStreamingTokens(float* context, int* tokens, int max_tokens) {
        // Simplified greedy decoding for streaming
        // In production, would use beam search with constraints

        for (int i = 0; i < max_tokens; i++) {
            // Get next token probabilities
            float* logits = context + i * 51865;  // Vocab size

            // Find max probability token
            int best_token = 0;
            float best_score = -INFINITY;

            for (int t = 0; t < 51865; t++) {
                if (logits[t] > best_score) {
                    best_score = logits[t];
                    best_token = t;
                }
            }

            tokens[i] = best_token;

            // Stop at end token
            if (best_token == 50258) {  // <|endoftranscript|>
                break;
            }
        }
    }

    void UpdateKVCache(float* new_features) {
        // Shift cache if needed
        if (state_->cache_position >= config_.max_context_length - 50) {
            // Shift cache by half to make room
            int shift_amount = config_.max_context_length / 2;

            cuda::streaming::update_kv_cache(
                state_->kv_cache_k, state_->kv_cache_v,
                state_->kv_cache_k + shift_amount * 1024,
                state_->kv_cache_v + shift_amount * 1024,
                config_.max_context_length - shift_amount,
                shift_amount,
                stream_
            );

            state_->cache_position -= shift_amount;
        }

        // Add new features to cache
        // (Simplified - would compute K,V from features in production)
        cudaMemcpyAsync(state_->kv_cache_k + state_->cache_position * 1024,
                       new_features, 50 * 1024 * sizeof(float),
                       cudaMemcpyDeviceToDevice, stream_);
        cudaMemcpyAsync(state_->kv_cache_v + state_->cache_position * 1024,
                       new_features, 50 * 1024 * sizeof(float),
                       cudaMemcpyDeviceToDevice, stream_);

        state_->cache_position += 50;
    }

    void UpdateTranscripts(const StreamingResult& result) {
        if (result.text.empty()) return;

        // Update partial transcript
        state_->partial_transcript = result.text;

        // Confirm stable portions
        if (!result.is_partial) {
            // Move partial to confirmed
            if (!state_->confirmed_transcript.empty()) {
                state_->confirmed_transcript += " ";
            }
            state_->confirmed_transcript += state_->partial_transcript;
            state_->partial_transcript.clear();

            // Count words
            metrics_.words_transcribed += CountWords(result.text);
        }

        // Apply punctuation if enabled
        if (config_.enable_punctuation) {
            ApplyPunctuation(state_->partial_transcript);
            ApplyPunctuation(state_->confirmed_transcript);
        }
    }

    void FinalizeSegment() {
        // Move any remaining partial transcript to confirmed
        if (!state_->partial_transcript.empty()) {
            if (!state_->confirmed_transcript.empty()) {
                state_->confirmed_transcript += " ";
            }
            state_->confirmed_transcript += state_->partial_transcript;
            state_->partial_transcript.clear();
        }
    }

    void FinalizeTranscript() {
        FinalizeSegment();

        // Apply final formatting
        if (config_.enable_punctuation) {
            ApplyPunctuation(state_->confirmed_transcript);
        }
    }

    void ApplyPunctuation(std::string& text) {
        // Simple rule-based punctuation
        // In production, would use a neural punctuation model

        // Capitalize first letter
        if (!text.empty()) {
            text[0] = std::toupper(text[0]);
        }

        // Add period at end if missing
        if (!text.empty() && text.back() != '.' &&
            text.back() != '!' && text.back() != '?') {
            text += '.';
        }

        // Capitalize after sentence endings
        for (size_t i = 2; i < text.length(); i++) {
            if ((text[i-2] == '.' || text[i-2] == '!' || text[i-2] == '?') &&
                text[i-1] == ' ' && std::islower(text[i])) {
                text[i] = std::toupper(text[i]);
            }
        }
    }

    float ComputeEnergy(const std::vector<float>& samples) {
        float energy = 0.0f;
        for (float sample : samples) {
            energy += sample * sample;
        }
        return std::sqrt(energy / samples.size());
    }

    std::vector<float> Resample(const float* input, size_t input_len,
                                uint32_t input_rate, uint32_t output_rate) {
        // Simple linear resampling
        // In production, would use a proper resampling library

        float ratio = static_cast<float>(input_rate) / output_rate;
        size_t output_len = input_len / ratio;
        std::vector<float> output(output_len);

        for (size_t i = 0; i < output_len; i++) {
            float src_idx = i * ratio;
            size_t idx0 = static_cast<size_t>(src_idx);
            size_t idx1 = std::min(idx0 + 1, input_len - 1);
            float frac = src_idx - idx0;

            output[i] = input[idx0] * (1.0f - frac) + input[idx1] * frac;
        }

        return output;
    }

    size_t CountWords(const std::string& text) {
        std::istringstream stream(text);
        std::string word;
        size_t count = 0;
        while (stream >> word) {
            count++;
        }
        return count;
    }

    void Stop() {
        is_running_ = false;
        queue_cv_.notify_all();
    }
};

// WebSocket streaming server for real-time transcription
class StreamingServer {
private:
    std::unique_ptr<StreamingTranscriber> transcriber_;
    std::thread server_thread_;
    std::atomic<bool> is_running_{true};
    int port_;

public:
    StreamingServer(WhisperTurbo* model, int port = 8765)
        : port_(port) {

        StreamingTranscriber::StreamingConfig config;
        config.chunk_length_ms = 1000;  // 1 second chunks for low latency
        config.overlap_ms = 200;
        config.latency_target_ms = 50.0f;

        transcriber_ = std::make_unique<StreamingTranscriber>(model, config);

        // Set callback for WebSocket broadcast
        transcriber_->SetCallback([this](const std::string& partial,
                                        const std::string& confirmed,
                                        bool is_final) {
            BroadcastTranscript(partial, confirmed, is_final);
        });

        // Start server thread
        server_thread_ = std::thread(&StreamingServer::RunServer, this);
    }

    ~StreamingServer() {
        Stop();
        if (server_thread_.joinable()) {
            server_thread_.join();
        }
    }

    void FeedAudioData(const uint8_t* pcm_data, size_t num_bytes) {
        // Convert PCM bytes to float samples
        std::vector<float> samples(num_bytes / 2);  // 16-bit PCM
        const int16_t* pcm16 = reinterpret_cast<const int16_t*>(pcm_data);

        for (size_t i = 0; i < samples.size(); i++) {
            samples[i] = pcm16[i] / 32768.0f;
        }

        transcriber_->FeedAudio(samples.data(), samples.size(), 16000);
    }

private:
    void RunServer() {
        // WebSocket server implementation would go here
        // Using a library like libwebsockets or Boost.Beast

        while (is_running_) {
            // Accept connections and handle audio streaming
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    void BroadcastTranscript(const std::string& partial,
                            const std::string& confirmed,
                            bool is_final) {
        // Send transcript update to all connected WebSocket clients
        // JSON format: {"partial": "...", "confirmed": "...", "is_final": true/false}
    }

    void Stop() {
        is_running_ = false;
    }
};

// Global streaming instance
static std::unique_ptr<StreamingTranscriber> g_streaming_transcriber;

void InitializeStreaming(WhisperTurbo* model,
                        const StreamingTranscriber::StreamingConfig& config) {
    g_streaming_transcriber = std::make_unique<StreamingTranscriber>(model, config);
}

void FeedStreamingAudio(const float* samples, size_t num_samples, uint32_t sample_rate) {
    if (g_streaming_transcriber) {
        g_streaming_transcriber->FeedAudio(samples, num_samples, sample_rate);
    }
}

std::string FlushStreamingTranscript() {
    if (g_streaming_transcriber) {
        return g_streaming_transcriber->Flush();
    }
    return "";
}

void ResetStreamingState() {
    if (g_streaming_transcriber) {
        g_streaming_transcriber->Reset();
    }
}

} // namespace whisper_turbo