#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <functional>
#include <optional>
#include <chrono>

namespace whisper_turbo {

// Quantization types for model optimization
enum class QuantizationType {
    FP32,     // Full precision
    FP16,     // Half precision
    BF16,     // BFloat16
    INT8,     // 8-bit integer quantization
    FP8,      // 8-bit floating point (H100 GPUs)
    INT4,     // 4-bit integer quantization
    DYNAMIC   // Dynamic quantization
};

// Model variants optimized for different hardware
enum class ModelVariant {
    WHISPER_TURBO,      // Fastest, optimized for latency
    FASTER_WHISPER,     // CTranslate2 based
    WHISPER_STANDARD,   // Original OpenAI Whisper
    PARAFORMER,         // Alibaba's Paraformer for Chinese
    FUNASR,            // FunASR for Chinese
    QWEN3_ASR          // Qwen3-ASR for Chinese
};

// Model sizes
enum class ModelSize {
    TINY,      // 39M parameters
    BASE,      // 74M parameters
    SMALL,     // 244M parameters
    MEDIUM,    // 769M parameters
    LARGE,     // 1550M parameters
    TURBO,     // Optimized large variant
    ULTRA      // Custom ultra-large model
};

// Language optimization settings
enum class LanguageOptimization {
    AUTO_DETECT,
    ENGLISH,
    CHINESE,
    MULTILINGUAL
};

// Acceleration features
struct AccelerationConfig {
    bool enable_cuda_graphs = true;      // CUDA Graph optimization
    bool enable_kernel_fusion = true;    // Fused CUDA kernels
    bool enable_flash_attention = true;  // Flash Attention v2
    bool enable_tensor_cores = true;     // Tensor Core acceleration
    bool enable_fp8 = false;             // FP8 on H100/H200
    bool enable_int4 = false;            // INT4 quantization
    bool enable_kv_cache = true;         // Key-Value cache
    bool enable_beam_pruning = true;     // Dynamic beam pruning
    bool enable_batching = true;         // Dynamic batching
    bool enable_streaming = false;       // Real-time streaming mode
};

// Transcription options
struct TranscriptionOptions {
    ModelVariant model_variant = ModelVariant::WHISPER_TURBO;
    ModelSize model_size = ModelSize::LARGE;
    QuantizationType quantization = QuantizationType::INT8;
    LanguageOptimization language = LanguageOptimization::AUTO_DETECT;

    uint32_t batch_size = 1;              // Batch size for processing
    uint32_t beam_size = 5;               // Beam search width
    float temperature = 0.0f;             // Sampling temperature
    float vad_threshold = 0.5f;           // Voice activity detection threshold
    uint32_t chunk_length_ms = 5000;      // Audio chunk length in milliseconds
    uint32_t max_context_tokens = 448;    // Maximum context tokens

    bool enable_timestamps = true;        // Generate word-level timestamps
    bool enable_diarization = false;      // Speaker diarization
    bool suppress_blanks = true;          // Suppress blank outputs
    bool condition_on_previous = true;    // Use previous text as context

    AccelerationConfig acceleration;      // Hardware acceleration settings

    std::string model_path;              // Path to model files
    std::string vocab_path;              // Path to vocabulary
    std::vector<std::string> hotwords;   // Domain-specific hotwords for biasing
};

// Transcription segment
struct TranscriptionSegment {
    std::string text;                    // Transcribed text
    float start_time;                    // Start time in seconds
    float end_time;                      // End time in seconds
    float confidence;                    // Confidence score [0, 1]
    std::string language;                // Detected language
    std::optional<int> speaker_id;       // Speaker ID if diarization enabled
    std::vector<std::pair<std::string, float>> word_timestamps; // Word-level timing
};

// Transcription result
struct TranscriptionResult {
    std::vector<TranscriptionSegment> segments;
    std::string full_text;               // Complete transcription
    std::string detected_language;       // Primary language detected
    float average_confidence;            // Average confidence score
    std::chrono::milliseconds processing_time; // Total processing time
    float rtf;                           // Real-time factor (audio_duration / processing_time)
    size_t tokens_generated;             // Number of tokens generated
    size_t audio_samples_processed;      // Number of audio samples processed
};

// Performance metrics
struct PerformanceMetrics {
    float tokens_per_second;            // Inference speed
    float audio_processing_speed;       // xRT (times real-time)
    size_t peak_memory_mb;              // Peak memory usage
    size_t current_memory_mb;           // Current memory usage
    float gpu_utilization;              // GPU utilization percentage
    float cpu_utilization;              // CPU utilization percentage
    std::chrono::milliseconds total_time;
    std::chrono::milliseconds model_load_time;
    std::chrono::milliseconds preprocessing_time;
    std::chrono::milliseconds inference_time;
    std::chrono::milliseconds postprocessing_time;
};

// Progress callback for long-running transcriptions
using ProgressCallback = std::function<void(float progress, const std::string& status)>;

// Main Whisper Turbo engine class
class WhisperTurbo {
public:
    WhisperTurbo();
    ~WhisperTurbo();

    // Initialize the model with given options
    bool Initialize(const TranscriptionOptions& options);

    // Load model from file
    bool LoadModel(const std::string& model_path,
                   ModelSize size = ModelSize::LARGE,
                   QuantizationType quantization = QuantizationType::INT8);

    // Transcribe audio data
    TranscriptionResult Transcribe(const float* audio_data,
                                  size_t num_samples,
                                  uint32_t sample_rate = 16000,
                                  ProgressCallback callback = nullptr);

    // Transcribe audio file
    TranscriptionResult TranscribeFile(const std::string& audio_path,
                                       ProgressCallback callback = nullptr);

    // Stream transcription (real-time)
    void StartStreaming();
    void ProcessStreamChunk(const float* audio_data, size_t num_samples);
    TranscriptionSegment GetStreamResult();
    void StopStreaming();

    // Batch transcription for multiple audio files
    std::vector<TranscriptionResult> TranscribeBatch(
        const std::vector<std::string>& audio_paths,
        ProgressCallback callback = nullptr);

    // Dynamic configuration adjustment
    void AdjustForThermalThrottling(float throttle_factor);
    void SetBatchSize(uint32_t batch_size);
    void SetBeamSize(uint32_t beam_size);
    void EnableAcceleration(const AccelerationConfig& config);

    // Performance monitoring
    PerformanceMetrics GetPerformanceMetrics() const;
    void ResetMetrics();

    // Model management
    void WarmupModel(size_t num_iterations = 10);
    void OptimizeForLatency();
    void OptimizeForThroughput();
    size_t GetModelMemoryUsage() const;

    // Language and customization
    void SetLanguage(const std::string& language_code);
    void AddHotwords(const std::vector<std::string>& hotwords, float boost_weight = 10.0f);
    void SetContextBias(const std::string& context);

    // Utility functions
    static std::vector<std::string> GetAvailableModels();
    static bool IsGPUAvailable();
    static std::string GetVersion();

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

// C API for FFI (Foreign Function Interface)
extern "C" {
    // Opaque handle for the transcriber
    typedef void* whisper_turbo_handle;

    // Create and destroy
    whisper_turbo_handle whisper_turbo_create();
    void whisper_turbo_destroy(whisper_turbo_handle handle);

    // Initialize with model
    int whisper_turbo_init(whisper_turbo_handle handle,
                          const char* model_path,
                          int model_size,
                          int quantization);

    // Transcribe audio
    const char* whisper_turbo_transcribe(whisper_turbo_handle handle,
                                        const float* audio_data,
                                        size_t num_samples,
                                        int sample_rate);

    // Get last error
    const char* whisper_turbo_get_error(whisper_turbo_handle handle);

    // Performance metrics
    float whisper_turbo_get_rtf(whisper_turbo_handle handle);
    size_t whisper_turbo_get_memory_usage(whisper_turbo_handle handle);
}

} // namespace whisper_turbo