#pragma once

#include <string>
#include <memory>
#include <functional>
#include <vector>
#include <atomic>
#include <chrono>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswresample/swresample.h>
#include <libavutil/opt.h>
}

namespace audio_processor {

/**
 * High-performance audio extraction from video streams
 * Target: Real-time streaming with <500ms latency
 */
class AudioExtractor {
public:
    struct Config {
        // Output format
        int sample_rate = 16000;       // Target sample rate (Hz)
        int channels = 1;               // Mono for speech processing
        AVSampleFormat format = AV_SAMPLE_FMT_FLT;  // Float32

        // Performance
        bool use_gpu = true;            // Enable GPU acceleration
        int buffer_size = 4096;         // Audio buffer size
        int thread_count = 4;           // Decoder threads

        // Streaming
        bool enable_streaming = true;   // Enable real-time streaming
        int chunk_duration_ms = 5000;  // Chunk duration in milliseconds
        int overlap_ms = 500;           // Overlap between chunks

        // Network
        std::string input_protocol = "auto";  // auto, http, rtmp, file
        int network_timeout_ms = 10000;       // Network timeout
        int reconnect_attempts = 3;           // Reconnection attempts
    };

    struct AudioFrame {
        std::vector<float> data;       // Audio samples
        int64_t timestamp_ms;          // Timestamp in milliseconds
        int sample_rate;               // Sample rate
        int channels;                  // Number of channels
        size_t sample_count;           // Number of samples

        // Metadata
        float duration_ms;             // Duration in milliseconds
        float energy;                  // Audio energy level
        bool is_speech;                // VAD result
    };

    using FrameCallback = std::function<void(const AudioFrame&)>;
    using ErrorCallback = std::function<void(const std::string&)>;

    explicit AudioExtractor(const Config& config = Config());
    ~AudioExtractor();

    // Non-copyable
    AudioExtractor(const AudioExtractor&) = delete;
    AudioExtractor& operator=(const AudioExtractor&) = delete;

    /**
     * Extract audio from URL without downloading
     * @param url Video/audio stream URL
     * @param callback Frame callback for streaming
     * @return Success status
     */
    bool extract(const std::string& url, FrameCallback callback);

    /**
     * Extract audio asynchronously
     */
    void extract_async(const std::string& url, FrameCallback callback, ErrorCallback error_cb);

    /**
     * Stop extraction
     */
    void stop();

    /**
     * Check if extraction is running
     */
    bool is_running() const { return running_.load(); }

    /**
     * Get extraction statistics
     */
    struct Stats {
        size_t frames_processed = 0;
        size_t bytes_processed = 0;
        float extraction_fps = 0.0f;
        float network_bandwidth_mbps = 0.0f;
        float gpu_utilization = 0.0f;
        std::chrono::milliseconds total_time;
    };

    Stats get_stats() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
    std::atomic<bool> running_{false};
    Config config_;

    // Internal methods
    bool initialize_decoder(const std::string& url);
    bool decode_audio_stream();
    void cleanup();
};

/**
 * Batch audio extractor for multiple URLs
 */
class BatchAudioExtractor {
public:
    struct BatchConfig {
        int max_parallel = 4;          // Maximum parallel extractions
        bool use_gpu_pool = true;      // Use GPU resource pool
        int gpu_count = 1;              // Number of GPUs to use
    };

    explicit BatchAudioExtractor(const BatchConfig& config = BatchConfig());
    ~BatchAudioExtractor();

    /**
     * Extract audio from multiple URLs
     */
    void extract_batch(
        const std::vector<std::string>& urls,
        AudioExtractor::FrameCallback callback
    );

    /**
     * Get batch processing statistics
     */
    struct BatchStats {
        size_t total_urls = 0;
        size_t completed = 0;
        size_t failed = 0;
        float avg_extraction_time_ms = 0.0f;
        float total_throughput_mbps = 0.0f;
    };

    BatchStats get_batch_stats() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
    BatchConfig config_;
};

} // namespace audio_processor