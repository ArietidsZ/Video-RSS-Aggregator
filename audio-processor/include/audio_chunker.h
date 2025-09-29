#pragma once

#include <vector>
#include <memory>
#include <functional>
#include <chrono>
#include <queue>
#include <mutex>
#include <atomic>

namespace audio_processor {

/**
 * Audio chunking with overlap for optimal processing
 * Target: 5-second chunks with 0.5-second overlap
 */
class AudioChunker {
public:
    struct ChunkConfig {
        int chunk_duration_ms = 5000;     // 5 seconds
        int overlap_ms = 500;              // 0.5 seconds overlap
        int sample_rate = 16000;           // 16kHz
        int channels = 1;                  // Mono

        // Advanced settings
        bool apply_window = true;          // Apply windowing function
        enum WindowType {
            RECTANGULAR,
            HAMMING,
            HANNING,
            BLACKMAN
        } window_type = HAMMING;

        // Performance
        int buffer_chunks = 10;            // Pre-allocated chunks
        bool use_zero_padding = true;      // Pad last chunk with zeros
        bool normalize_energy = false;     // Normalize chunk energy
    };

    struct AudioChunk {
        std::vector<float> data;           // Audio samples
        int64_t timestamp_ms;              // Start timestamp
        int duration_ms;                   // Actual duration
        int sample_count;                  // Number of samples

        // Metadata
        float energy;                      // RMS energy
        float peak_amplitude;              // Peak sample value
        bool is_final;                     // Last chunk in stream
        int chunk_index;                   // Sequential index

        // Overlap info
        int overlap_samples;               // Number of overlapping samples
        float crossfade_ratio;             // For smooth transitions
    };

    using ChunkCallback = std::function<void(const AudioChunk&)>;
    using CompletionCallback = std::function<void()>;

    explicit AudioChunker(const ChunkConfig& config = ChunkConfig());
    ~AudioChunker();

    // Non-copyable
    AudioChunker(const AudioChunker&) = delete;
    AudioChunker& operator=(const AudioChunker&) = delete;

    /**
     * Process audio data into chunks
     * @param data Input audio samples
     * @param size Number of samples
     * @param timestamp_ms Starting timestamp
     */
    void process(const float* data, size_t size, int64_t timestamp_ms = 0);

    /**
     * Process and get chunks synchronously
     */
    std::vector<AudioChunk> process_sync(const float* data, size_t size, int64_t timestamp_ms = 0);

    /**
     * Flush remaining data as final chunk
     */
    void flush();

    /**
     * Reset chunker state
     */
    void reset();

    /**
     * Set chunk callback for streaming processing
     */
    void set_chunk_callback(ChunkCallback callback);

    /**
     * Set completion callback
     */
    void set_completion_callback(CompletionCallback callback);

    /**
     * Get statistics
     */
    struct Stats {
        size_t total_samples_processed = 0;
        size_t total_chunks_created = 0;
        float avg_chunk_energy = 0.0f;
        float avg_overlap_ratio = 0.0f;
        std::chrono::milliseconds processing_time;
    };
    Stats get_stats() const;

    /**
     * Get optimal chunk size for given duration
     */
    static size_t calculate_chunk_size(int duration_ms, int sample_rate, int channels);

    /**
     * Calculate overlap samples
     */
    static size_t calculate_overlap_samples(int overlap_ms, int sample_rate, int channels);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
    ChunkConfig config_;
};

/**
 * Advanced chunker with VAD-aware segmentation
 */
class SmartAudioChunker {
public:
    struct SmartChunkConfig {
        // Base chunking
        int target_duration_ms = 5000;     // Target chunk duration
        int min_duration_ms = 2000;        // Minimum chunk duration
        int max_duration_ms = 10000;       // Maximum chunk duration
        int overlap_ms = 500;               // Overlap duration

        // VAD settings
        bool enable_vad = true;             // Enable VAD-based segmentation
        float vad_threshold = 0.3f;         // VAD confidence threshold
        int min_silence_ms = 300;          // Minimum silence for split
        int max_silence_ms = 1000;         // Maximum silence to include

        // Smart segmentation
        bool split_on_silence = true;      // Split chunks at silence
        bool merge_short_segments = true;  // Merge segments shorter than min
        bool align_to_words = false;       // Try to align to word boundaries

        // Audio settings
        int sample_rate = 16000;
        int channels = 1;

        // Performance
        int lookahead_ms = 1000;           // Lookahead for better decisions
        bool use_ml_segmentation = false;  // Use ML model for segmentation
    };

    struct SmartChunk {
        std::vector<float> data;
        int64_t timestamp_ms;
        int duration_ms;

        // Segmentation metadata
        float confidence;                   // Segmentation confidence
        bool contains_speech;               // Has speech content
        std::vector<std::pair<int, int>> speech_segments; // Speech regions

        // Boundary info
        bool clean_start;                  // Starts at silence
        bool clean_end;                    // Ends at silence
        float boundary_quality;            // Quality of chunk boundaries
    };

    using SmartChunkCallback = std::function<void(const SmartChunk&)>;

    explicit SmartAudioChunker(const SmartChunkConfig& config = SmartChunkConfig());
    ~SmartAudioChunker();

    /**
     * Process audio with smart chunking
     */
    void process_smart(const float* data, size_t size, int64_t timestamp_ms = 0);

    /**
     * Process with VAD results
     */
    void process_with_vad(
        const float* data,
        size_t size,
        const std::vector<bool>& vad_results,
        int64_t timestamp_ms = 0
    );

    /**
     * Get optimal chunk boundaries
     */
    std::vector<std::pair<int, int>> find_optimal_boundaries(
        const float* data,
        size_t size,
        const std::vector<bool>& vad_results
    );

    /**
     * Set callbacks
     */
    void set_chunk_callback(SmartChunkCallback callback);

    /**
     * Get chunking quality metrics
     */
    struct QualityMetrics {
        float avg_boundary_quality = 0.0f;
        float speech_coverage = 0.0f;
        float chunk_duration_variance = 0.0f;
        int total_chunks = 0;
        int speech_chunks = 0;
        int silence_chunks = 0;
    };
    QualityMetrics get_quality_metrics() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
    SmartChunkConfig config_;
};

/**
 * Ring buffer for efficient audio buffering
 */
class AudioRingBuffer {
public:
    explicit AudioRingBuffer(size_t capacity);
    ~AudioRingBuffer();

    /**
     * Write samples to buffer
     */
    bool write(const float* data, size_t size);

    /**
     * Read samples from buffer
     */
    bool read(float* data, size_t size);

    /**
     * Peek at samples without removing
     */
    bool peek(float* data, size_t size) const;

    /**
     * Skip samples
     */
    bool skip(size_t size);

    /**
     * Get available samples
     */
    size_t available() const;

    /**
     * Get free space
     */
    size_t free_space() const;

    /**
     * Clear buffer
     */
    void clear();

    /**
     * Check if empty
     */
    bool empty() const;

    /**
     * Check if full
     */
    bool full() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace audio_processor