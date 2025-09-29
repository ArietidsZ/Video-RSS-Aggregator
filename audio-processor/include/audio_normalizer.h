#pragma once

#include <vector>
#include <memory>
#include <string>
#include <atomic>

namespace audio_processor {

/**
 * Audio format normalization for consistent processing
 * Target: 16kHz, mono, float32
 */
class AudioNormalizer {
public:
    struct NormalizationConfig {
        // Target format
        int target_sample_rate = 16000;    // 16kHz for speech processing
        int target_channels = 1;            // Mono
        enum SampleFormat {
            FLOAT32,
            INT16,
            INT32
        } target_format = FLOAT32;

        // Normalization settings
        float target_loudness_lufs = -23.0f;  // EBU R128 standard
        float max_peak_db = -1.0f;            // Maximum peak level
        float gate_threshold_lufs = -50.0f;   // Gate threshold

        // Dynamic range control
        bool enable_compression = false;
        float compression_ratio = 4.0f;       // 4:1 compression
        float compression_threshold_db = -20.0f;
        float compression_knee_db = 2.0f;
        float compression_attack_ms = 10.0f;
        float compression_release_ms = 100.0f;

        // Noise reduction
        bool enable_noise_reduction = true;
        float noise_gate_threshold_db = -40.0f;
        float noise_reduction_amount_db = 12.0f;

        // Equalization
        bool enable_eq = false;
        std::vector<float> eq_frequencies = {100, 1000, 5000, 10000};
        std::vector<float> eq_gains_db = {0, 0, 0, 0};

        // Quality settings
        enum ResampleQuality {
            FAST,       // Linear interpolation
            MEDIUM,     // Cubic interpolation
            HIGH,       // Sinc interpolation
            ULTRA       // Ultra-high quality sinc
        } resample_quality = HIGH;

        // Performance
        bool use_gpu = true;
        int processing_threads = 4;
        bool enable_simd = true;
    };

    struct NormalizationResult {
        std::vector<float> data;           // Normalized audio
        int sample_rate;                   // Output sample rate
        int channels;                      // Output channels

        // Loudness metrics
        float integrated_loudness_lufs;    // Integrated loudness
        float loudness_range_lu;           // Loudness range
        float true_peak_db;                // True peak level
        float short_term_loudness_lufs;    // Short-term loudness

        // Processing info
        float gain_applied_db;             // Total gain applied
        bool clipping_detected;            // Clipping occurred
        int clipped_samples;               // Number of clipped samples
        float processing_time_ms;          // Processing time
    };

    explicit AudioNormalizer(const NormalizationConfig& config = NormalizationConfig());
    ~AudioNormalizer();

    // Non-copyable
    AudioNormalizer(const AudioNormalizer&) = delete;
    AudioNormalizer& operator=(const AudioNormalizer&) = delete;

    /**
     * Normalize audio to target format
     * @param data Input audio samples
     * @param size Number of samples
     * @param input_sample_rate Input sample rate
     * @param input_channels Input channel count
     * @return Normalized audio
     */
    NormalizationResult normalize(
        const float* data,
        size_t size,
        int input_sample_rate,
        int input_channels
    );

    /**
     * Normalize with custom loudness target
     */
    NormalizationResult normalize_to_lufs(
        const float* data,
        size_t size,
        int input_sample_rate,
        int input_channels,
        float target_lufs
    );

    /**
     * Batch normalization for multiple audio segments
     */
    std::vector<NormalizationResult> normalize_batch(
        const std::vector<std::vector<float>>& segments,
        int input_sample_rate,
        int input_channels
    );

    /**
     * Calculate loudness metrics (EBU R128)
     */
    struct LoudnessMetrics {
        float integrated_lufs;
        float short_term_lufs;
        float momentary_lufs;
        float loudness_range_lu;
        float true_peak_db;
        std::vector<float> short_term_history;
    };
    static LoudnessMetrics calculate_loudness(
        const float* data,
        size_t size,
        int sample_rate
    );

    /**
     * Apply gain with limiter
     */
    static void apply_gain_with_limiter(
        float* data,
        size_t size,
        float gain_db,
        float limit_db = -0.1f
    );

    /**
     * Get normalization statistics
     */
    struct Stats {
        size_t total_samples_processed = 0;
        float avg_gain_applied_db = 0.0f;
        float avg_loudness_lufs = 0.0f;
        size_t clipping_events = 0;
        float processing_speed_ratio = 0.0f;  // Speed vs real-time
    };
    Stats get_stats() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
    NormalizationConfig config_;
};

/**
 * Fast resampler for sample rate conversion
 */
class AudioResampler {
public:
    enum Quality {
        LINEAR,         // Fastest, lowest quality
        CUBIC,          // Good balance
        SINC_FAST,      // High quality, fast
        SINC_MEDIUM,    // Higher quality
        SINC_BEST       // Best quality, slowest
    };

    struct ResampleConfig {
        Quality quality = SINC_FAST;
        int filter_size = 32;          // For sinc resampling
        float cutoff_frequency = 0.95f; // Normalized cutoff
        bool enable_antialiasing = true;
        bool use_gpu = true;
    };

    explicit AudioResampler(const ResampleConfig& config = ResampleConfig());
    ~AudioResampler();

    /**
     * Resample audio
     */
    std::vector<float> resample(
        const float* data,
        size_t size,
        int input_rate,
        int output_rate
    );

    /**
     * Get resampling ratio
     */
    static float get_resample_ratio(int input_rate, int output_rate) {
        return static_cast<float>(output_rate) / input_rate;
    }

    /**
     * Calculate output size
     */
    static size_t calculate_output_size(
        size_t input_size,
        int input_rate,
        int output_rate
    ) {
        return static_cast<size_t>(input_size * get_resample_ratio(input_rate, output_rate));
    }

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
    ResampleConfig config_;
};

/**
 * Channel mixer for mono/stereo conversion
 */
class ChannelMixer {
public:
    enum MixMode {
        AVERAGE,        // Average channels
        LEFT_ONLY,      // Use left channel only
        RIGHT_ONLY,     // Use right channel only
        MID_SIDE,       // Mid-side processing
        CUSTOM          // Custom mixing matrix
    };

    struct MixConfig {
        MixMode mode = AVERAGE;
        std::vector<float> mix_matrix;  // Custom mixing matrix
        float stereo_width = 1.0f;      // For stereo widening
        bool normalize = true;           // Normalize after mixing
    };

    explicit ChannelMixer(const MixConfig& config = MixConfig());
    ~ChannelMixer();

    /**
     * Convert stereo to mono
     */
    std::vector<float> stereo_to_mono(const float* data, size_t size);

    /**
     * Convert mono to stereo
     */
    std::vector<float> mono_to_stereo(const float* data, size_t size);

    /**
     * Mix multiple channels to target channel count
     */
    std::vector<float> mix_channels(
        const float* data,
        size_t size,
        int input_channels,
        int output_channels
    );

    /**
     * Apply custom channel mixing matrix
     */
    std::vector<float> apply_mix_matrix(
        const float* data,
        size_t size,
        int input_channels,
        int output_channels,
        const std::vector<float>& matrix
    );

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
    MixConfig config_;
};

/**
 * Dynamic range compressor
 */
class DynamicRangeCompressor {
public:
    struct CompressorConfig {
        float threshold_db = -20.0f;
        float ratio = 4.0f;           // 4:1 compression
        float knee_db = 2.0f;          // Soft knee width
        float attack_ms = 10.0f;
        float release_ms = 100.0f;
        float makeup_gain_db = 0.0f;
        float lookahead_ms = 5.0f;

        // Sidechain
        bool enable_sidechain = false;
        float sidechain_hpf_hz = 0.0f;  // High-pass filter

        // Adaptive release
        bool adaptive_release = true;
        float release_zone_1 = 50.0f;   // Fast release zone
        float release_zone_2 = 100.0f;  // Medium release zone
        float release_zone_3 = 200.0f;  // Slow release zone
        float release_zone_4 = 400.0f;  // Very slow release zone
    };

    explicit DynamicRangeCompressor(const CompressorConfig& config = CompressorConfig());
    ~DynamicRangeCompressor();

    /**
     * Apply compression
     */
    void process(float* data, size_t size, int sample_rate);

    /**
     * Process with sidechain input
     */
    void process_with_sidechain(
        float* data,
        size_t size,
        const float* sidechain,
        int sample_rate
    );

    /**
     * Get gain reduction in dB
     */
    float get_gain_reduction_db() const;

    /**
     * Reset compressor state
     */
    void reset();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
    CompressorConfig config_;
};

/**
 * Noise gate for removing background noise
 */
class NoiseGate {
public:
    struct GateConfig {
        float threshold_db = -40.0f;
        float hysteresis_db = 3.0f;     // Hysteresis to prevent chatter
        float attack_ms = 0.1f;          // Gate opening time
        float hold_ms = 10.0f;           // Hold time after signal drops
        float release_ms = 100.0f;       // Gate closing time
        float range_db = -60.0f;        // Maximum attenuation

        // Frequency-selective gating
        bool enable_sidechain_filter = false;
        float sidechain_hpf_hz = 100.0f;
        float sidechain_lpf_hz = 8000.0f;

        // Lookahead
        float lookahead_ms = 0.0f;
    };

    explicit NoiseGate(const GateConfig& config = GateConfig());
    ~NoiseGate();

    /**
     * Apply noise gate
     */
    void process(float* data, size_t size, int sample_rate);

    /**
     * Get gate state (0.0 = closed, 1.0 = open)
     */
    float get_gate_state() const;

    /**
     * Reset gate state
     */
    void reset();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
    GateConfig config_;
};

} // namespace audio_processor