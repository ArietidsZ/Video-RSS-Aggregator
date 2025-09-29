#pragma once

#include <vector>
#include <memory>
#include <functional>
#include <string>
#include <atomic>

namespace audio_processor {

/**
 * Voice Activity Detection with WebRTCVAD + SileroVAD fusion
 * Achieves high accuracy with low latency
 */
class VADProcessor {
public:
    enum class VADModel {
        WebRTCVAD,      // Fast, lightweight VAD
        SileroVAD,      // ML-based, high accuracy
        Fusion          // Combination of both
    };

    struct VADConfig {
        VADModel model = VADModel::Fusion;

        // WebRTCVAD settings
        int webrtc_mode = 3;              // 0-3, higher = more aggressive
        int frame_duration_ms = 30;       // 10, 20, or 30ms frames

        // SileroVAD settings
        std::string silero_model_path;    // Path to ONNX model
        float silero_threshold = 0.5f;    // Detection threshold
        int silero_min_speech_ms = 250;   // Minimum speech duration
        int silero_min_silence_ms = 100;  // Minimum silence duration

        // Fusion settings
        float fusion_weight_webrtc = 0.3f;
        float fusion_weight_silero = 0.7f;
        float fusion_threshold = 0.5f;

        // Common settings
        int sample_rate = 16000;
        int channels = 1;

        // Smoothing
        bool enable_smoothing = true;
        int smoothing_window_ms = 150;    // Smoothing window
        float speech_pad_ms = 300;        // Padding around speech

        // Performance
        bool use_gpu = true;              // GPU acceleration for Silero
        int batch_size = 32;              // Batch processing size
    };

    struct VADResult {
        std::vector<bool> is_speech;      // Per-frame speech detection
        std::vector<float> confidence;    // Confidence scores
        std::vector<std::pair<int, int>> segments; // Speech segments (start, end)

        // Statistics
        float speech_ratio;                // Ratio of speech frames
        float avg_confidence;              // Average confidence
        int num_segments;                  // Number of speech segments
        float total_speech_ms;             // Total speech duration

        // Detailed info
        std::vector<float> energy;        // Frame energy
        std::vector<float> zero_crossing_rate; // ZCR
    };

    using VADCallback = std::function<void(const VADResult&)>;
    using FrameCallback = std::function<void(bool is_speech, float confidence)>;

    explicit VADProcessor(const VADConfig& config = VADConfig());
    ~VADProcessor();

    // Non-copyable
    VADProcessor(const VADProcessor&) = delete;
    VADProcessor& operator=(const VADProcessor&) = delete;

    /**
     * Initialize VAD models
     */
    bool initialize();

    /**
     * Process audio for VAD
     * @param data Audio samples
     * @param size Number of samples
     * @return VAD results
     */
    VADResult process(const float* data, size_t size);

    /**
     * Process audio in streaming mode
     */
    void process_stream(const float* data, size_t size, FrameCallback callback);

    /**
     * Process batch of audio segments
     */
    std::vector<VADResult> process_batch(const std::vector<std::vector<float>>& segments);

    /**
     * Reset VAD state
     */
    void reset();

    /**
     * Get current configuration
     */
    VADConfig get_config() const;

    /**
     * Update configuration
     */
    void update_config(const VADConfig& config);

    /**
     * Get performance statistics
     */
    struct PerformanceStats {
        float avg_processing_time_ms = 0.0f;
        float real_time_factor = 0.0f;   // Processing speed vs real-time
        size_t frames_processed = 0;
        size_t total_samples = 0;
        float gpu_utilization = 0.0f;
    };
    PerformanceStats get_performance_stats() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
    VADConfig config_;
    std::atomic<bool> initialized_{false};
};

/**
 * Advanced VAD with multi-model ensemble
 */
class EnsembleVAD {
public:
    struct EnsembleConfig {
        // Models to use
        bool use_webrtc = true;
        bool use_silero = true;
        bool use_energy = true;
        bool use_spectral = true;

        // Ensemble method
        enum Method {
            VOTING,         // Majority voting
            AVERAGING,      // Average confidence
            WEIGHTED,       // Weighted average
            STACKING        // Meta-model stacking
        } method = WEIGHTED;

        // Weights for weighted averaging
        float weight_webrtc = 0.2f;
        float weight_silero = 0.4f;
        float weight_energy = 0.2f;
        float weight_spectral = 0.2f;

        // Voting threshold
        int min_votes = 2;

        // Common settings
        int sample_rate = 16000;
        float threshold = 0.5f;
    };

    struct EnsembleResult {
        std::vector<bool> final_decision;
        std::vector<float> ensemble_confidence;

        // Individual model results
        std::vector<bool> webrtc_decision;
        std::vector<bool> silero_decision;
        std::vector<bool> energy_decision;
        std::vector<bool> spectral_decision;

        // Confidence scores
        std::vector<float> webrtc_confidence;
        std::vector<float> silero_confidence;
        std::vector<float> energy_confidence;
        std::vector<float> spectral_confidence;

        // Agreement metrics
        float model_agreement;     // How much models agree
        float decision_confidence; // Confidence in final decision
    };

    explicit EnsembleVAD(const EnsembleConfig& config = EnsembleConfig());
    ~EnsembleVAD();

    /**
     * Process with ensemble
     */
    EnsembleResult process(const float* data, size_t size);

    /**
     * Get individual model predictions
     */
    struct ModelPredictions {
        std::string model_name;
        std::vector<bool> predictions;
        std::vector<float> confidences;
        float accuracy;  // If ground truth available
    };
    std::vector<ModelPredictions> get_model_predictions(const float* data, size_t size);

    /**
     * Train stacking meta-model
     */
    bool train_stacking_model(
        const std::vector<std::vector<float>>& audio_samples,
        const std::vector<std::vector<bool>>& ground_truth
    );

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
    EnsembleConfig config_;
};

/**
 * Energy-based VAD for fast processing
 */
class EnergyVAD {
public:
    struct EnergyConfig {
        float energy_threshold = -35.0f;  // dB threshold
        float zcr_threshold = 0.25f;      // Zero-crossing rate threshold
        int frame_size = 512;              // Frame size in samples
        int frame_shift = 256;             // Frame shift in samples

        // Adaptive threshold
        bool adaptive_threshold = true;
        float noise_floor_db = -50.0f;
        float snr_threshold = 10.0f;      // SNR threshold in dB

        // Smoothing
        int median_filter_size = 5;       // Median filter window
        int min_speech_frames = 10;       // Minimum consecutive frames
    };

    explicit EnergyVAD(const EnergyConfig& config = EnergyConfig());
    ~EnergyVAD();

    /**
     * Process with energy-based VAD
     */
    std::vector<bool> process(const float* data, size_t size);

    /**
     * Calculate frame energy in dB
     */
    static float calculate_energy_db(const float* frame, size_t size);

    /**
     * Calculate zero-crossing rate
     */
    static float calculate_zcr(const float* frame, size_t size);

    /**
     * Update noise floor estimate
     */
    void update_noise_floor(const float* noise_samples, size_t size);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
    EnergyConfig config_;
};

/**
 * Spectral-based VAD using frequency domain features
 */
class SpectralVAD {
public:
    struct SpectralConfig {
        // Spectral features
        bool use_spectral_centroid = true;
        bool use_spectral_flux = true;
        bool use_spectral_rolloff = true;
        bool use_mfcc = false;

        // Thresholds
        float centroid_threshold = 2000.0f;  // Hz
        float flux_threshold = 0.1f;
        float rolloff_threshold = 0.85f;

        // FFT settings
        int fft_size = 1024;
        int hop_size = 512;
        std::string window_type = "hamming";

        // ML classifier (optional)
        bool use_ml_classifier = false;
        std::string classifier_model_path;
    };

    explicit SpectralVAD(const SpectralConfig& config = SpectralConfig());
    ~SpectralVAD();

    /**
     * Process with spectral VAD
     */
    std::vector<bool> process(const float* data, size_t size);

    /**
     * Extract spectral features
     */
    struct SpectralFeatures {
        std::vector<float> spectral_centroid;
        std::vector<float> spectral_flux;
        std::vector<float> spectral_rolloff;
        std::vector<std::vector<float>> mfcc;  // Optional MFCC features
    };
    SpectralFeatures extract_features(const float* data, size_t size);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
    SpectralConfig config_;
};

} // namespace audio_processor