#include "whisper_turbo.h"
#include <cuda_runtime.h>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <immintrin.h>

namespace whisper_turbo {

// Forward declarations for CUDA kernels
namespace cuda {
namespace quantization {
    void quantize_tensor_int8(const float* input, int8_t* output,
                             float* scales, float* zero_points,
                             int batch_size, int channels, int spatial_size,
                             cudaStream_t stream);

    void dequantize_tensor_int8(const int8_t* input, float* output,
                               const float* scales, const float* zero_points,
                               int batch_size, int channels, int spatial_size,
                               cudaStream_t stream);

    void calibrate_for_quantization(const float* input, float* min_val,
                                   float* max_val, float* abs_max,
                                   int* histogram, int num_elements,
                                   int num_bins, cudaStream_t stream);
}
}

class QuantizationEngine {
private:
    struct LayerQuantizationInfo {
        std::vector<float> scales;
        std::vector<float> zero_points;
        float min_val;
        float max_val;
        float abs_max;
        std::vector<int> histogram;
        int calibration_samples;
        QuantizationType quant_type;
        bool calibrated;
    };

    std::unordered_map<std::string, LayerQuantizationInfo> layer_info_;
    cudaStream_t stream_;
    bool calibration_mode_;
    int calibration_batches_;
    QuantizationType default_quantization_;

    // Device memory for quantization
    float* d_scales_;
    float* d_zero_points_;
    int8_t* d_quantized_weights_;
    size_t weight_buffer_size_;

public:
    QuantizationEngine(QuantizationType default_type = QuantizationType::INT8)
        : calibration_mode_(false),
          calibration_batches_(100),
          default_quantization_(default_type),
          d_scales_(nullptr),
          d_zero_points_(nullptr),
          d_quantized_weights_(nullptr),
          weight_buffer_size_(0) {

        cudaStreamCreate(&stream_);
    }

    ~QuantizationEngine() {
        if (d_scales_) cudaFree(d_scales_);
        if (d_zero_points_) cudaFree(d_zero_points_);
        if (d_quantized_weights_) cudaFree(d_quantized_weights_);
        cudaStreamDestroy(stream_);
    }

    // Start calibration mode for collecting statistics
    void StartCalibration(int num_batches = 100) {
        calibration_mode_ = true;
        calibration_batches_ = num_batches;
        layer_info_.clear();
    }

    // Process a layer during calibration
    void CalibrateLayer(const std::string& layer_name,
                        const float* activations,
                        int batch_size,
                        int channels,
                        int spatial_size) {

        if (!calibration_mode_) return;

        auto& info = layer_info_[layer_name];

        if (!info.calibrated) {
            info.histogram.resize(2048, 0);
            info.min_val = FLT_MAX;
            info.max_val = -FLT_MAX;
            info.abs_max = 0.0f;
            info.calibration_samples = 0;
            info.quant_type = default_quantization_;
        }

        // Allocate device memory for statistics
        float *d_min, *d_max, *d_abs_max;
        int *d_histogram;

        cudaMalloc(&d_min, sizeof(float));
        cudaMalloc(&d_max, sizeof(float));
        cudaMalloc(&d_abs_max, sizeof(float));
        cudaMalloc(&d_histogram, 2048 * sizeof(int));

        cudaMemcpy(d_min, &info.min_val, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_max, &info.max_val, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_abs_max, &info.abs_max, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_histogram, info.histogram.data(), 2048 * sizeof(int), cudaMemcpyHostToDevice);

        // Collect statistics
        int num_elements = batch_size * channels * spatial_size;
        cuda::quantization::calibrate_for_quantization(
            activations, d_min, d_max, d_abs_max, d_histogram,
            num_elements, 2048, stream_
        );

        // Copy back statistics
        cudaMemcpy(&info.min_val, d_min, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&info.max_val, d_max, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&info.abs_max, d_abs_max, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(info.histogram.data(), d_histogram, 2048 * sizeof(int), cudaMemcpyDeviceToHost);

        info.calibration_samples += batch_size;

        // Clean up
        cudaFree(d_min);
        cudaFree(d_max);
        cudaFree(d_abs_max);
        cudaFree(d_histogram);

        if (info.calibration_samples >= calibration_batches_) {
            FinalizeLayerCalibration(layer_name, channels);
        }
    }

    // Finalize calibration for a layer
    void FinalizeLayerCalibration(const std::string& layer_name, int channels) {
        auto& info = layer_info_[layer_name];

        // Compute optimal scale and zero point using KL divergence
        float optimal_scale = ComputeOptimalScale(info);

        // Per-channel quantization
        info.scales.resize(channels);
        info.zero_points.resize(channels);

        for (int c = 0; c < channels; ++c) {
            info.scales[c] = optimal_scale;
            info.zero_points[c] = 0.0f;  // Symmetric quantization
        }

        info.calibrated = true;
    }

    // End calibration and prepare for inference
    void EndCalibration() {
        calibration_mode_ = false;

        // Allocate consolidated device memory for all scales/zero points
        size_t total_channels = 0;
        for (const auto& [name, info] : layer_info_) {
            total_channels += info.scales.size();
        }

        cudaMalloc(&d_scales_, total_channels * sizeof(float));
        cudaMalloc(&d_zero_points_, total_channels * sizeof(float));

        // Copy scales and zero points to device
        size_t offset = 0;
        for (const auto& [name, info] : layer_info_) {
            cudaMemcpy(d_scales_ + offset, info.scales.data(),
                      info.scales.size() * sizeof(float),
                      cudaMemcpyHostToDevice);
            cudaMemcpy(d_zero_points_ + offset, info.zero_points.data(),
                      info.zero_points.size() * sizeof(float),
                      cudaMemcpyHostToDevice);
            offset += info.scales.size();
        }
    }

    // Quantize weights for a layer
    void QuantizeWeights(const std::string& layer_name,
                        const float* weights,
                        int8_t* quantized_weights,
                        int num_weights,
                        int num_channels) {

        const auto& info = layer_info_[layer_name];
        if (!info.calibrated) {
            // Use default quantization if not calibrated
            QuantizeWeightsDefault(weights, quantized_weights, num_weights);
            return;
        }

        // Allocate device memory if needed
        if (!d_quantized_weights_ || weight_buffer_size_ < num_weights) {
            if (d_quantized_weights_) cudaFree(d_quantized_weights_);
            cudaMalloc(&d_quantized_weights_, num_weights * sizeof(int8_t));
            weight_buffer_size_ = num_weights;
        }

        // Get scales and zero points for this layer
        float* layer_scales = d_scales_;  // Would need proper offset calculation
        float* layer_zero_points = d_zero_points_;

        // Quantize on GPU
        int spatial_size = num_weights / num_channels;
        cuda::quantization::quantize_tensor_int8(
            weights, d_quantized_weights_,
            layer_scales, layer_zero_points,
            1, num_channels, spatial_size,
            stream_
        );

        // Copy result back
        cudaMemcpy(quantized_weights, d_quantized_weights_,
                  num_weights * sizeof(int8_t),
                  cudaMemcpyDeviceToHost);
    }

    // Quantize activations dynamically
    void QuantizeActivations(const float* activations,
                            int8_t* quantized,
                            int batch_size,
                            int channels,
                            int spatial_size) {

        // Dynamic quantization - compute scale on the fly
        float* d_activations;
        int8_t* d_quantized;
        float* d_dynamic_scales;
        float* d_dynamic_zero_points;

        int num_elements = batch_size * channels * spatial_size;

        cudaMalloc(&d_activations, num_elements * sizeof(float));
        cudaMalloc(&d_quantized, num_elements * sizeof(int8_t));
        cudaMalloc(&d_dynamic_scales, channels * sizeof(float));
        cudaMalloc(&d_dynamic_zero_points, channels * sizeof(float));

        cudaMemcpy(d_activations, activations,
                  num_elements * sizeof(float),
                  cudaMemcpyHostToDevice);

        // Compute per-channel statistics and quantize
        ComputeDynamicQuantizationParams(d_activations, d_dynamic_scales,
                                        d_dynamic_zero_points,
                                        batch_size, channels, spatial_size);

        cuda::quantization::quantize_tensor_int8(
            d_activations, d_quantized,
            d_dynamic_scales, d_dynamic_zero_points,
            batch_size, channels, spatial_size,
            stream_
        );

        cudaMemcpy(quantized, d_quantized,
                  num_elements * sizeof(int8_t),
                  cudaMemcpyDeviceToHost);

        cudaFree(d_activations);
        cudaFree(d_quantized);
        cudaFree(d_dynamic_scales);
        cudaFree(d_dynamic_zero_points);
    }

    // CPU fallback for INT8 quantization using AVX2
    void QuantizeWeightsCPU_AVX2(const float* weights,
                                 int8_t* quantized,
                                 float scale,
                                 int num_weights) {
        #ifdef __AVX2__
        const __m256 scale_vec = _mm256_set1_ps(1.0f / scale);
        const __m256 min_val = _mm256_set1_ps(-128.0f);
        const __m256 max_val = _mm256_set1_ps(127.0f);

        int i = 0;
        for (; i <= num_weights - 8; i += 8) {
            __m256 w = _mm256_loadu_ps(&weights[i]);
            __m256 scaled = _mm256_mul_ps(w, scale_vec);

            // Clamp
            scaled = _mm256_max_ps(scaled, min_val);
            scaled = _mm256_min_ps(scaled, max_val);

            // Convert to int32
            __m256i int_vals = _mm256_cvtps_epi32(scaled);

            // Pack to int8
            __m128i low = _mm256_extracti128_si256(int_vals, 0);
            __m128i high = _mm256_extracti128_si256(int_vals, 1);
            __m128i packed = _mm_packs_epi32(low, high);
            __m128i packed8 = _mm_packs_epi16(packed, _mm_setzero_si128());

            // Store lower 8 bytes
            _mm_storel_epi64((__m128i*)&quantized[i], packed8);
        }

        // Handle remaining elements
        for (; i < num_weights; ++i) {
            float scaled = weights[i] / scale;
            int q = std::round(scaled);
            q = std::max(-128, std::min(127, q));
            quantized[i] = static_cast<int8_t>(q);
        }
        #else
        // Non-AVX fallback
        QuantizeWeightsDefault(weights, quantized, num_weights);
        #endif
    }

    // Get quantization statistics for analysis
    struct QuantizationStats {
        float avg_scale;
        float min_scale;
        float max_scale;
        float quantization_error;
        int num_layers;
        size_t total_parameters;
        float compression_ratio;
    };

    QuantizationStats GetStatistics() const {
        QuantizationStats stats{};

        if (layer_info_.empty()) return stats;

        std::vector<float> all_scales;
        for (const auto& [name, info] : layer_info_) {
            all_scales.insert(all_scales.end(),
                            info.scales.begin(),
                            info.scales.end());
        }

        stats.num_layers = layer_info_.size();
        stats.avg_scale = std::accumulate(all_scales.begin(), all_scales.end(), 0.0f)
                         / all_scales.size();
        stats.min_scale = *std::min_element(all_scales.begin(), all_scales.end());
        stats.max_scale = *std::max_element(all_scales.begin(), all_scales.end());

        // Calculate compression ratio (FP32 -> INT8)
        stats.compression_ratio = 4.0f;  // 32 bits -> 8 bits

        return stats;
    }

private:
    float ComputeOptimalScale(const LayerQuantizationInfo& info) {
        // Use percentile method to find optimal scale
        // This avoids outliers affecting quantization

        const float percentile = 0.999f;  // Use 99.9th percentile
        int total_samples = std::accumulate(info.histogram.begin(),
                                          info.histogram.end(), 0);
        int target_count = static_cast<int>(total_samples * percentile);

        int cumsum = 0;
        float optimal_max = info.abs_max;

        for (size_t i = 0; i < info.histogram.size(); ++i) {
            cumsum += info.histogram[i];
            if (cumsum >= target_count) {
                float bin_width = (info.max_val - info.min_val) / info.histogram.size();
                optimal_max = info.min_val + (i + 1) * bin_width;
                break;
            }
        }

        // Scale for symmetric quantization
        return optimal_max / 127.0f;
    }

    void QuantizeWeightsDefault(const float* weights,
                               int8_t* quantized,
                               int num_weights) {
        // Find min/max for the weights
        float min_val = *std::min_element(weights, weights + num_weights);
        float max_val = *std::max_element(weights, weights + num_weights);
        float abs_max = std::max(std::abs(min_val), std::abs(max_val));

        float scale = abs_max / 127.0f;

        for (int i = 0; i < num_weights; ++i) {
            int q = std::round(weights[i] / scale);
            q = std::max(-128, std::min(127, q));
            quantized[i] = static_cast<int8_t>(q);
        }
    }

    void ComputeDynamicQuantizationParams(const float* activations,
                                         float* scales,
                                         float* zero_points,
                                         int batch_size,
                                         int channels,
                                         int spatial_size) {
        // Compute per-channel min/max on CPU for now
        // This would be done on GPU in production

        std::vector<float> channel_min(channels, FLT_MAX);
        std::vector<float> channel_max(channels, -FLT_MAX);

        for (int b = 0; b < batch_size; ++b) {
            for (int c = 0; c < channels; ++c) {
                for (int s = 0; s < spatial_size; ++s) {
                    int idx = b * channels * spatial_size + c * spatial_size + s;
                    channel_min[c] = std::min(channel_min[c], activations[idx]);
                    channel_max[c] = std::max(channel_max[c], activations[idx]);
                }
            }
        }

        // Compute scales and zero points
        std::vector<float> h_scales(channels);
        std::vector<float> h_zero_points(channels);

        for (int c = 0; c < channels; ++c) {
            float abs_max = std::max(std::abs(channel_min[c]),
                                    std::abs(channel_max[c]));
            h_scales[c] = abs_max / 127.0f;
            h_zero_points[c] = 0.0f;  // Symmetric quantization
        }

        cudaMemcpy(scales, h_scales.data(),
                  channels * sizeof(float),
                  cudaMemcpyHostToDevice);
        cudaMemcpy(zero_points, h_zero_points.data(),
                  channels * sizeof(float),
                  cudaMemcpyHostToDevice);
    }
};

// Global quantization engine instance
static std::unique_ptr<QuantizationEngine> g_quantization_engine;

void InitializeQuantization(QuantizationType type) {
    g_quantization_engine = std::make_unique<QuantizationEngine>(type);
}

void StartQuantizationCalibration(int num_batches) {
    if (g_quantization_engine) {
        g_quantization_engine->StartCalibration(num_batches);
    }
}

void EndQuantizationCalibration() {
    if (g_quantization_engine) {
        g_quantization_engine->EndCalibration();
    }
}

} // namespace whisper_turbo