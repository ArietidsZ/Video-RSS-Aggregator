#include "whisper_turbo.h"
#include <cuda_runtime.h>
#include <cudnn.h>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <thread>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <unordered_map>

namespace whisper_turbo {

// Forward declarations for CUDA batch kernels
namespace cuda {
namespace batch {
    void batch_preprocess_audio(const float** audio_batch, float* preprocessed,
                                int batch_size, int num_samples,
                                cudaStream_t stream);

    void batch_encode_transformer(const float* input, float* encoded,
                                  int batch_size, int seq_len, int hidden_dim,
                                  cudaStream_t stream);

    void batch_decode_transformer(const float* encoded, int* tokens,
                                  int batch_size, int max_length,
                                  cudaStream_t stream);
}
}

struct BatchRequest {
    std::string request_id;
    const float* audio_data;
    size_t num_samples;
    uint32_t sample_rate;
    TranscriptionOptions options;
    std::promise<TranscriptionResult> promise;
    std::chrono::steady_clock::time_point submission_time;
    int priority;
};

class DynamicBatchProcessor {
private:
    // Batch configuration
    struct BatchConfig {
        int min_batch_size;
        int max_batch_size;
        int optimal_batch_size;
        int current_batch_size;
        float batch_timeout_ms;
        bool enable_dynamic_batching;
        bool enable_sequence_bucketing;
        bool enable_padding_optimization;
    };

    // Performance metrics
    struct BatchMetrics {
        std::atomic<size_t> total_requests{0};
        std::atomic<size_t> total_batches{0};
        std::atomic<double> avg_batch_size{0};
        std::atomic<double> avg_latency_ms{0};
        std::atomic<double> throughput_samples_per_sec{0};
        std::atomic<double> gpu_utilization{0};
        std::atomic<size_t> padding_waste_samples{0};
    };

    // Sequence bucket for efficient padding
    struct SequenceBucket {
        int min_length;
        int max_length;
        int padded_length;
        std::vector<BatchRequest*> requests;
    };

    BatchConfig config_;
    BatchMetrics metrics_;

    // Request queue and processing
    std::queue<std::unique_ptr<BatchRequest>> pending_requests_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::atomic<bool> processing_enabled_{true};
    std::thread batch_thread_;

    // CUDA resources
    cudaStream_t* streams_;
    int num_streams_;
    cudnnHandle_t cudnn_handle_;

    // Memory pools for batching
    struct MemoryPool {
        float* audio_buffer;
        float* preprocessed_buffer;
        float* encoder_buffer;
        float* decoder_buffer;
        int* token_buffer;
        size_t buffer_size;
    };

    MemoryPool* memory_pools_;
    int num_memory_pools_;
    std::atomic<int> current_pool_{0};

    // Model instance
    WhisperTurbo* model_;

public:
    DynamicBatchProcessor(WhisperTurbo* model, int gpu_id = 0)
        : model_(model), num_streams_(4), num_memory_pools_(2) {

        InitializeConfig();
        InitializeCUDAResources(gpu_id);
        InitializeMemoryPools();

        // Start batch processing thread
        batch_thread_ = std::thread(&DynamicBatchProcessor::ProcessBatches, this);
    }

    ~DynamicBatchProcessor() {
        processing_enabled_ = false;
        queue_cv_.notify_all();
        if (batch_thread_.joinable()) {
            batch_thread_.join();
        }

        CleanupCUDAResources();
        CleanupMemoryPools();
    }

    // Submit request for batch processing
    std::future<TranscriptionResult> SubmitRequest(
        const float* audio_data,
        size_t num_samples,
        uint32_t sample_rate,
        const TranscriptionOptions& options,
        int priority = 0) {

        auto request = std::make_unique<BatchRequest>();
        request->request_id = GenerateRequestId();
        request->audio_data = audio_data;
        request->num_samples = num_samples;
        request->sample_rate = sample_rate;
        request->options = options;
        request->submission_time = std::chrono::steady_clock::now();
        request->priority = priority;

        auto future = request->promise.get_future();

        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            pending_requests_.push(std::move(request));
            metrics_.total_requests++;
        }

        queue_cv_.notify_one();
        return future;
    }

    // Update batch configuration dynamically
    void UpdateBatchConfig(int max_batch_size, float timeout_ms) {
        config_.max_batch_size = max_batch_size;
        config_.batch_timeout_ms = timeout_ms;
        OptimizeBatchSize();
    }

    // Get current metrics
    BatchMetrics GetMetrics() const {
        return metrics_;
    }

private:
    void InitializeConfig() {
        config_.min_batch_size = 1;
        config_.max_batch_size = 32;
        config_.optimal_batch_size = 8;
        config_.current_batch_size = 8;
        config_.batch_timeout_ms = 50.0f;  // 50ms timeout
        config_.enable_dynamic_batching = true;
        config_.enable_sequence_bucketing = true;
        config_.enable_padding_optimization = true;
    }

    void InitializeCUDAResources(int gpu_id) {
        cudaSetDevice(gpu_id);

        // Create streams for concurrent execution
        streams_ = new cudaStream_t[num_streams_];
        for (int i = 0; i < num_streams_; i++) {
            cudaStreamCreateWithFlags(&streams_[i], cudaStreamNonBlocking);
        }

        // Initialize cuDNN for batch operations
        cudnnCreate(&cudnn_handle_);
    }

    void InitializeMemoryPools() {
        memory_pools_ = new MemoryPool[num_memory_pools_];

        // Calculate buffer sizes based on max batch size
        size_t max_audio_samples = 30 * 16000;  // 30 seconds at 16kHz
        size_t max_seq_len = 1500;  // Max sequence length
        size_t hidden_dim = 1024;   // Model hidden dimension

        for (int i = 0; i < num_memory_pools_; i++) {
            auto& pool = memory_pools_[i];

            // Allocate unified memory for zero-copy access
            size_t audio_size = config_.max_batch_size * max_audio_samples * sizeof(float);
            size_t encoder_size = config_.max_batch_size * max_seq_len * hidden_dim * sizeof(float);
            size_t decoder_size = config_.max_batch_size * max_seq_len * hidden_dim * sizeof(float);
            size_t token_size = config_.max_batch_size * max_seq_len * sizeof(int);

            cudaMallocManaged(&pool.audio_buffer, audio_size);
            cudaMallocManaged(&pool.preprocessed_buffer, audio_size);
            cudaMallocManaged(&pool.encoder_buffer, encoder_size);
            cudaMallocManaged(&pool.decoder_buffer, decoder_size);
            cudaMallocManaged(&pool.token_buffer, token_size);

            pool.buffer_size = audio_size + encoder_size + decoder_size + token_size;
        }
    }

    void ProcessBatches() {
        while (processing_enabled_) {
            std::vector<std::unique_ptr<BatchRequest>> batch;

            {
                std::unique_lock<std::mutex> lock(queue_mutex_);

                // Wait for requests or timeout
                auto timeout_point = std::chrono::steady_clock::now() +
                    std::chrono::milliseconds(static_cast<int>(config_.batch_timeout_ms));

                queue_cv_.wait_until(lock, timeout_point, [this]() {
                    return !pending_requests_.empty() || !processing_enabled_;
                });

                if (!processing_enabled_) break;

                // Collect batch based on dynamic strategy
                batch = CollectBatch();
            }

            if (!batch.empty()) {
                ProcessSingleBatch(batch);
                metrics_.total_batches++;
                metrics_.avg_batch_size = (metrics_.avg_batch_size * 0.9) +
                                         (batch.size() * 0.1);
            }
        }
    }

    std::vector<std::unique_ptr<BatchRequest>> CollectBatch() {
        std::vector<std::unique_ptr<BatchRequest>> batch;

        if (config_.enable_sequence_bucketing) {
            batch = CollectBucketedBatch();
        } else {
            batch = CollectSimpleBatch();
        }

        // Dynamic batch size adjustment based on queue depth
        if (config_.enable_dynamic_batching) {
            AdjustBatchSizeDynamically();
        }

        return batch;
    }

    std::vector<std::unique_ptr<BatchRequest>> CollectSimpleBatch() {
        std::vector<std::unique_ptr<BatchRequest>> batch;

        while (!pending_requests_.empty() &&
               batch.size() < static_cast<size_t>(config_.current_batch_size)) {
            batch.push_back(std::move(pending_requests_.front()));
            pending_requests_.pop();
        }

        return batch;
    }

    std::vector<std::unique_ptr<BatchRequest>> CollectBucketedBatch() {
        // Group requests by sequence length for efficient padding
        std::unordered_map<int, std::vector<std::unique_ptr<BatchRequest>>> buckets;

        while (!pending_requests_.empty() &&
               buckets.size() < static_cast<size_t>(config_.current_batch_size)) {
            auto request = std::move(pending_requests_.front());
            pending_requests_.pop();

            // Bucket by 5-second intervals
            int bucket_id = (request->num_samples / 80000) * 80000;
            buckets[bucket_id].push_back(std::move(request));
        }

        // Select the bucket with most requests
        std::vector<std::unique_ptr<BatchRequest>> batch;
        int max_bucket_size = 0;

        for (auto& [bucket_id, bucket_requests] : buckets) {
            if (bucket_requests.size() > max_bucket_size) {
                max_bucket_size = bucket_requests.size();
                batch = std::move(bucket_requests);
            }
        }

        // Put unused buckets back in queue
        for (auto& [bucket_id, bucket_requests] : buckets) {
            for (auto& request : bucket_requests) {
                pending_requests_.push(std::move(request));
            }
        }

        return batch;
    }

    void ProcessSingleBatch(std::vector<std::unique_ptr<BatchRequest>>& batch) {
        auto start_time = std::chrono::high_resolution_clock::now();

        // Get next memory pool
        int pool_idx = current_pool_.fetch_add(1) % num_memory_pools_;
        auto& pool = memory_pools_[pool_idx];

        // Determine stream for this batch
        int stream_idx = pool_idx % num_streams_;
        cudaStream_t stream = streams_[stream_idx];

        // Find max sequence length in batch
        size_t max_samples = 0;
        for (const auto& request : batch) {
            max_samples = std::max(max_samples, request->num_samples);
        }

        // Prepare batch data with padding
        PrepareBatchData(batch, pool, max_samples);

        // Execute batch inference
        ExecuteBatchInference(pool, batch.size(), max_samples, stream);

        // Decode results
        DecodeBatchResults(batch, pool, stream);

        // Wait for completion
        cudaStreamSynchronize(stream);

        // Update metrics
        auto end_time = std::chrono::high_resolution_clock::now();
        double latency_ms = std::chrono::duration<double, std::milli>(
            end_time - start_time).count();

        metrics_.avg_latency_ms = (metrics_.avg_latency_ms * 0.9) + (latency_ms * 0.1);

        double total_samples = 0;
        for (const auto& request : batch) {
            total_samples += request->num_samples;
        }
        metrics_.throughput_samples_per_sec = (total_samples / latency_ms) * 1000;

        // Calculate padding waste
        size_t actual_samples = 0;
        for (const auto& request : batch) {
            actual_samples += request->num_samples;
        }
        size_t padded_samples = batch.size() * max_samples;
        metrics_.padding_waste_samples += (padded_samples - actual_samples);
    }

    void PrepareBatchData(const std::vector<std::unique_ptr<BatchRequest>>& batch,
                         MemoryPool& pool, size_t max_samples) {
        // Copy and pad audio data
        for (size_t i = 0; i < batch.size(); i++) {
            const auto& request = batch[i];
            float* dest = pool.audio_buffer + i * max_samples;

            // Copy actual audio
            memcpy(dest, request->audio_data, request->num_samples * sizeof(float));

            // Zero-pad remaining samples
            if (request->num_samples < max_samples) {
                memset(dest + request->num_samples, 0,
                      (max_samples - request->num_samples) * sizeof(float));
            }
        }
    }

    void ExecuteBatchInference(MemoryPool& pool, int batch_size,
                              size_t num_samples, cudaStream_t stream) {
        // Preprocess audio batch
        const float* audio_ptrs[32];  // Max batch size
        for (int i = 0; i < batch_size; i++) {
            audio_ptrs[i] = pool.audio_buffer + i * num_samples;
        }

        cuda::batch::batch_preprocess_audio(
            audio_ptrs, pool.preprocessed_buffer,
            batch_size, num_samples, stream
        );

        // Encode with transformer
        int seq_len = num_samples / 160;  // Assuming 160 samples per frame
        cuda::batch::batch_encode_transformer(
            pool.preprocessed_buffer, pool.encoder_buffer,
            batch_size, seq_len, 1024, stream
        );

        // Decode tokens
        cuda::batch::batch_decode_transformer(
            pool.encoder_buffer, pool.token_buffer,
            batch_size, 448, stream  // Max 448 tokens
        );
    }

    void DecodeBatchResults(std::vector<std::unique_ptr<BatchRequest>>& batch,
                           MemoryPool& pool, cudaStream_t stream) {
        // Wait for GPU processing
        cudaStreamSynchronize(stream);

        // Decode tokens to text for each request
        for (size_t i = 0; i < batch.size(); i++) {
            auto& request = batch[i];
            TranscriptionResult result;

            // Extract tokens for this batch item
            int* tokens = pool.token_buffer + i * 448;

            // Convert tokens to text (using model's tokenizer)
            result.text = model_->DecodeTokens(tokens, 448);

            // Calculate timing
            auto end_time = std::chrono::steady_clock::now();
            result.processing_time_ms = std::chrono::duration<double, std::milli>(
                end_time - request->submission_time).count();

            result.language = request->options.language;
            result.confidence = 0.95f;  // Would be computed from model

            // Set the promise
            request->promise.set_value(result);
        }
    }

    void AdjustBatchSizeDynamically() {
        // Adjust based on queue depth and latency targets
        size_t queue_depth = pending_requests_.size();

        if (queue_depth > config_.max_batch_size * 2) {
            // Increase batch size if queue is growing
            config_.current_batch_size = std::min(
                config_.current_batch_size + 2,
                config_.max_batch_size
            );
        } else if (queue_depth < config_.min_batch_size &&
                  metrics_.avg_latency_ms < config_.batch_timeout_ms / 2) {
            // Decrease batch size for lower latency when queue is small
            config_.current_batch_size = std::max(
                config_.current_batch_size - 1,
                config_.min_batch_size
            );
        }
    }

    void OptimizeBatchSize() {
        // Use GPU memory and compute capability to determine optimal batch size
        size_t free_memory, total_memory;
        cudaMemGetInfo(&free_memory, &total_memory);

        // Estimate memory per batch item (rough estimate)
        size_t memory_per_item = 512 * 1024 * 1024;  // 512MB per item

        int memory_limited_batch = free_memory / memory_per_item;
        config_.optimal_batch_size = std::min(memory_limited_batch, config_.max_batch_size);
        config_.current_batch_size = config_.optimal_batch_size;
    }

    std::string GenerateRequestId() {
        static std::atomic<uint64_t> counter{0};
        return "req_" + std::to_string(counter.fetch_add(1));
    }

    void CleanupCUDAResources() {
        for (int i = 0; i < num_streams_; i++) {
            cudaStreamDestroy(streams_[i]);
        }
        delete[] streams_;

        cudnnDestroy(cudnn_handle_);
    }

    void CleanupMemoryPools() {
        for (int i = 0; i < num_memory_pools_; i++) {
            auto& pool = memory_pools_[i];
            cudaFree(pool.audio_buffer);
            cudaFree(pool.preprocessed_buffer);
            cudaFree(pool.encoder_buffer);
            cudaFree(pool.decoder_buffer);
            cudaFree(pool.token_buffer);
        }
        delete[] memory_pools_;
    }
};

// Adaptive batch scheduler for multi-GPU systems
class AdaptiveBatchScheduler {
private:
    struct GPUWorker {
        int gpu_id;
        std::unique_ptr<DynamicBatchProcessor> processor;
        std::atomic<int> pending_requests{0};
        std::atomic<double> avg_latency_ms{0};
        std::atomic<double> throughput{0};
    };

    std::vector<std::unique_ptr<GPUWorker>> workers_;
    std::atomic<int> next_worker_{0};

public:
    AdaptiveBatchScheduler(WhisperTurbo* model, int num_gpus = 1) {
        for (int i = 0; i < num_gpus; i++) {
            auto worker = std::make_unique<GPUWorker>();
            worker->gpu_id = i;
            worker->processor = std::make_unique<DynamicBatchProcessor>(model, i);
            workers_.push_back(std::move(worker));
        }
    }

    std::future<TranscriptionResult> SubmitRequest(
        const float* audio_data,
        size_t num_samples,
        uint32_t sample_rate,
        const TranscriptionOptions& options) {

        // Select worker with least pending requests (load balancing)
        int min_pending = INT_MAX;
        int selected_worker = 0;

        for (size_t i = 0; i < workers_.size(); i++) {
            int pending = workers_[i]->pending_requests.load();
            if (pending < min_pending) {
                min_pending = pending;
                selected_worker = i;
            }
        }

        workers_[selected_worker]->pending_requests++;

        auto future = workers_[selected_worker]->processor->SubmitRequest(
            audio_data, num_samples, sample_rate, options
        );

        // Update metrics asynchronously
        std::thread([this, selected_worker]() {
            auto metrics = workers_[selected_worker]->processor->GetMetrics();
            workers_[selected_worker]->avg_latency_ms = metrics.avg_latency_ms;
            workers_[selected_worker]->throughput = metrics.throughput_samples_per_sec;
            workers_[selected_worker]->pending_requests--;
        }).detach();

        return future;
    }

    void OptimizeScheduling() {
        // Rebalance batch sizes based on GPU performance
        for (auto& worker : workers_) {
            auto metrics = worker->processor->GetMetrics();

            // Adjust batch configuration based on performance
            if (metrics.avg_latency_ms > 100.0) {
                // Reduce batch size if latency is too high
                worker->processor->UpdateBatchConfig(16, 30.0f);
            } else if (metrics.gpu_utilization < 0.7) {
                // Increase batch size if GPU underutilized
                worker->processor->UpdateBatchConfig(48, 70.0f);
            }
        }
    }
};

// Global batch processor instance
static std::unique_ptr<AdaptiveBatchScheduler> g_batch_scheduler;

void InitializeBatchProcessor(WhisperTurbo* model, int num_gpus) {
    g_batch_scheduler = std::make_unique<AdaptiveBatchScheduler>(model, num_gpus);
}

std::future<TranscriptionResult> SubmitBatchRequest(
    const float* audio_data,
    size_t num_samples,
    uint32_t sample_rate,
    const TranscriptionOptions& options) {

    if (!g_batch_scheduler) {
        throw std::runtime_error("Batch processor not initialized");
    }

    return g_batch_scheduler->SubmitRequest(audio_data, num_samples, sample_rate, options);
}

void OptimizeBatchScheduling() {
    if (g_batch_scheduler) {
        g_batch_scheduler->OptimizeScheduling();
    }
}

} // namespace whisper_turbo