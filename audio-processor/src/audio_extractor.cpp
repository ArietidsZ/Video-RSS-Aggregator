#include "audio_extractor.h"
#include "gpu_accelerator.h"
#include "performance_monitor.h"

#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <iostream>
#include <cstring>

namespace audio_processor {

class AudioExtractor::Impl {
public:
    Impl(const Config& config) : config_(config) {
        av_register_all();
        avformat_network_init();

        if (config.use_gpu) {
            gpu_accelerator_ = std::make_unique<GPUAccelerator>();
        }

        perf_monitor_ = std::make_unique<PerformanceMonitor>();
    }

    ~Impl() {
        cleanup();
        avformat_network_deinit();
    }

    bool open_input(const std::string& url) {
        // Set options for network streams
        AVDictionary* options = nullptr;
        av_dict_set(&options, "rtsp_transport", "tcp", 0);
        av_dict_set(&options, "stimeout", std::to_string(config_.network_timeout_ms * 1000).c_str(), 0);

        // Open input
        format_ctx_ = avformat_alloc_context();
        format_ctx_->interrupt_callback.callback = interrupt_callback;
        format_ctx_->interrupt_callback.opaque = this;

        int ret = avformat_open_input(&format_ctx_, url.c_str(), nullptr, &options);
        av_dict_free(&options);

        if (ret < 0) {
            char errbuf[AV_ERROR_MAX_STRING_SIZE];
            av_strerror(ret, errbuf, sizeof(errbuf));
            last_error_ = std::string("Failed to open input: ") + errbuf;
            return false;
        }

        // Find stream info
        ret = avformat_find_stream_info(format_ctx_, nullptr);
        if (ret < 0) {
            last_error_ = "Failed to find stream info";
            return false;
        }

        // Find audio stream
        audio_stream_index_ = -1;
        for (unsigned int i = 0; i < format_ctx_->nb_streams; i++) {
            if (format_ctx_->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
                audio_stream_index_ = i;
                break;
            }
        }

        if (audio_stream_index_ == -1) {
            last_error_ = "No audio stream found";
            return false;
        }

        return true;
    }

    bool setup_decoder() {
        AVStream* stream = format_ctx_->streams[audio_stream_index_];
        AVCodecParameters* codecpar = stream->codecpar;

        // Find decoder
        const AVCodec* codec = avcodec_find_decoder(codecpar->codec_id);
        if (!codec) {
            last_error_ = "Codec not found";
            return false;
        }

        // Allocate codec context
        codec_ctx_ = avcodec_alloc_context3(codec);
        if (!codec_ctx_) {
            last_error_ = "Failed to allocate codec context";
            return false;
        }

        // Copy codec parameters
        int ret = avcodec_parameters_to_context(codec_ctx_, codecpar);
        if (ret < 0) {
            last_error_ = "Failed to copy codec parameters";
            return false;
        }

        // Set threading
        codec_ctx_->thread_count = config_.thread_count;
        codec_ctx_->thread_type = FF_THREAD_FRAME | FF_THREAD_SLICE;

        // Try hardware acceleration if available
        if (config_.use_gpu) {
            setup_hw_decoder(codec);
        }

        // Open codec
        ret = avcodec_open2(codec_ctx_, codec, nullptr);
        if (ret < 0) {
            char errbuf[AV_ERROR_MAX_STRING_SIZE];
            av_strerror(ret, errbuf, sizeof(errbuf));
            last_error_ = std::string("Failed to open codec: ") + errbuf;
            return false;
        }

        return true;
    }

    bool setup_resampler() {
        // Setup software resampler
        swr_ctx_ = swr_alloc();
        if (!swr_ctx_) {
            last_error_ = "Failed to allocate resampler";
            return false;
        }

        // Set options
        av_opt_set_int(swr_ctx_, "in_channel_layout", codec_ctx_->channel_layout, 0);
        av_opt_set_int(swr_ctx_, "in_sample_rate", codec_ctx_->sample_rate, 0);
        av_opt_set_sample_fmt(swr_ctx_, "in_sample_fmt", codec_ctx_->sample_fmt, 0);

        av_opt_set_int(swr_ctx_, "out_channel_layout", config_.channels == 1 ? AV_CH_LAYOUT_MONO : AV_CH_LAYOUT_STEREO, 0);
        av_opt_set_int(swr_ctx_, "out_sample_rate", config_.sample_rate, 0);
        av_opt_set_sample_fmt(swr_ctx_, "out_sample_fmt", config_.format, 0);

        // Initialize resampler
        int ret = swr_init(swr_ctx_);
        if (ret < 0) {
            last_error_ = "Failed to initialize resampler";
            return false;
        }

        return true;
    }

    void process_audio_frame(AVFrame* frame, FrameCallback& callback) {
        // Calculate output samples
        int out_samples = av_rescale_rnd(
            swr_get_delay(swr_ctx_, codec_ctx_->sample_rate) + frame->nb_samples,
            config_.sample_rate,
            codec_ctx_->sample_rate,
            AV_ROUND_UP
        );

        // Allocate output buffer
        std::vector<float> output_buffer(out_samples * config_.channels);
        uint8_t* out_planes[8] = {reinterpret_cast<uint8_t*>(output_buffer.data())};

        // Resample
        int converted_samples = swr_convert(
            swr_ctx_,
            out_planes,
            out_samples,
            (const uint8_t**)frame->data,
            frame->nb_samples
        );

        if (converted_samples > 0) {
            // Create audio frame
            AudioFrame audio_frame;
            audio_frame.data.resize(converted_samples * config_.channels);
            std::memcpy(audio_frame.data.data(), output_buffer.data(),
                       audio_frame.data.size() * sizeof(float));

            audio_frame.sample_rate = config_.sample_rate;
            audio_frame.channels = config_.channels;
            audio_frame.sample_count = converted_samples;
            audio_frame.timestamp_ms = frame->pts * av_q2d(format_ctx_->streams[audio_stream_index_->time_base) * 1000;
            audio_frame.duration_ms = (float)converted_samples / config_.sample_rate * 1000;

            // Calculate energy
            audio_frame.energy = calculate_energy(audio_frame.data);

            // Apply GPU processing if enabled
            if (config_.use_gpu && gpu_accelerator_) {
                gpu_accelerator_->process_audio(audio_frame.data.data(), audio_frame.data.size());
            }

            // Update stats
            stats_.frames_processed++;
            stats_.bytes_processed += audio_frame.data.size() * sizeof(float);

            // Invoke callback
            callback(audio_frame);
        }
    }

    bool decode_stream(FrameCallback callback) {
        AVPacket* packet = av_packet_alloc();
        AVFrame* frame = av_frame_alloc();

        if (!packet || !frame) {
            av_packet_free(&packet);
            av_frame_free(&frame);
            return false;
        }

        auto start_time = std::chrono::steady_clock::now();

        while (running_ && av_read_frame(format_ctx_, packet) >= 0) {
            if (packet->stream_index == audio_stream_index_) {
                // Send packet to decoder
                int ret = avcodec_send_packet(codec_ctx_, packet);
                if (ret < 0) {
                    av_packet_unref(packet);
                    continue;
                }

                // Receive decoded frames
                while (ret >= 0) {
                    ret = avcodec_receive_frame(codec_ctx_, frame);
                    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                        break;
                    } else if (ret < 0) {
                        last_error_ = "Error during decoding";
                        av_packet_free(&packet);
                        av_frame_free(&frame);
                        return false;
                    }

                    // Process the frame
                    process_audio_frame(frame, callback);
                    av_frame_unref(frame);
                }
            }

            av_packet_unref(packet);
        }

        // Calculate stats
        auto end_time = std::chrono::steady_clock::now();
        stats_.total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        if (stats_.total_time.count() > 0) {
            stats_.extraction_fps = (float)stats_.frames_processed / (stats_.total_time.count() / 1000.0f);
            stats_.network_bandwidth_mbps = (float)stats_.bytes_processed * 8 / (stats_.total_time.count() * 1000.0f);
        }

        av_packet_free(&packet);
        av_frame_free(&frame);

        return true;
    }

    void cleanup() {
        if (swr_ctx_) {
            swr_free(&swr_ctx_);
            swr_ctx_ = nullptr;
        }

        if (codec_ctx_) {
            avcodec_free_context(&codec_ctx_);
            codec_ctx_ = nullptr;
        }

        if (format_ctx_) {
            avformat_close_input(&format_ctx_);
            format_ctx_ = nullptr;
        }
    }

    void setup_hw_decoder(const AVCodec* codec) {
        // Find hardware acceleration config
        for (int i = 0;; i++) {
            const AVCodecHWConfig* config = avcodec_get_hw_config(codec, i);
            if (!config) break;

            if (config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX) {
                hw_device_type_ = config->device_type;

                // Create hardware device context
                AVBufferRef* hw_device_ctx = nullptr;
                int ret = av_hwdevice_ctx_create(&hw_device_ctx, hw_device_type_, nullptr, nullptr, 0);

                if (ret >= 0) {
                    codec_ctx_->hw_device_ctx = av_buffer_ref(hw_device_ctx);
                    av_buffer_unref(&hw_device_ctx);
                    std::cout << "Hardware acceleration enabled: " << av_hwdevice_get_type_name(hw_device_type_) << std::endl;
                    break;
                }
            }
        }
    }

    float calculate_energy(const std::vector<float>& samples) {
        float energy = 0.0f;
        for (float sample : samples) {
            energy += sample * sample;
        }
        return std::sqrt(energy / samples.size());
    }

    static int interrupt_callback(void* ctx) {
        Impl* impl = static_cast<Impl*>(ctx);
        return !impl->running_ ? 1 : 0;
    }

public:
    // FFmpeg contexts
    AVFormatContext* format_ctx_ = nullptr;
    AVCodecContext* codec_ctx_ = nullptr;
    SwrContext* swr_ctx_ = nullptr;

    // Stream info
    int audio_stream_index_ = -1;
    enum AVHWDeviceType hw_device_type_ = AV_HWDEVICE_TYPE_NONE;

    // Configuration
    Config config_;

    // GPU acceleration
    std::unique_ptr<GPUAccelerator> gpu_accelerator_;

    // Performance monitoring
    std::unique_ptr<PerformanceMonitor> perf_monitor_;

    // State
    std::atomic<bool> running_{false};
    std::string last_error_;
    Stats stats_;
};

// AudioExtractor implementation

AudioExtractor::AudioExtractor(const Config& config)
    : config_(config), impl_(std::make_unique<Impl>(config)) {}

AudioExtractor::~AudioExtractor() {
    stop();
}

bool AudioExtractor::extract(const std::string& url, FrameCallback callback) {
    if (running_.exchange(true)) {
        return false; // Already running
    }

    impl_->running_ = true;

    // Open input
    if (!impl_->open_input(url)) {
        running_ = false;
        impl_->running_ = false;
        return false;
    }

    // Setup decoder
    if (!impl_->setup_decoder()) {
        impl_->cleanup();
        running_ = false;
        impl_->running_ = false;
        return false;
    }

    // Setup resampler
    if (!impl_->setup_resampler()) {
        impl_->cleanup();
        running_ = false;
        impl_->running_ = false;
        return false;
    }

    // Decode stream
    bool success = impl_->decode_stream(callback);

    // Cleanup
    impl_->cleanup();
    running_ = false;
    impl_->running_ = false;

    return success;
}

void AudioExtractor::extract_async(const std::string& url, FrameCallback callback, ErrorCallback error_cb) {
    std::thread([this, url, callback, error_cb]() {
        if (!extract(url, callback)) {
            if (error_cb) {
                error_cb(impl_->last_error_);
            }
        }
    }).detach();
}

void AudioExtractor::stop() {
    if (running_.exchange(false)) {
        impl_->running_ = false;
    }
}

AudioExtractor::Stats AudioExtractor::get_stats() const {
    return impl_->stats_;
}

} // namespace audio_processor