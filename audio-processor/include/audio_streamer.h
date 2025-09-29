#pragma once

#include <string>
#include <memory>
#include <functional>
#include <vector>
#include <atomic>
#include <chrono>
#include <queue>
#include <mutex>

namespace audio_processor {

/**
 * Real-time audio streaming with WebRTC/WebSocket
 * Achieves <500ms latency for live streaming
 */
class AudioStreamer {
public:
    enum class Protocol {
        WebSocket,      // WebSocket for browser compatibility
        WebRTC,        // WebRTC for peer-to-peer streaming
        RTMP,          // RTMP for traditional streaming
        HLS            // HLS for adaptive streaming
    };

    struct StreamConfig {
        Protocol protocol = Protocol::WebSocket;
        std::string endpoint_url;
        int port = 8080;

        // Audio settings
        int sample_rate = 16000;
        int channels = 1;
        int bitrate = 128000;      // 128 kbps

        // Streaming settings
        int chunk_duration_ms = 100;  // 100ms chunks for low latency
        int buffer_size_ms = 500;     // 500ms buffer
        bool enable_opus = true;       // Use Opus codec
        bool enable_compression = true;

        // WebRTC specific
        std::vector<std::string> ice_servers;
        bool enable_dtls = true;
        bool enable_srtp = true;

        // Performance
        int worker_threads = 4;
        bool use_gpu = true;
    };

    struct StreamStats {
        size_t bytes_sent = 0;
        size_t packets_sent = 0;
        size_t packets_dropped = 0;
        float bitrate_kbps = 0.0f;
        float latency_ms = 0.0f;
        float jitter_ms = 0.0f;
        float packet_loss_percent = 0.0f;
        std::chrono::steady_clock::time_point start_time;
    };

    using AudioDataCallback = std::function<void(const float*, size_t)>;
    using StateChangeCallback = std::function<void(bool connected)>;
    using ErrorCallback = std::function<void(const std::string&)>;

    explicit AudioStreamer(const StreamConfig& config = StreamConfig());
    ~AudioStreamer();

    // Non-copyable
    AudioStreamer(const AudioStreamer&) = delete;
    AudioStreamer& operator=(const AudioStreamer&) = delete;

    /**
     * Initialize streaming connection
     */
    bool initialize();

    /**
     * Start streaming
     */
    bool start_stream();

    /**
     * Send audio data
     * @param data Audio samples
     * @param size Number of samples
     * @return Success status
     */
    bool send_audio(const float* data, size_t size);

    /**
     * Send audio data with timestamp
     */
    bool send_audio_timestamped(const float* data, size_t size, int64_t timestamp_ms);

    /**
     * Stop streaming
     */
    void stop_stream();

    /**
     * Set callbacks
     */
    void set_state_callback(StateChangeCallback callback);
    void set_error_callback(ErrorCallback callback);

    /**
     * Get streaming statistics
     */
    StreamStats get_stats() const;

    /**
     * Get current latency in milliseconds
     */
    float get_latency_ms() const;

    /**
     * Check if streaming
     */
    bool is_streaming() const { return streaming_.load(); }

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
    StreamConfig config_;
    std::atomic<bool> streaming_{false};
    std::atomic<bool> initialized_{false};
};

/**
 * WebSocket audio server for browser clients
 */
class WebSocketAudioServer {
public:
    struct ServerConfig {
        int port = 8080;
        std::string bind_address = "0.0.0.0";
        int max_clients = 100;
        int worker_threads = 4;

        // SSL/TLS
        bool enable_ssl = false;
        std::string cert_file;
        std::string key_file;

        // Audio settings
        int sample_rate = 16000;
        int channels = 1;

        // Performance
        int send_buffer_size = 65536;
        int recv_buffer_size = 65536;
        bool enable_compression = true;
    };

    struct ClientInfo {
        std::string client_id;
        std::string ip_address;
        std::chrono::steady_clock::time_point connect_time;
        size_t bytes_sent = 0;
        size_t bytes_received = 0;
        float latency_ms = 0.0f;
    };

    using ClientConnectedCallback = std::function<void(const std::string& client_id)>;
    using ClientDisconnectedCallback = std::function<void(const std::string& client_id)>;
    using AudioReceivedCallback = std::function<void(const std::string& client_id, const float*, size_t)>;

    explicit WebSocketAudioServer(const ServerConfig& config = ServerConfig());
    ~WebSocketAudioServer();

    /**
     * Start WebSocket server
     */
    bool start();

    /**
     * Stop server
     */
    void stop();

    /**
     * Broadcast audio to all clients
     */
    void broadcast_audio(const float* data, size_t size);

    /**
     * Send audio to specific client
     */
    bool send_to_client(const std::string& client_id, const float* data, size_t size);

    /**
     * Get connected clients
     */
    std::vector<ClientInfo> get_clients() const;

    /**
     * Set callbacks
     */
    void set_client_connected_callback(ClientConnectedCallback callback);
    void set_client_disconnected_callback(ClientDisconnectedCallback callback);
    void set_audio_received_callback(AudioReceivedCallback callback);

    /**
     * Get server statistics
     */
    struct ServerStats {
        size_t total_connections = 0;
        size_t active_connections = 0;
        size_t total_bytes_sent = 0;
        size_t total_bytes_received = 0;
        float avg_latency_ms = 0.0f;
        float bandwidth_mbps = 0.0f;
    };
    ServerStats get_stats() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
    ServerConfig config_;
    std::atomic<bool> running_{false};
};

/**
 * WebRTC audio handler for peer-to-peer streaming
 */
class WebRTCAudioHandler {
public:
    struct RTCConfig {
        std::vector<std::string> ice_servers = {
            "stun:stun.l.google.com:19302"
        };

        // Audio codec preferences
        std::vector<std::string> audio_codecs = {"opus", "PCMU", "PCMA"};

        // Network settings
        int min_port = 10000;
        int max_port = 20000;
        bool enable_tcp = true;
        bool enable_udp = true;

        // Security
        bool require_encryption = true;
        bool enable_dtls_srtp = true;

        // Performance
        int packet_size = 960;  // 20ms at 48kHz
        int jitter_buffer_ms = 100;
        bool enable_fec = true;  // Forward error correction
        bool enable_dtx = true;  // Discontinuous transmission
    };

    struct PeerConnection {
        std::string peer_id;
        std::string local_sdp;
        std::string remote_sdp;
        std::vector<std::string> ice_candidates;
        bool connected = false;
        std::chrono::steady_clock::time_point connect_time;
    };

    using OnOfferCallback = std::function<void(const std::string& sdp)>;
    using OnAnswerCallback = std::function<void(const std::string& sdp)>;
    using OnICECandidateCallback = std::function<void(const std::string& candidate)>;
    using OnAudioCallback = std::function<void(const float*, size_t)>;

    explicit WebRTCAudioHandler(const RTCConfig& config = RTCConfig());
    ~WebRTCAudioHandler();

    /**
     * Create peer connection
     */
    std::string create_peer_connection();

    /**
     * Create offer
     */
    std::string create_offer(const std::string& peer_id);

    /**
     * Create answer
     */
    std::string create_answer(const std::string& peer_id, const std::string& offer_sdp);

    /**
     * Set remote description
     */
    bool set_remote_description(const std::string& peer_id, const std::string& sdp);

    /**
     * Add ICE candidate
     */
    bool add_ice_candidate(const std::string& peer_id, const std::string& candidate);

    /**
     * Send audio to peer
     */
    bool send_audio(const std::string& peer_id, const float* data, size_t size);

    /**
     * Close peer connection
     */
    void close_peer_connection(const std::string& peer_id);

    /**
     * Set callbacks
     */
    void set_on_offer_callback(OnOfferCallback callback);
    void set_on_answer_callback(OnAnswerCallback callback);
    void set_on_ice_candidate_callback(OnICECandidateCallback callback);
    void set_on_audio_callback(OnAudioCallback callback);

    /**
     * Get peer connection info
     */
    PeerConnection get_peer_info(const std::string& peer_id) const;

    /**
     * Get all active connections
     */
    std::vector<PeerConnection> get_active_connections() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
    RTCConfig config_;
};

} // namespace audio_processor