#include "whisper_turbo.h"
#include <onnxruntime_cxx_api.h>
#include <torch/torch.h>
#include <torch/script.h>
#include <vector>
#include <unordered_map>
#include <memory>
#include <chrono>
#include <regex>
#include <codecvt>
#include <locale>

namespace whisper_turbo {
namespace chinese {

// Chinese-specific text normalization utilities
class ChineseTextProcessor {
private:
    std::unordered_map<std::wstring, std::wstring> traditional_to_simplified_;
    std::unordered_map<std::wstring, std::wstring> number_mappings_;
    std::vector<std::wstring> punctuation_marks_;

public:
    ChineseTextProcessor() {
        InitializeMappings();
    }

    void InitializeMappings() {
        // Common traditional to simplified mappings
        traditional_to_simplified_[L"臺"] = L"台";
        traditional_to_simplified_[L"灣"] = L"湾";
        traditional_to_simplified_[L"國"] = L"国";
        traditional_to_simplified_[L"會"] = L"会";
        traditional_to_simplified_[L"時"] = L"时";
        traditional_to_simplified_[L"來"] = L"来";
        traditional_to_simplified_[L"為"] = L"为";
        traditional_to_simplified_[L"說"] = L"说";
        traditional_to_simplified_[L"對"] = L"对";
        traditional_to_simplified_[L"這"] = L"这";

        // Chinese number mappings
        number_mappings_[L"零"] = L"0";
        number_mappings_[L"一"] = L"1";
        number_mappings_[L"二"] = L"2";
        number_mappings_[L"三"] = L"3";
        number_mappings_[L"四"] = L"4";
        number_mappings_[L"五"] = L"5";
        number_mappings_[L"六"] = L"6";
        number_mappings_[L"七"] = L"7";
        number_mappings_[L"八"] = L"8";
        number_mappings_[L"九"] = L"9";
        number_mappings_[L"十"] = L"10";
        number_mappings_[L"百"] = L"100";
        number_mappings_[L"千"] = L"1000";
        number_mappings_[L"万"] = L"10000";
        number_mappings_[L"亿"] = L"100000000";

        // Chinese punctuation
        punctuation_marks_ = {L"，", L"。", L"！", L"？", L"、", L"；", L"：",
                             L""", L""", L"'", L"'", L"（", L"）", L"【", L"】"};
    }

    std::string NormalizeText(const std::string& text) {
        // Convert to wide string for Unicode processing
        std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
        std::wstring wtext = converter.from_bytes(text);

        // Apply traditional to simplified conversion
        for (const auto& [trad, simp] : traditional_to_simplified_) {
            size_t pos = 0;
            while ((pos = wtext.find(trad, pos)) != std::wstring::npos) {
                wtext.replace(pos, trad.length(), simp);
                pos += simp.length();
            }
        }

        // Normalize punctuation
        for (const auto& punct : punctuation_marks_) {
            // Add spaces around punctuation for better tokenization
            size_t pos = 0;
            while ((pos = wtext.find(punct, pos)) != std::wstring::npos) {
                wtext.insert(pos, L" ");
                wtext.insert(pos + punct.length() + 1, L" ");
                pos += punct.length() + 2;
            }
        }

        // Convert back to UTF-8
        return converter.to_bytes(wtext);
    }

    std::vector<std::string> SegmentText(const std::string& text) {
        // Basic character-level segmentation for Chinese
        std::vector<std::string> segments;
        std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
        std::wstring wtext = converter.from_bytes(text);

        for (wchar_t ch : wtext) {
            segments.push_back(converter.to_bytes(std::wstring(1, ch)));
        }

        return segments;
    }
};

// =================================================================
// Paraformer Model Integration
// =================================================================

class ParaformerEngine {
private:
    std::unique_ptr<Ort::Session> onnx_session_;
    std::unique_ptr<Ort::Env> ort_env_;
    Ort::MemoryInfo memory_info_;
    Ort::SessionOptions session_options_;

    struct ParaformerConfig {
        int sample_rate = 16000;
        int feature_dim = 80;
        int max_length = 3000;
        int vocab_size = 4234;  // Chinese character vocabulary
        int hidden_dim = 512;
        int num_layers = 12;
        int num_heads = 8;
        float ctc_weight = 0.3f;
        float attention_weight = 0.7f;
        bool use_cmvn = true;
        bool use_specaug = false;
    } config_;

    ChineseTextProcessor text_processor_;

public:
    ParaformerEngine(const std::string& model_path) {
        InitializeONNX();
        LoadModel(model_path);
    }

    void InitializeONNX() {
        ort_env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "Paraformer");
        memory_info_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        session_options_.SetIntraOpNumThreads(4);
        session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // Enable CUDA execution provider if available
        #ifdef USE_CUDA
        OrtCUDAProviderOptions cuda_options;
        cuda_options.device_id = 0;
        cuda_options.arena_extend_strategy = 0;
        cuda_options.cuda_mem_limit = 2 * 1024 * 1024 * 1024;  // 2GB
        session_options_.AppendExecutionProvider_CUDA(cuda_options);
        #endif
    }

    void LoadModel(const std::string& model_path) {
        onnx_session_ = std::make_unique<Ort::Session>(*ort_env_, model_path.c_str(), session_options_);
    }

    std::vector<float> ExtractFbank(const float* audio_data, size_t num_samples) {
        // Extract 80-dimensional Fbank features
        const int frame_length = 25;  // 25ms
        const int frame_shift = 10;   // 10ms
        const int n_fft = 512;
        const int n_mels = config_.feature_dim;

        int frame_length_samples = frame_length * config_.sample_rate / 1000;
        int frame_shift_samples = frame_shift * config_.sample_rate / 1000;
        int num_frames = (num_samples - frame_length_samples) / frame_shift_samples + 1;

        std::vector<float> features(num_frames * n_mels);

        // Simplified Fbank extraction (would use actual DSP library)
        for (int frame = 0; frame < num_frames; frame++) {
            int start = frame * frame_shift_samples;

            // Window and FFT
            std::vector<float> windowed(frame_length_samples);
            for (int i = 0; i < frame_length_samples; i++) {
                if (start + i < num_samples) {
                    // Hamming window
                    float window = 0.54f - 0.46f * cosf(2.0f * M_PI * i / (frame_length_samples - 1));
                    windowed[i] = audio_data[start + i] * window;
                }
            }

            // Apply mel filterbank and log
            for (int mel = 0; mel < n_mels; mel++) {
                float energy = 0.0f;
                // Simplified mel filter application
                for (int i = 0; i < frame_length_samples; i++) {
                    energy += windowed[i] * windowed[i];
                }
                features[frame * n_mels + mel] = logf(energy + 1e-10f);
            }
        }

        // Apply CMVN (Cepstral Mean and Variance Normalization)
        if (config_.use_cmvn) {
            ApplyCMVN(features.data(), num_frames, n_mels);
        }

        return features;
    }

    void ApplyCMVN(float* features, int num_frames, int feature_dim) {
        // Compute mean
        std::vector<float> mean(feature_dim, 0.0f);
        for (int frame = 0; frame < num_frames; frame++) {
            for (int dim = 0; dim < feature_dim; dim++) {
                mean[dim] += features[frame * feature_dim + dim];
            }
        }
        for (int dim = 0; dim < feature_dim; dim++) {
            mean[dim] /= num_frames;
        }

        // Compute variance
        std::vector<float> variance(feature_dim, 0.0f);
        for (int frame = 0; frame < num_frames; frame++) {
            for (int dim = 0; dim < feature_dim; dim++) {
                float diff = features[frame * feature_dim + dim] - mean[dim];
                variance[dim] += diff * diff;
            }
        }
        for (int dim = 0; dim < feature_dim; dim++) {
            variance[dim] = sqrtf(variance[dim] / num_frames + 1e-10f);
        }

        // Normalize
        for (int frame = 0; frame < num_frames; frame++) {
            for (int dim = 0; dim < feature_dim; dim++) {
                features[frame * feature_dim + dim] =
                    (features[frame * feature_dim + dim] - mean[dim]) / variance[dim];
            }
        }
    }

    std::string Transcribe(const float* audio_data, size_t num_samples) {
        // Extract features
        auto features = ExtractFbank(audio_data, num_samples);
        int num_frames = features.size() / config_.feature_dim;

        // Prepare input tensors
        std::vector<int64_t> input_shape = {1, num_frames, config_.feature_dim};
        std::vector<Ort::Value> input_tensors;

        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info_, features.data(), features.size(),
            input_shape.data(), input_shape.size()
        ));

        // Length tensor
        std::vector<int32_t> lengths = {num_frames};
        std::vector<int64_t> length_shape = {1};
        input_tensors.push_back(Ort::Value::CreateTensor<int32_t>(
            memory_info_, lengths.data(), lengths.size(),
            length_shape.data(), length_shape.size()
        ));

        // Run inference
        auto output_names = onnx_session_->GetOutputNames();
        std::vector<const char*> input_names = {"speech", "speech_lengths"};
        std::vector<const char*> output_names_ptr;
        for (const auto& name : output_names) {
            output_names_ptr.push_back(name.c_str());
        }

        auto outputs = onnx_session_->Run(
            Ort::RunOptions{nullptr},
            input_names.data(), input_tensors.data(), input_tensors.size(),
            output_names_ptr.data(), output_names_ptr.size()
        );

        // Decode output
        return DecodeOutput(outputs[0]);
    }

    std::string DecodeOutput(const Ort::Value& output) {
        auto output_shape = output.GetTensorTypeAndShapeInfo().GetShape();
        const float* output_data = output.GetTensorData<float>();

        int batch_size = output_shape[0];
        int seq_length = output_shape[1];
        int vocab_size = output_shape[2];

        std::vector<int> token_ids;

        // Greedy decoding
        for (int t = 0; t < seq_length; t++) {
            int max_idx = 0;
            float max_prob = output_data[t * vocab_size];

            for (int v = 1; v < vocab_size; v++) {
                if (output_data[t * vocab_size + v] > max_prob) {
                    max_prob = output_data[t * vocab_size + v];
                    max_idx = v;
                }
            }

            // Skip blank token (usually 0)
            if (max_idx != 0) {
                token_ids.push_back(max_idx);
            }
        }

        // Convert token IDs to text
        return TokensToText(token_ids);
    }

    std::string TokensToText(const std::vector<int>& tokens) {
        // This would use the actual vocabulary mapping
        std::string text;

        // Simplified: assume direct Unicode mapping
        for (int token : tokens) {
            if (token > 0 && token < config_.vocab_size) {
                // Map token to Chinese character
                wchar_t ch = L'一' + (token - 1);  // Simplified mapping
                std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
                text += converter.to_bytes(std::wstring(1, ch));
            }
        }

        return text_processor_.NormalizeText(text);
    }
};

// =================================================================
// FunASR Integration
// =================================================================

class FunASREngine {
private:
    torch::jit::script::Module model_;
    torch::Device device_;

    struct FunASRConfig {
        int sample_rate = 16000;
        int feature_dim = 560;  // FunASR uses larger features
        int max_length = 6000;
        int vocab_size = 4500;
        int hidden_dim = 256;
        int num_encoder_layers = 12;
        int num_decoder_layers = 6;
        bool use_vad = true;
        bool use_punc = true;  // Punctuation restoration
        bool use_timestamp = true;
    } config_;

    ChineseTextProcessor text_processor_;

public:
    FunASREngine(const std::string& model_path, bool use_gpu = true)
        : device_(use_gpu && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU) {

        LoadModel(model_path);
    }

    void LoadModel(const std::string& model_path) {
        try {
            model_ = torch::jit::load(model_path);
            model_.to(device_);
            model_.eval();
        } catch (const c10::Error& e) {
            throw std::runtime_error("Failed to load FunASR model: " + std::string(e.what()));
        }
    }

    struct VADSegment {
        int start_frame;
        int end_frame;
        float confidence;
    };

    std::vector<VADSegment> DetectVoiceActivity(const float* audio_data, size_t num_samples) {
        std::vector<VADSegment> segments;

        // Simple energy-based VAD (would use actual VAD model)
        const int frame_length = 25 * config_.sample_rate / 1000;
        const int frame_shift = 10 * config_.sample_rate / 1000;
        const int num_frames = (num_samples - frame_length) / frame_shift + 1;

        std::vector<float> frame_energy(num_frames);

        for (int i = 0; i < num_frames; i++) {
            float energy = 0.0f;
            for (int j = 0; j < frame_length; j++) {
                int idx = i * frame_shift + j;
                if (idx < num_samples) {
                    energy += audio_data[idx] * audio_data[idx];
                }
            }
            frame_energy[i] = 10.0f * log10f(energy / frame_length + 1e-10f);
        }

        // Dynamic threshold based on percentile
        std::vector<float> sorted_energy = frame_energy;
        std::sort(sorted_energy.begin(), sorted_energy.end());
        float threshold = sorted_energy[sorted_energy.size() / 10];  // 10th percentile

        // Find speech segments
        bool in_speech = false;
        int start_frame = 0;

        for (int i = 0; i < num_frames; i++) {
            if (!in_speech && frame_energy[i] > threshold) {
                in_speech = true;
                start_frame = i;
            } else if (in_speech && frame_energy[i] <= threshold) {
                segments.push_back({start_frame, i, 0.9f});
                in_speech = false;
            }
        }

        if (in_speech) {
            segments.push_back({start_frame, num_frames - 1, 0.9f});
        }

        return segments;
    }

    std::string TranscribeWithVAD(const float* audio_data, size_t num_samples) {
        // Detect voice segments
        auto vad_segments = DetectVoiceActivity(audio_data, num_samples);

        std::string full_transcript;

        for (const auto& segment : vad_segments) {
            int start_sample = segment.start_frame * 160;  // 10ms frame shift
            int end_sample = std::min(segment.end_frame * 160, (int)num_samples);
            int segment_length = end_sample - start_sample;

            if (segment_length > 0) {
                std::vector<float> segment_audio(audio_data + start_sample,
                                                audio_data + end_sample);

                std::string segment_text = TranscribeSegment(segment_audio.data(),
                                                            segment_audio.size());

                if (!full_transcript.empty()) {
                    full_transcript += " ";
                }
                full_transcript += segment_text;
            }
        }

        // Apply punctuation restoration
        if (config_.use_punc) {
            full_transcript = RestorePunctuation(full_transcript);
        }

        return full_transcript;
    }

    std::string TranscribeSegment(const float* audio_data, size_t num_samples) {
        // Convert audio to tensor
        torch::Tensor audio_tensor = torch::from_blob(
            const_cast<float*>(audio_data),
            {1, static_cast<long>(num_samples)},
            torch::kFloat32
        ).to(device_);

        // Run inference
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(audio_tensor);

        torch::NoGradGuard no_grad;
        auto output = model_.forward(inputs).toTensor();

        // Decode output
        return DecodeOutput(output);
    }

    std::string DecodeOutput(const torch::Tensor& output) {
        // Get argmax predictions
        auto predictions = output.argmax(-1);
        auto pred_cpu = predictions.to(torch::kCPU);

        std::vector<int> token_ids;
        auto accessor = pred_cpu.accessor<long, 2>();

        for (int i = 0; i < accessor.size(1); i++) {
            int token = accessor[0][i];
            if (token != 0) {  // Skip blank token
                token_ids.push_back(token);
            }
        }

        return TokensToChineseText(token_ids);
    }

    std::string TokensToChineseText(const std::vector<int>& tokens) {
        // Map tokens to Chinese characters using vocabulary
        std::string text;

        for (int token : tokens) {
            // This would use actual vocabulary mapping
            if (token > 0 && token < config_.vocab_size) {
                text += GetChineseChar(token);
            }
        }

        return text;
    }

    std::string GetChineseChar(int token_id) {
        // Simplified mapping - would use actual vocabulary file
        static std::vector<std::string> vocab = {
            "的", "一", "是", "了", "我", "不", "人", "在", "他", "有",
            "这", "个", "上", "们", "来", "到", "时", "大", "地", "为",
            "子", "中", "你", "说", "生", "国", "年", "着", "就", "那",
            "和", "要", "她", "出", "也", "得", "里", "后", "自", "会"
            // ... more characters
        };

        if (token_id > 0 && token_id <= vocab.size()) {
            return vocab[token_id - 1];
        }
        return "";
    }

    std::string RestorePunctuation(const std::string& text) {
        // Simple rule-based punctuation restoration
        // In production, would use a trained punctuation model

        std::string result = text;

        // Add period at the end
        if (!result.empty() && result.back() != '。') {
            result += "。";
        }

        // Add commas after common phrase boundaries
        std::vector<std::string> comma_triggers = {"然后", "但是", "所以", "因为", "如果"};

        for (const auto& trigger : comma_triggers) {
            size_t pos = 0;
            while ((pos = result.find(trigger, pos)) != std::string::npos) {
                pos += trigger.length();
                if (pos < result.length() && result[pos] != '，') {
                    result.insert(pos, "，");
                }
            }
        }

        return result;
    }
};

// =================================================================
// Qwen3-ASR Integration
// =================================================================

class Qwen3ASREngine {
private:
    std::unique_ptr<torch::jit::script::Module> encoder_;
    std::unique_ptr<torch::jit::script::Module> decoder_;
    torch::Device device_;

    struct Qwen3Config {
        int sample_rate = 16000;
        int feature_dim = 80;
        int max_audio_length = 30 * 16000;  // 30 seconds
        int vocab_size = 50000;  // Large vocabulary for technical terms
        int hidden_dim = 1024;
        int num_layers = 24;
        int num_heads = 16;
        int intermediate_size = 4096;
        bool use_rotary_embedding = true;
        bool use_flash_attention = true;
        float temperature = 0.0f;  // Greedy decoding for accuracy
    } config_;

    std::unordered_map<std::string, std::vector<float>> domain_embeddings_;
    ChineseTextProcessor text_processor_;

public:
    Qwen3ASREngine(const std::string& encoder_path,
                   const std::string& decoder_path,
                   bool use_gpu = true)
        : device_(use_gpu && torch::cuda::is_available() ? torch::kCUDA : torch::kCPU) {

        LoadModels(encoder_path, decoder_path);
        LoadDomainEmbeddings();
    }

    void LoadModels(const std::string& encoder_path, const std::string& decoder_path) {
        encoder_ = std::make_unique<torch::jit::script::Module>(torch::jit::load(encoder_path));
        decoder_ = std::make_unique<torch::jit::script::Module>(torch::jit::load(decoder_path));

        encoder_->to(device_);
        decoder_->to(device_);

        encoder_->eval();
        decoder_->eval();
    }

    void LoadDomainEmbeddings() {
        // Load pre-computed domain embeddings for specialized vocabulary
        // Technical, Medical, Legal, Educational domains

        domain_embeddings_["technical"] = std::vector<float>(config_.hidden_dim);
        domain_embeddings_["medical"] = std::vector<float>(config_.hidden_dim);
        domain_embeddings_["legal"] = std::vector<float>(config_.hidden_dim);
        domain_embeddings_["educational"] = std::vector<float>(config_.hidden_dim);

        // These would be loaded from files
    }

    std::string TranscribeWithDomain(const float* audio_data,
                                    size_t num_samples,
                                    const std::string& domain = "general") {
        // Extract features
        auto features = ExtractLogMelSpectrogram(audio_data, num_samples);

        // Get domain embedding if available
        torch::Tensor domain_embedding;
        if (domain_embeddings_.find(domain) != domain_embeddings_.end()) {
            domain_embedding = torch::from_blob(
                domain_embeddings_[domain].data(),
                {1, config_.hidden_dim},
                torch::kFloat32
            ).to(device_);
        }

        // Encode audio
        auto encoder_output = EncodeAudio(features, domain_embedding);

        // Decode with beam search
        auto tokens = DecodeWithBeamSearch(encoder_output, 5);  // beam_size = 5

        // Convert to text
        return TokensToText(tokens);
    }

    torch::Tensor ExtractLogMelSpectrogram(const float* audio_data, size_t num_samples) {
        // Similar to Paraformer but with different parameters
        const int n_fft = 512;
        const int hop_length = 160;
        const int n_mels = config_.feature_dim;

        int num_frames = (num_samples - n_fft) / hop_length + 1;
        torch::Tensor features = torch::zeros({1, num_frames, n_mels});

        // Simplified mel spectrogram extraction
        // In production, would use torchaudio or librosa

        return features.to(device_);
    }

    torch::Tensor EncodeAudio(const torch::Tensor& features,
                             const torch::Tensor& domain_embedding = {}) {
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(features);

        if (domain_embedding.defined()) {
            inputs.push_back(domain_embedding);
        }

        torch::NoGradGuard no_grad;
        return encoder_->forward(inputs).toTensor();
    }

    std::vector<int> DecodeWithBeamSearch(const torch::Tensor& encoder_output,
                                         int beam_size = 5) {
        std::vector<int> output_tokens;

        // Initialize beam
        struct Beam {
            std::vector<int> tokens;
            float score;
            torch::Tensor hidden_state;
        };

        std::vector<Beam> beams;
        beams.push_back({{}, 0.0f, torch::zeros({1, config_.hidden_dim}).to(device_)});

        int max_length = 200;
        int eos_token = 2;  // End of sequence token

        for (int step = 0; step < max_length; step++) {
            std::vector<Beam> candidates;

            for (const auto& beam : beams) {
                // Prepare decoder input
                torch::Tensor input_ids;
                if (beam.tokens.empty()) {
                    input_ids = torch::zeros({1, 1}, torch::kLong).to(device_);
                } else {
                    input_ids = torch::tensor(beam.tokens).unsqueeze(0).to(device_);
                }

                // Run decoder
                std::vector<torch::jit::IValue> decoder_inputs;
                decoder_inputs.push_back(encoder_output);
                decoder_inputs.push_back(input_ids);
                decoder_inputs.push_back(beam.hidden_state);

                auto decoder_output = decoder_->forward(decoder_inputs).toTuple();
                auto logits = decoder_output->elements()[0].toTensor();
                auto new_hidden = decoder_output->elements()[1].toTensor();

                // Get top-k predictions
                auto topk = logits[0][-1].topk(beam_size);
                auto scores = std::get<0>(topk);
                auto indices = std::get<1>(topk);

                // Create new candidates
                for (int k = 0; k < beam_size; k++) {
                    Beam new_beam;
                    new_beam.tokens = beam.tokens;
                    new_beam.tokens.push_back(indices[k].item<int>());
                    new_beam.score = beam.score + scores[k].item<float>();
                    new_beam.hidden_state = new_hidden;

                    candidates.push_back(new_beam);
                }
            }

            // Select top beams
            std::sort(candidates.begin(), candidates.end(),
                     [](const Beam& a, const Beam& b) { return a.score > b.score; });

            beams.clear();
            for (int i = 0; i < std::min(beam_size, (int)candidates.size()); i++) {
                if (candidates[i].tokens.back() == eos_token) {
                    return candidates[i].tokens;
                }
                beams.push_back(candidates[i]);
            }
        }

        // Return best beam
        return beams[0].tokens;
    }

    std::string TokensToText(const std::vector<int>& tokens) {
        // Use SentencePiece or similar tokenizer
        // This is a simplified version

        std::string text;
        for (int token : tokens) {
            if (token >= 4 && token < config_.vocab_size) {
                // Map token to text (would use actual vocabulary)
                text += GetTokenText(token);
            }
        }

        // Post-process
        text = text_processor_.NormalizeText(text);
        text = AddSpacingForMixedLanguage(text);

        return text;
    }

    std::string GetTokenText(int token_id) {
        // Simplified - would use actual vocabulary
        if (token_id < 1000) {
            // Common Chinese characters
            return GetCommonChineseChar(token_id);
        } else if (token_id < 2000) {
            // English words
            return GetEnglishWord(token_id - 1000);
        } else {
            // Technical terms
            return GetTechnicalTerm(token_id - 2000);
        }
    }

    std::string GetCommonChineseChar(int id) {
        static std::vector<std::string> chars = {
            "的", "一", "是", "了", "我", "不", "人", "在", "他", "有"
            // ... more characters
        };
        return (id < chars.size()) ? chars[id] : "";
    }

    std::string GetEnglishWord(int id) {
        static std::vector<std::string> words = {
            "the", "is", "are", "was", "were", "be", "have", "has", "had", "do",
            "API", "GPU", "CPU", "RAM", "SDK", "JSON", "XML", "HTTP", "HTTPS", "URL"
            // ... more words
        };
        return (id < words.size()) ? words[id] : "";
    }

    std::string GetTechnicalTerm(int id) {
        static std::vector<std::string> terms = {
            "机器学习", "深度学习", "神经网络", "卷积", "循环", "注意力机制",
            "梯度下降", "反向传播", "过拟合", "正则化", "批量归一化", "激活函数"
            // ... more technical terms
        };
        return (id < terms.size()) ? terms[id] : "";
    }

    std::string AddSpacingForMixedLanguage(const std::string& text) {
        // Add spaces between Chinese and English text
        std::string result;
        bool prev_is_chinese = false;
        bool prev_is_english = false;

        for (size_t i = 0; i < text.length(); ) {
            unsigned char c = text[i];

            bool is_chinese = (c >= 0xE4 && c <= 0xE9);  // Common Chinese UTF-8 range
            bool is_english = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');

            if ((prev_is_chinese && is_english) || (prev_is_english && is_chinese)) {
                result += " ";
            }

            // Handle multi-byte UTF-8
            if (c < 0x80) {
                result += c;
                i++;
            } else if (c < 0xE0) {
                result += text.substr(i, 2);
                i += 2;
            } else if (c < 0xF0) {
                result += text.substr(i, 3);
                i += 3;
            } else {
                result += text.substr(i, 4);
                i += 4;
            }

            prev_is_chinese = is_chinese;
            prev_is_english = is_english;
        }

        return result;
    }
};

// =================================================================
// Unified Chinese ASR Interface
// =================================================================

class ChineseASR {
private:
    std::unique_ptr<ParaformerEngine> paraformer_;
    std::unique_ptr<FunASREngine> funasr_;
    std::unique_ptr<Qwen3ASREngine> qwen3_;

    enum class ModelType {
        Paraformer,
        FunASR,
        Qwen3ASR
    };

    ModelType active_model_;

public:
    ChineseASR() : active_model_(ModelType::Paraformer) {}

    void LoadModel(const std::string& model_name, const std::string& model_path) {
        if (model_name == "paraformer") {
            paraformer_ = std::make_unique<ParaformerEngine>(model_path);
            active_model_ = ModelType::Paraformer;
        } else if (model_name == "funasr") {
            funasr_ = std::make_unique<FunASREngine>(model_path);
            active_model_ = ModelType::FunASR;
        } else if (model_name == "qwen3") {
            qwen3_ = std::make_unique<Qwen3ASREngine>(model_path + "/encoder",
                                                      model_path + "/decoder");
            active_model_ = ModelType::Qwen3ASR;
        }
    }

    std::string Transcribe(const float* audio_data, size_t num_samples,
                          const std::string& domain = "general") {
        switch (active_model_) {
            case ModelType::Paraformer:
                return paraformer_->Transcribe(audio_data, num_samples);

            case ModelType::FunASR:
                return funasr_->TranscribeWithVAD(audio_data, num_samples);

            case ModelType::Qwen3ASR:
                return qwen3_->TranscribeWithDomain(audio_data, num_samples, domain);

            default:
                return "";
        }
    }

    void SetModel(const std::string& model_name) {
        if (model_name == "paraformer" && paraformer_) {
            active_model_ = ModelType::Paraformer;
        } else if (model_name == "funasr" && funasr_) {
            active_model_ = ModelType::FunASR;
        } else if (model_name == "qwen3" && qwen3_) {
            active_model_ = ModelType::Qwen3ASR;
        }
    }

    std::string GetActiveModel() const {
        switch (active_model_) {
            case ModelType::Paraformer: return "paraformer";
            case ModelType::FunASR: return "funasr";
            case ModelType::Qwen3ASR: return "qwen3";
            default: return "unknown";
        }
    }
};

} // namespace chinese
} // namespace whisper_turbo