#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include <whisper.h>
#include <llama.h>

namespace {

enum class Purpose {
    kUnknown,
    kTranscription,
    kSummarization,
};

struct BackendState {
    Purpose purpose = Purpose::kUnknown;
    std::string model_path;
    std::string language;
    int beam_size = 5;
    float temperature = 0.0f;
    int max_tokens = 256;
    whisper_context * whisper = nullptr;
    llama_model * llama_model = nullptr;
    llama_context * llama_ctx = nullptr;
    bool initialized = false;
};

BackendState g_state;
std::mutex g_mutex;

std::string trim(const std::string & value) {
    size_t start = value.find_first_not_of(" \t\n\r");
    size_t end = value.find_last_not_of(" \t\n\r");
    if (start == std::string::npos || end == std::string::npos) {
        return "";
    }
    return value.substr(start, end - start + 1);
}

std::string json_get_string(const std::string & json, const std::string & key) {
    std::string needle = "\"" + key + "\"";
    size_t pos = json.find(needle);
    if (pos == std::string::npos) {
        return "";
    }
    pos = json.find(":", pos + needle.size());
    if (pos == std::string::npos) {
        return "";
    }
    pos = json.find("\"", pos);
    if (pos == std::string::npos) {
        return "";
    }
    size_t end = json.find("\"", pos + 1);
    if (end == std::string::npos || end <= pos + 1) {
        return "";
    }
    return json.substr(pos + 1, end - pos - 1);
}

double json_get_number(const std::string & json, const std::string & key, double fallback) {
    std::string needle = "\"" + key + "\"";
    size_t pos = json.find(needle);
    if (pos == std::string::npos) {
        return fallback;
    }
    pos = json.find(":", pos + needle.size());
    if (pos == std::string::npos) {
        return fallback;
    }
    size_t end = json.find_first_of(",}", pos + 1);
    std::string token = json.substr(pos + 1, end == std::string::npos ? std::string::npos : end - pos - 1);
    token = trim(token);
    if (token.empty()) {
        return fallback;
    }
    char * end_ptr = nullptr;
    double value = std::strtod(token.c_str(), &end_ptr);
    if (end_ptr == token.c_str()) {
        return fallback;
    }
    return value;
}

std::string json_escape(const std::string & value) {
    std::string out;
    out.reserve(value.size() + 8);
    for (char c : value) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '\"': out += "\\\""; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default: out += c; break;
        }
    }
    return out;
}

void reset_state() {
    if (g_state.whisper) {
        whisper_free(g_state.whisper);
    }
    if (g_state.llama_ctx) {
        llama_free(g_state.llama_ctx);
    }
    if (g_state.llama_model) {
        llama_free_model(g_state.llama_model);
    }
    g_state = BackendState{};
}

bool read_wav_mono_16k(const char * path, std::vector<float> & samples, std::string & error) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        error = "Failed to open WAV file";
        return false;
    }

    auto read_u32 = [&file]() {
        uint32_t value = 0;
        file.read(reinterpret_cast<char *>(&value), sizeof(uint32_t));
        return value;
    };

    auto read_u16 = [&file]() {
        uint16_t value = 0;
        file.read(reinterpret_cast<char *>(&value), sizeof(uint16_t));
        return value;
    };

    char riff[4] = {0};
    file.read(riff, 4);
    if (std::strncmp(riff, "RIFF", 4) != 0) {
        error = "Invalid RIFF header";
        return false;
    }
    read_u32();
    char wave[4] = {0};
    file.read(wave, 4);
    if (std::strncmp(wave, "WAVE", 4) != 0) {
        error = "Invalid WAVE header";
        return false;
    }

    bool found_fmt = false;
    bool found_data = false;
    uint16_t audio_format = 0;
    uint16_t channels = 0;
    uint32_t sample_rate = 0;
    uint16_t bits_per_sample = 0;
    std::vector<int16_t> pcm;

    while (file && (!found_fmt || !found_data)) {
        char chunk_id[4] = {0};
        file.read(chunk_id, 4);
        if (file.gcount() != 4) {
            break;
        }
        uint32_t chunk_size = read_u32();

        if (std::strncmp(chunk_id, "fmt ", 4) == 0) {
            audio_format = read_u16();
            channels = read_u16();
            sample_rate = read_u32();
            read_u32();
            read_u16();
            bits_per_sample = read_u16();
            if (chunk_size > 16) {
                file.seekg(chunk_size - 16, std::ios::cur);
            }
            found_fmt = true;
        } else if (std::strncmp(chunk_id, "data", 4) == 0) {
            size_t sample_count = chunk_size / sizeof(int16_t);
            pcm.resize(sample_count);
            file.read(reinterpret_cast<char *>(pcm.data()), chunk_size);
            found_data = true;
        } else {
            file.seekg(chunk_size, std::ios::cur);
        }
    }

    if (!found_fmt || !found_data) {
        error = "Missing fmt or data chunk";
        return false;
    }

    if (audio_format != 1 || channels != 1 || sample_rate != 16000 || bits_per_sample != 16) {
        error = "WAV must be 16kHz mono PCM 16-bit";
        return false;
    }

    samples.resize(pcm.size());
    for (size_t i = 0; i < pcm.size(); ++i) {
        samples[i] = static_cast<float>(pcm[i]) / 32768.0f;
    }

    return true;
}

std::string summarize_with_llama(const std::string & input) {
    std::string prompt = "Summarize the following transcript in a concise paragraph.\n\n";
    prompt += input;
    prompt += "\n\nSummary:";

    const int max_tokens = g_state.max_tokens;
    const llama_model * model = g_state.llama_model;
    llama_context * ctx = g_state.llama_ctx;

    llama_kv_cache_clear(ctx);

    int max_prompt_tokens = static_cast<int>(prompt.size()) + 16;
    std::vector<llama_token> prompt_tokens(max_prompt_tokens);
    int token_count = llama_tokenize(model, prompt.c_str(), static_cast<int>(prompt.size()),
        prompt_tokens.data(), static_cast<int>(prompt_tokens.size()), true, true);
    if (token_count < 0) {
        prompt_tokens.resize(-token_count);
        token_count = llama_tokenize(model, prompt.c_str(), static_cast<int>(prompt.size()),
            prompt_tokens.data(), static_cast<int>(prompt_tokens.size()), true, true);
    }

    if (token_count < 0) {
        return "";
    }

    prompt_tokens.resize(token_count);

    llama_batch batch = llama_batch_init(static_cast<int>(prompt_tokens.size()), 0, 1);
    for (int i = 0; i < static_cast<int>(prompt_tokens.size()); ++i) {
        batch.token[i] = prompt_tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = false;
    }
    batch.logits[batch.n_tokens - 1] = true;

    if (llama_decode(ctx, batch) != 0) {
        llama_batch_free(batch);
        return "";
    }
    llama_batch_free(batch);

    std::string output;
    int n_ctx = static_cast<int>(prompt_tokens.size());
    int n_vocab = llama_n_vocab(model);

    float temperature = g_state.temperature <= 0.0f ? 0.0f : g_state.temperature;

    for (int i = 0; i < max_tokens; ++i) {
        const float * logits = llama_get_logits(ctx);
        std::vector<llama_token_data> candidates(static_cast<size_t>(n_vocab));
        for (int token_id = 0; token_id < n_vocab; ++token_id) {
            float logit = logits[token_id];
            if (temperature > 0.0f) {
                logit /= temperature;
            }
            candidates[static_cast<size_t>(token_id)] = {token_id, logit, 0.0f};
        }
        llama_token_data_array candidate_array = {candidates.data(), candidates.size(), false};
        llama_sample_softmax(ctx, &candidate_array);
        llama_token next_token = llama_sample_token_greedy(ctx, &candidate_array);
        if (next_token == llama_token_eos(model)) {
            break;
        }

        char piece[4096];
        int piece_len = llama_token_to_piece(model, next_token, piece, sizeof(piece), true);
        if (piece_len > 0) {
            output.append(piece, static_cast<size_t>(piece_len));
        }

        llama_batch token_batch = llama_batch_init(1, 0, 1);
        token_batch.token[0] = next_token;
        token_batch.pos[0] = n_ctx;
        token_batch.n_seq_id[0] = 1;
        token_batch.seq_id[0][0] = 0;
        token_batch.logits[0] = true;

        if (llama_decode(ctx, token_batch) != 0) {
            llama_batch_free(token_batch);
            break;
        }

        llama_batch_free(token_batch);
        n_ctx += 1;
    }

    return output;
}

}  // namespace

extern "C" int vra_backend_init(const char * config_json) {
    if (config_json == nullptr) {
        return -1;
    }

    std::lock_guard<std::mutex> guard(g_mutex);
    reset_state();

    std::string config(config_json);
    std::string purpose = json_get_string(config, "purpose");
    std::string model_path = json_get_string(config, "model_path");
    std::string language = json_get_string(config, "language");

    if (purpose == "transcription") {
        g_state.purpose = Purpose::kTranscription;
    } else if (purpose == "summarization") {
        g_state.purpose = Purpose::kSummarization;
    } else {
        g_state.purpose = Purpose::kUnknown;
    }

    if (model_path.empty()) {
        return -2;
    }

    g_state.model_path = model_path;
    g_state.language = language;
    g_state.beam_size = static_cast<int>(json_get_number(config, "beam_size", 5.0));
    g_state.temperature = static_cast<float>(json_get_number(config, "temperature", 0.0));
    g_state.max_tokens = static_cast<int>(json_get_number(config, "max_tokens", 256.0));
    g_state.max_tokens = std::max(1, g_state.max_tokens);

    if (g_state.purpose == Purpose::kTranscription) {
        g_state.whisper = whisper_init_from_file(g_state.model_path.c_str());
        if (!g_state.whisper) {
            reset_state();
            return -3;
        }
    } else if (g_state.purpose == Purpose::kSummarization) {
        llama_model_params mparams = llama_model_default_params();
        llama_context_params cparams = llama_context_default_params();
        g_state.llama_model = llama_load_model_from_file(g_state.model_path.c_str(), mparams);
        if (!g_state.llama_model) {
            reset_state();
            return -4;
        }
        g_state.llama_ctx = llama_new_context_with_model(g_state.llama_model, cparams);
        if (!g_state.llama_ctx) {
            reset_state();
            return -5;
        }
    } else {
        return -6;
    }

    g_state.initialized = true;
    return 0;
}

extern "C" int vra_backend_transcribe(const char * audio_path, const char ** output_json) {
    if (audio_path == nullptr || output_json == nullptr) {
        return -1;
    }

    std::lock_guard<std::mutex> guard(g_mutex);
    if (!g_state.initialized || g_state.purpose != Purpose::kTranscription || !g_state.whisper) {
        return -2;
    }

    std::vector<float> samples;
    std::string error;
    if (!read_wav_mono_16k(audio_path, samples, error)) {
        return -3;
    }

    whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    params.language = g_state.language.empty() ? nullptr : g_state.language.c_str();
    params.temperature = g_state.temperature;
    params.beam_size = g_state.beam_size;

    int status = whisper_full(g_state.whisper, params, samples.data(), static_cast<int>(samples.size()));
    if (status != 0) {
        return -4;
    }

    std::string text;
    int n_segments = whisper_full_n_segments(g_state.whisper);
    for (int i = 0; i < n_segments; ++i) {
        const char * segment = whisper_full_get_segment_text(g_state.whisper, i);
        if (segment) {
            text += segment;
        }
    }

    std::ostringstream out;
    out << "{\"text\":\"" << json_escape(text) << "\",\"language\":\"";
    out << json_escape(g_state.language.empty() ? "auto" : g_state.language) << "\"}";

    std::string payload = out.str();
    char * buffer = static_cast<char *>(std::malloc(payload.size() + 1));
    if (!buffer) {
        return -5;
    }
    std::memcpy(buffer, payload.c_str(), payload.size());
    buffer[payload.size()] = '\0';
    *output_json = buffer;
    return 0;
}

extern "C" int vra_backend_summarize(const char * text, const char ** output_json) {
    if (text == nullptr || output_json == nullptr) {
        return -1;
    }

    std::lock_guard<std::mutex> guard(g_mutex);
    if (!g_state.initialized || g_state.purpose != Purpose::kSummarization || !g_state.llama_ctx) {
        return -2;
    }

    std::string summary = summarize_with_llama(text);
    if (summary.empty()) {
        return -3;
    }

    std::ostringstream out;
    out << "{\"summary\":\"" << json_escape(summary) << "\",\"key_points\":[]}";

    std::string payload = out.str();
    char * buffer = static_cast<char *>(std::malloc(payload.size() + 1));
    if (!buffer) {
        return -4;
    }
    std::memcpy(buffer, payload.c_str(), payload.size());
    buffer[payload.size()] = '\0';
    *output_json = buffer;
    return 0;
}

extern "C" void vra_backend_free_string(const char * ptr) {
    if (ptr) {
        std::free(const_cast<char *>(ptr));
    }
}
