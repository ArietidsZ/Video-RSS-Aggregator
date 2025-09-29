#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include <unordered_map>
#include <queue>
#include <cmath>
#include <regex>
#include <sstream>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace summarization {

// Forward declarations for CUDA kernels
namespace cuda {
namespace chunking {
    void compute_sentence_embeddings(const float* token_embeddings,
                                    const int* sentence_boundaries,
                                    float* sentence_embeddings,
                                    int num_sentences, int embedding_dim,
                                    cudaStream_t stream);

    void compute_similarity_matrix(const float* embeddings,
                                  float* similarity_matrix,
                                  int num_sentences, int embedding_dim,
                                  cudaStream_t stream);

    void find_optimal_boundaries(const float* similarity_matrix,
                                const int* sentence_lengths,
                                int* chunk_boundaries,
                                int num_sentences, int max_chunk_size,
                                float similarity_threshold,
                                cudaStream_t stream);
}
}

class ContextAwareChunker {
public:
    struct ChunkingConfig {
        int min_chunk_tokens = 500;
        int max_chunk_tokens = 2000;
        int target_chunk_tokens = 1500;
        int overlap_tokens = 200;
        float semantic_similarity_threshold = 0.7f;
        bool preserve_sentences = true;
        bool preserve_paragraphs = true;
        bool detect_topic_shifts = true;
        bool maintain_speaker_continuity = true;
        bool respect_timestamp_boundaries = true;
        int max_context_window = 8192;
        std::string chunking_strategy = "semantic";  // semantic, sliding, hierarchical
    };

    struct Chunk {
        std::string text;
        int start_token;
        int end_token;
        int start_char;
        int end_char;
        float start_timestamp;
        float end_timestamp;
        std::vector<std::string> keywords;
        std::string summary;
        float coherence_score;
        int speaker_id;
        std::string topic;
        std::vector<int> sentence_indices;
        std::unordered_map<std::string, float> metadata;
    };

    struct ChunkingResult {
        std::vector<Chunk> chunks;
        std::vector<std::pair<int, int>> chunk_overlaps;
        float avg_coherence_score;
        int total_chunks;
        std::vector<std::string> detected_topics;
        std::unordered_map<int, std::string> speaker_map;
    };

private:
    ChunkingConfig config_;
    std::unique_ptr<class SentenceTokenizer> tokenizer_;
    std::unique_ptr<class EmbeddingModel> embedding_model_;

    // CUDA resources
    cudaStream_t stream_;
    float* d_embeddings_;
    float* d_similarity_matrix_;
    int* d_boundaries_;
    size_t embedding_buffer_size_;

    // Caching for efficiency
    struct EmbeddingCache {
        std::unordered_map<std::string, std::vector<float>> sentence_embeddings;
        std::unordered_map<std::string, float> sentence_importance;
        void clear() { sentence_embeddings.clear(); sentence_importance.clear(); }
    };
    EmbeddingCache cache_;

public:
    ContextAwareChunker(const ChunkingConfig& config = {})
        : config_(config) {
        InitializeResources();
    }

    ~ContextAwareChunker() {
        CleanupResources();
    }

    ChunkingResult ChunkTranscript(const std::string& transcript,
                                   const std::vector<float>& timestamps = {},
                                   const std::vector<int>& speaker_ids = {}) {
        // Preprocessing
        auto sentences = SplitIntoSentences(transcript);
        auto embeddings = ComputeSentenceEmbeddings(sentences);

        ChunkingResult result;

        if (config_.chunking_strategy == "semantic") {
            result = SemanticChunking(sentences, embeddings, timestamps, speaker_ids);
        } else if (config_.chunking_strategy == "sliding") {
            result = SlidingWindowChunking(sentences, timestamps, speaker_ids);
        } else if (config_.chunking_strategy == "hierarchical") {
            result = HierarchicalChunking(sentences, embeddings, timestamps, speaker_ids);
        }

        // Post-processing
        PostProcessChunks(result);

        return result;
    }

    std::vector<Chunk> ChunkForContext(const std::string& transcript,
                                       int context_window_size) {
        // Adaptive chunking based on context window
        config_.max_chunk_tokens = std::min(context_window_size / 2,
                                           config_.max_chunk_tokens);
        config_.target_chunk_tokens = context_window_size / 3;

        auto result = ChunkTranscript(transcript);
        return result.chunks;
    }

private:
    void InitializeResources() {
        cudaStreamCreate(&stream_);

        // Allocate device memory for embeddings
        embedding_buffer_size_ = 1000 * 768 * sizeof(float);  // Max 1000 sentences, 768 dims
        cudaMalloc(&d_embeddings_, embedding_buffer_size_);
        cudaMalloc(&d_similarity_matrix_, 1000 * 1000 * sizeof(float));
        cudaMalloc(&d_boundaries_, 1000 * sizeof(int));

        // Initialize tokenizer and embedding model
        // (Would load actual models in production)
    }

    void CleanupResources() {
        if (d_embeddings_) cudaFree(d_embeddings_);
        if (d_similarity_matrix_) cudaFree(d_similarity_matrix_);
        if (d_boundaries_) cudaFree(d_boundaries_);
        cudaStreamDestroy(stream_);
    }

    std::vector<std::string> SplitIntoSentences(const std::string& text) {
        std::vector<std::string> sentences;

        // Enhanced sentence splitting with abbreviation handling
        std::regex sentence_regex(R"((?<=[.!?])\s+(?=[A-Z]))");
        std::sregex_token_iterator iter(text.begin(), text.end(), sentence_regex, -1);
        std::sregex_token_iterator end;

        for (; iter != end; ++iter) {
            std::string sentence = *iter;

            // Clean and validate sentence
            sentence.erase(0, sentence.find_first_not_of(" \t\n\r"));
            sentence.erase(sentence.find_last_not_of(" \t\n\r") + 1);

            if (!sentence.empty() && sentence.length() > 5) {
                sentences.push_back(sentence);
            }
        }

        // Handle edge cases (very long sentences, etc.)
        std::vector<std::string> processed_sentences;
        for (const auto& sentence : sentences) {
            if (CountTokens(sentence) > config_.max_chunk_tokens) {
                // Split very long sentences
                auto sub_sentences = SplitLongSentence(sentence);
                processed_sentences.insert(processed_sentences.end(),
                                         sub_sentences.begin(),
                                         sub_sentences.end());
            } else {
                processed_sentences.push_back(sentence);
            }
        }

        return processed_sentences;
    }

    std::vector<std::vector<float>> ComputeSentenceEmbeddings(
        const std::vector<std::string>& sentences) {

        std::vector<std::vector<float>> embeddings;
        embeddings.reserve(sentences.size());

        // Check cache first
        std::vector<int> uncached_indices;
        for (size_t i = 0; i < sentences.size(); i++) {
            if (cache_.sentence_embeddings.count(sentences[i])) {
                embeddings.push_back(cache_.sentence_embeddings[sentences[i]]);
            } else {
                uncached_indices.push_back(i);
                embeddings.push_back(std::vector<float>(768, 0.0f));  // Placeholder
            }
        }

        // Compute embeddings for uncached sentences
        if (!uncached_indices.empty()) {
            // Batch compute on GPU
            BatchComputeEmbeddings(sentences, uncached_indices, embeddings);
        }

        return embeddings;
    }

    void BatchComputeEmbeddings(const std::vector<std::string>& sentences,
                               const std::vector<int>& indices,
                               std::vector<std::vector<float>>& embeddings) {
        // Prepare batch
        std::vector<std::vector<int>> token_ids;
        for (int idx : indices) {
            token_ids.push_back(TokenizeSentence(sentences[idx]));
        }

        // Copy to device
        // (Simplified - would use actual tokenizer and model)

        // Run embedding model on GPU
        const int embedding_dim = 768;
        cuda::chunking::compute_sentence_embeddings(
            d_embeddings_,  // Input token embeddings
            nullptr,        // Sentence boundaries
            d_embeddings_,  // Output sentence embeddings
            indices.size(), embedding_dim, stream_
        );

        // Copy back and cache
        std::vector<float> h_embeddings(indices.size() * embedding_dim);
        cudaMemcpyAsync(h_embeddings.data(), d_embeddings_,
                       indices.size() * embedding_dim * sizeof(float),
                       cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);

        // Update embeddings and cache
        for (size_t i = 0; i < indices.size(); i++) {
            int orig_idx = indices[i];
            std::vector<float> emb(h_embeddings.begin() + i * embedding_dim,
                                  h_embeddings.begin() + (i + 1) * embedding_dim);
            embeddings[orig_idx] = emb;
            cache_.sentence_embeddings[sentences[orig_idx]] = emb;
        }
    }

    ChunkingResult SemanticChunking(const std::vector<std::string>& sentences,
                                   const std::vector<std::vector<float>>& embeddings,
                                   const std::vector<float>& timestamps,
                                   const std::vector<int>& speaker_ids) {
        ChunkingResult result;

        // Compute similarity matrix on GPU
        ComputeSimilarityMatrix(embeddings);

        // Find optimal chunk boundaries
        std::vector<int> boundaries = FindSemanticBoundaries(embeddings.size());

        // Create chunks
        for (size_t i = 0; i < boundaries.size() - 1; i++) {
            Chunk chunk = CreateChunk(sentences, boundaries[i], boundaries[i + 1],
                                    timestamps, speaker_ids);
            result.chunks.push_back(chunk);
        }

        // Add overlaps if configured
        if (config_.overlap_tokens > 0) {
            AddChunkOverlaps(result);
        }

        return result;
    }

    void ComputeSimilarityMatrix(const std::vector<std::vector<float>>& embeddings) {
        // Flatten embeddings for GPU
        std::vector<float> flat_embeddings;
        for (const auto& emb : embeddings) {
            flat_embeddings.insert(flat_embeddings.end(), emb.begin(), emb.end());
        }

        // Copy to device
        cudaMemcpyAsync(d_embeddings_, flat_embeddings.data(),
                       flat_embeddings.size() * sizeof(float),
                       cudaMemcpyHostToDevice, stream_);

        // Compute cosine similarity matrix
        cuda::chunking::compute_similarity_matrix(
            d_embeddings_, d_similarity_matrix_,
            embeddings.size(), 768, stream_
        );
    }

    std::vector<int> FindSemanticBoundaries(int num_sentences) {
        std::vector<int> boundaries;
        boundaries.push_back(0);

        // Get sentence lengths for constraints
        std::vector<int> sentence_lengths(num_sentences);
        // (Would compute actual token counts)

        // Find optimal boundaries on GPU
        cuda::chunking::find_optimal_boundaries(
            d_similarity_matrix_,
            sentence_lengths.data(),
            d_boundaries_,
            num_sentences,
            config_.max_chunk_tokens,
            config_.semantic_similarity_threshold,
            stream_
        );

        // Copy back boundaries
        std::vector<int> h_boundaries(num_sentences);
        cudaMemcpyAsync(h_boundaries.data(), d_boundaries_,
                       num_sentences * sizeof(int),
                       cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);

        // Extract actual boundary indices
        for (int i = 0; i < num_sentences; i++) {
            if (h_boundaries[i] == 1) {
                boundaries.push_back(i);
            }
        }
        boundaries.push_back(num_sentences);

        return boundaries;
    }

    ChunkingResult SlidingWindowChunking(const std::vector<std::string>& sentences,
                                        const std::vector<float>& timestamps,
                                        const std::vector<int>& speaker_ids) {
        ChunkingResult result;

        int current_tokens = 0;
        int chunk_start = 0;

        for (size_t i = 0; i < sentences.size(); i++) {
            int sentence_tokens = CountTokens(sentences[i]);

            if (current_tokens + sentence_tokens > config_.target_chunk_tokens &&
                current_tokens >= config_.min_chunk_tokens) {

                // Create chunk
                Chunk chunk = CreateChunk(sentences, chunk_start, i,
                                        timestamps, speaker_ids);
                result.chunks.push_back(chunk);

                // Sliding window overlap
                int overlap_start = std::max(0, static_cast<int>(i) -
                    (config_.overlap_tokens / 20));  // Approximate sentences
                chunk_start = overlap_start;
                current_tokens = CalculateTokens(sentences, overlap_start, i);
            }

            current_tokens += sentence_tokens;
        }

        // Add final chunk
        if (chunk_start < sentences.size()) {
            Chunk chunk = CreateChunk(sentences, chunk_start, sentences.size(),
                                    timestamps, speaker_ids);
            result.chunks.push_back(chunk);
        }

        return result;
    }

    ChunkingResult HierarchicalChunking(const std::vector<std::string>& sentences,
                                       const std::vector<std::vector<float>>& embeddings,
                                       const std::vector<float>& timestamps,
                                       const std::vector<int>& speaker_ids) {
        ChunkingResult result;

        // First level: Topic segmentation
        auto topics = DetectTopics(sentences, embeddings);

        // Second level: Chunk within topics
        for (const auto& topic_segment : topics) {
            auto topic_chunks = ChunkTopicSegment(
                sentences, embeddings, topic_segment,
                timestamps, speaker_ids
            );

            for (auto& chunk : topic_chunks) {
                chunk.topic = topic_segment.topic_name;
                result.chunks.push_back(chunk);
            }
        }

        // Third level: Merge small chunks if needed
        MergeSmallChunks(result);

        return result;
    }

    struct TopicSegment {
        int start_idx;
        int end_idx;
        std::string topic_name;
        float coherence_score;
    };

    std::vector<TopicSegment> DetectTopics(const std::vector<std::string>& sentences,
                                          const std::vector<std::vector<float>>& embeddings) {
        std::vector<TopicSegment> topics;

        // Use clustering or change point detection
        // Simplified implementation
        int segment_size = sentences.size() / 5;  // Assume 5 topics

        for (int i = 0; i < 5 && i * segment_size < sentences.size(); i++) {
            TopicSegment segment;
            segment.start_idx = i * segment_size;
            segment.end_idx = std::min((i + 1) * segment_size,
                                      static_cast<int>(sentences.size()));
            segment.topic_name = "Topic_" + std::to_string(i + 1);
            segment.coherence_score = CalculateCoherence(embeddings,
                                                        segment.start_idx,
                                                        segment.end_idx);
            topics.push_back(segment);
        }

        return topics;
    }

    std::vector<Chunk> ChunkTopicSegment(const std::vector<std::string>& sentences,
                                        const std::vector<std::vector<float>>& embeddings,
                                        const TopicSegment& segment,
                                        const std::vector<float>& timestamps,
                                        const std::vector<int>& speaker_ids) {
        std::vector<Chunk> chunks;

        // Extract sentences for this topic
        std::vector<std::string> topic_sentences(
            sentences.begin() + segment.start_idx,
            sentences.begin() + segment.end_idx
        );

        std::vector<std::vector<float>> topic_embeddings(
            embeddings.begin() + segment.start_idx,
            embeddings.begin() + segment.end_idx
        );

        // Chunk within topic maintaining context
        auto topic_result = SemanticChunking(topic_sentences, topic_embeddings,
                                            timestamps, speaker_ids);

        // Adjust indices to global positions
        for (auto& chunk : topic_result.chunks) {
            chunk.start_token += segment.start_idx;
            chunk.end_token += segment.start_idx;
            chunks.push_back(chunk);
        }

        return chunks;
    }

    void MergeSmallChunks(ChunkingResult& result) {
        std::vector<Chunk> merged_chunks;

        for (size_t i = 0; i < result.chunks.size(); i++) {
            if (CountTokens(result.chunks[i].text) < config_.min_chunk_tokens &&
                i + 1 < result.chunks.size()) {

                // Merge with next chunk if semantically similar
                float similarity = ComputeChunkSimilarity(result.chunks[i],
                                                         result.chunks[i + 1]);

                if (similarity > config_.semantic_similarity_threshold) {
                    Chunk merged = MergeChunks(result.chunks[i], result.chunks[i + 1]);
                    merged_chunks.push_back(merged);
                    i++;  // Skip next chunk
                } else {
                    merged_chunks.push_back(result.chunks[i]);
                }
            } else {
                merged_chunks.push_back(result.chunks[i]);
            }
        }

        result.chunks = merged_chunks;
    }

    Chunk CreateChunk(const std::vector<std::string>& sentences,
                     int start_idx, int end_idx,
                     const std::vector<float>& timestamps,
                     const std::vector<int>& speaker_ids) {
        Chunk chunk;

        // Combine sentences
        std::stringstream text;
        for (int i = start_idx; i < end_idx; i++) {
            if (i > start_idx) text << " ";
            text << sentences[i];
        }
        chunk.text = text.str();

        // Set indices
        chunk.start_token = start_idx;  // Would be actual token index
        chunk.end_token = end_idx;

        // Set timestamps if available
        if (!timestamps.empty() && start_idx < timestamps.size()) {
            chunk.start_timestamp = timestamps[start_idx];
            chunk.end_timestamp = timestamps[std::min(end_idx - 1,
                                                     static_cast<int>(timestamps.size()) - 1)];
        }

        // Set speaker if available
        if (!speaker_ids.empty() && start_idx < speaker_ids.size()) {
            chunk.speaker_id = speaker_ids[start_idx];
        }

        // Extract keywords
        chunk.keywords = ExtractKeywords(chunk.text);

        // Calculate coherence
        chunk.coherence_score = CalculateChunkCoherence(sentences, start_idx, end_idx);

        // Store sentence indices
        for (int i = start_idx; i < end_idx; i++) {
            chunk.sentence_indices.push_back(i);
        }

        return chunk;
    }

    void AddChunkOverlaps(ChunkingResult& result) {
        for (size_t i = 0; i < result.chunks.size() - 1; i++) {
            int overlap_sentences = config_.overlap_tokens / 20;  // Approximate

            // Find overlap sentences
            int overlap_start = std::max(0,
                result.chunks[i].end_token - overlap_sentences);
            int overlap_end = std::min(
                static_cast<int>(result.chunks[i + 1].sentence_indices.size()),
                overlap_sentences);

            result.chunk_overlaps.push_back({overlap_start, overlap_end});
        }
    }

    void PostProcessChunks(ChunkingResult& result) {
        // Calculate statistics
        float total_coherence = 0.0f;
        for (const auto& chunk : result.chunks) {
            total_coherence += chunk.coherence_score;
        }
        result.avg_coherence_score = total_coherence / result.chunks.size();
        result.total_chunks = result.chunks.size();

        // Extract unique topics
        std::unordered_set<std::string> unique_topics;
        for (const auto& chunk : result.chunks) {
            if (!chunk.topic.empty()) {
                unique_topics.insert(chunk.topic);
            }
        }
        result.detected_topics.assign(unique_topics.begin(), unique_topics.end());

        // Build speaker map
        std::unordered_set<int> unique_speakers;
        for (const auto& chunk : result.chunks) {
            unique_speakers.insert(chunk.speaker_id);
        }
        for (int speaker : unique_speakers) {
            result.speaker_map[speaker] = "Speaker_" + std::to_string(speaker);
        }
    }

    // Utility functions
    int CountTokens(const std::string& text) {
        // Approximate: 1 token ≈ 4 characters
        return text.length() / 4;
    }

    int CalculateTokens(const std::vector<std::string>& sentences,
                       int start, int end) {
        int tokens = 0;
        for (int i = start; i < end; i++) {
            tokens += CountTokens(sentences[i]);
        }
        return tokens;
    }

    std::vector<std::string> SplitLongSentence(const std::string& sentence) {
        std::vector<std::string> parts;

        // Split on conjunctions and punctuation
        std::regex split_regex(R"((?:,\s+(?:and|or|but)\s+)|(?:;\s+))");
        std::sregex_token_iterator iter(sentence.begin(), sentence.end(),
                                       split_regex, -1);
        std::sregex_token_iterator end;

        for (; iter != end; ++iter) {
            parts.push_back(*iter);
        }

        return parts;
    }

    std::vector<int> TokenizeSentence(const std::string& sentence) {
        // Simplified tokenization (would use actual tokenizer)
        std::vector<int> tokens;
        std::istringstream iss(sentence);
        std::string word;

        while (iss >> word) {
            // Convert word to token ID (simplified)
            tokens.push_back(std::hash<std::string>{}(word) % 50000);
        }

        return tokens;
    }

    std::vector<std::string> ExtractKeywords(const std::string& text) {
        // Simplified keyword extraction (would use TF-IDF or RAKE)
        std::vector<std::string> keywords;
        std::unordered_map<std::string, int> word_freq;

        std::istringstream iss(text);
        std::string word;
        while (iss >> word) {
            // Remove punctuation
            word.erase(std::remove_if(word.begin(), word.end(), ::ispunct),
                      word.end());

            // Convert to lowercase
            std::transform(word.begin(), word.end(), word.begin(), ::tolower);

            if (word.length() > 4) {  // Skip short words
                word_freq[word]++;
            }
        }

        // Get top 5 keywords
        std::vector<std::pair<std::string, int>> sorted_words(
            word_freq.begin(), word_freq.end());

        std::sort(sorted_words.begin(), sorted_words.end(),
                 [](const auto& a, const auto& b) {
                     return a.second > b.second;
                 });

        for (size_t i = 0; i < std::min(size_t(5), sorted_words.size()); i++) {
            keywords.push_back(sorted_words[i].first);
        }

        return keywords;
    }

    float CalculateCoherence(const std::vector<std::vector<float>>& embeddings,
                            int start_idx, int end_idx) {
        if (end_idx - start_idx <= 1) return 1.0f;

        float total_similarity = 0.0f;
        int comparisons = 0;

        for (int i = start_idx; i < end_idx - 1; i++) {
            for (int j = i + 1; j < end_idx; j++) {
                total_similarity += CosineSimilarity(embeddings[i], embeddings[j]);
                comparisons++;
            }
        }

        return comparisons > 0 ? total_similarity / comparisons : 1.0f;
    }

    float CalculateChunkCoherence(const std::vector<std::string>& sentences,
                                 int start_idx, int end_idx) {
        // Calculate based on lexical overlap and continuity
        float coherence = 1.0f;

        for (int i = start_idx; i < end_idx - 1; i++) {
            float overlap = LexicalOverlap(sentences[i], sentences[i + 1]);
            coherence *= (0.5f + 0.5f * overlap);
        }

        return coherence;
    }

    float ComputeChunkSimilarity(const Chunk& chunk1, const Chunk& chunk2) {
        // Compare keywords
        std::unordered_set<std::string> keywords1(chunk1.keywords.begin(),
                                                  chunk1.keywords.end());
        std::unordered_set<std::string> keywords2(chunk2.keywords.begin(),
                                                  chunk2.keywords.end());

        int intersection = .0f;
        for (const auto& kw : keywords1) {
            if (keywords2.count(kw)) intersection++;
        }

        int union_size = keywords1.size() + keywords2.size() - intersection;
        return union_size > 0 ? float(intersection) / union_size : 0.0f;
    }

    Chunk MergeChunks(const Chunk& chunk1, const Chunk& chunk2) {
        Chunk merged;

        merged.text = chunk1.text + " " + chunk2.text;
        merged.start_token = chunk1.start_token;
        merged.end_token = chunk2.end_token;
        merged.start_timestamp = chunk1.start_timestamp;
        merged.end_timestamp = chunk2.end_timestamp;

        // Merge keywords
        merged.keywords = chunk1.keywords;
        merged.keywords.insert(merged.keywords.end(),
                              chunk2.keywords.begin(),
                              chunk2.keywords.end());

        // Remove duplicates
        std::sort(merged.keywords.begin(), merged.keywords.end());
        merged.keywords.erase(std::unique(merged.keywords.begin(),
                                        merged.keywords.end()),
                            merged.keywords.end());

        // Average coherence scores
        merged.coherence_score = (chunk1.coherence_score + chunk2.coherence_score) / 2.0f;

        // Merge sentence indices
        merged.sentence_indices = chunk1.sentence_indices;
        merged.sentence_indices.insert(merged.sentence_indices.end(),
                                      chunk2.sentence_indices.begin(),
                                      chunk2.sentence_indices.end());

        return merged;
    }

    float CosineSimilarity(const std::vector<float>& v1,
                          const std::vector<float>& v2) {
        float dot = 0.0f, norm1 = 0.0f, norm2 = 0.0f;

        for (size_t i = 0; i < v1.size(); i++) {
            dot += v1[i] * v2[i];
            norm1 += v1[i] * v1[i];
            norm2 += v2[i] * v2[i];
        }

        return dot / (std::sqrt(norm1) * std::sqrt(norm2) + 1e-8f);
    }

    float LexicalOverlap(const std::string& s1, const std::string& s2) {
        std::unordered_set<std::string> words1, words2;

        std::istringstream iss1(s1), iss2(s2);
        std::string word;

        while (iss1 >> word) words1.insert(word);
        while (iss2 >> word) words2.insert(word);

        int intersection = 0;
        for (const auto& w : words1) {
            if (words2.count(w)) intersection++;
        }

        int union_size = words1.size() + words2.size() - intersection;
        return union_size > 0 ? float(intersection) / union_size : 0.0f;
    }
};

// Hierarchical summarization using chunks
class HierarchicalSummarizer {
private:
    std::unique_ptr<ContextAwareChunker> chunker_;
    std::unique_ptr<class LLMEngine> llm_;

public:
    struct HierarchicalConfig {
        int max_depth = 3;
        float compression_ratio = 0.3f;
        std::string aggregation_method = "recursive";  // recursive, map-reduce, tree
        bool preserve_key_details = true;
        bool maintain_chronology = true;
    };

    HierarchicalSummarizer(const HierarchicalConfig& config = {})
        : config_(config) {
        chunker_ = std::make_unique<ContextAwareChunker>();
    }

    std::string SummarizeLongDocument(const std::string& document) {
        // Step 1: Chunk the document
        auto chunks = chunker_->ChunkTranscript(document);

        // Step 2: Summarize each chunk
        std::vector<std::string> chunk_summaries;
        for (const auto& chunk : chunks.chunks) {
            std::string summary = SummarizeChunk(chunk);
            chunk_summaries.push_back(summary);
        }

        // Step 3: Hierarchical aggregation
        std::string final_summary;
        if (config_.aggregation_method == "recursive") {
            final_summary = RecursiveAggregation(chunk_summaries);
        } else if (config_.aggregation_method == "map-reduce") {
            final_summary = MapReduceAggregation(chunk_summaries);
        } else if (config_.aggregation_method == "tree") {
            final_summary = TreeAggregation(chunk_summaries);
        }

        return final_summary;
    }

private:
    HierarchicalConfig config_;

    std::string SummarizeChunk(const ContextAwareChunker::Chunk& chunk) {
        // Generate prompt for chunk summarization
        std::string prompt = "Summarize the following text segment concisely:\n\n";
        prompt += chunk.text;
        prompt += "\n\nSummary:";

        // Call LLM (simplified)
        return "Summary of chunk " + std::to_string(chunk.start_token);
    }

    std::string RecursiveAggregation(const std::vector<std::string>& summaries) {
        if (summaries.size() <= 1) {
            return summaries.empty() ? "" : summaries[0];
        }

        // Combine summaries in pairs
        std::vector<std::string> next_level;
        for (size_t i = 0; i < summaries.size(); i += 2) {
            std::string combined;
            if (i + 1 < summaries.size()) {
                combined = CombineSummaries(summaries[i], summaries[i + 1]);
            } else {
                combined = summaries[i];
            }
            next_level.push_back(combined);
        }

        // Recurse
        return RecursiveAggregation(next_level);
    }

    std::string MapReduceAggregation(const std::vector<std::string>& summaries) {
        // Map phase: Extract key points from each summary
        std::vector<std::string> key_points;
        for (const auto& summary : summaries) {
            key_points.push_back(ExtractKeyPoints(summary));
        }

        // Reduce phase: Combine all key points
        return CombineKeyPoints(key_points);
    }

    std::string TreeAggregation(const std::vector<std::string>& summaries) {
        // Build tree structure
        int num_leaves = summaries.size();
        int tree_height = std::ceil(std::log2(num_leaves));

        // Bottom-up aggregation
        std::vector<std::string> current_level = summaries;

        for (int level = 0; level < tree_height; level++) {
            std::vector<std::string> next_level;

            for (size_t i = 0; i < current_level.size(); i += 2) {
                if (i + 1 < current_level.size()) {
                    next_level.push_back(
                        CombineSummaries(current_level[i], current_level[i + 1])
                    );
                } else {
                    next_level.push_back(current_level[i]);
                }
            }

            current_level = next_level;
            if (current_level.size() == 1) break;
        }

        return current_level[0];
    }

    std::string CombineSummaries(const std::string& s1, const std::string& s2) {
        std::string prompt = "Combine these two summaries into one cohesive summary:\n\n";
        prompt += "Summary 1: " + s1 + "\n\n";
        prompt += "Summary 2: " + s2 + "\n\n";
        prompt += "Combined summary:";

        // Call LLM (simplified)
        return "Combined: " + s1.substr(0, 50) + " + " + s2.substr(0, 50);
    }

    std::string ExtractKeyPoints(const std::string& summary) {
        // Extract main points (simplified)
        return "Key points from: " + summary.substr(0, 100);
    }

    std::string CombineKeyPoints(const std::vector<std::string>& key_points) {
        std::stringstream combined;
        combined << "Final summary combining " << key_points.size() << " segments:\n";
        for (const auto& points : key_points) {
            combined << "- " << points << "\n";
        }
        return combined.str();
    }
};

// Global instances
static std::unique_ptr<ContextAwareChunker> g_chunker;
static std::unique_ptr<HierarchicalSummarizer> g_hierarchical_summarizer;

void InitializeChunking() {
    g_chunker = std::make_unique<ContextAwareChunker>();
    g_hierarchical_summarizer = std::make_unique<HierarchicalSummarizer>();
}

std::vector<ContextAwareChunker::Chunk> ChunkTranscript(const std::string& transcript) {
    if (!g_chunker) InitializeChunking();
    auto result = g_chunker->ChunkTranscript(transcript);
    return result.chunks;
}

std::string SummarizeLongTranscript(const std::string& transcript) {
    if (!g_hierarchical_summarizer) InitializeChunking();
    return g_hierarchical_summarizer->SummarizeLongDocument(transcript);
}

} // namespace summarization