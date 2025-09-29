#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <regex>
#include <cuda_runtime.h>

namespace summarization {

// Forward declarations for CUDA kernels
namespace cuda {
namespace quality {
    void compute_sentence_embeddings(const float* token_embeddings,
                                    float* sentence_embeddings,
                                    int num_sentences, int embedding_dim,
                                    cudaStream_t stream);

    void compute_similarity_scores(const float* embeddings1,
                                  const float* embeddings2,
                                  float* scores,
                                  int num_vectors, int embedding_dim,
                                  cudaStream_t stream);

    void compute_bertscore(const float* reference_embeddings,
                         const float* candidate_embeddings,
                         float* precision, float* recall, float* f1,
                         int ref_len, int cand_len, int embedding_dim,
                         cudaStream_t stream);
}
}

class SummaryQualityValidator {
public:
    struct QualityMetrics {
        // Content Coverage
        float coverage_score;           // How well key information is covered
        float completeness_score;       // Completeness of summary
        float relevance_score;          // Relevance of included information

        // Coherence & Fluency
        float coherence_score;          // Logical flow and structure
        float fluency_score;            // Language quality and readability
        float grammaticality_score;     // Grammar correctness

        // Factual Accuracy
        float factuality_score;         // Factual consistency with source
        float hallucination_score;      // Inverse of hallucination detection
        float consistency_score;        // Internal consistency

        // Information Density
        float conciseness_score;        // Brevity without loss of info
        float redundancy_score;         // Lack of repetition
        float informativeness_score;    // Information per word ratio

        // Semantic Similarity
        float semantic_similarity;      // Similarity to source
        float bertscore_precision;     // BERTScore metrics
        float bertscore_recall;
        float bertscore_f1;

        // Overall Score
        float overall_quality_score;    // Weighted combination
        std::string quality_grade;      // A, B, C, D, F
        std::vector<std::string> issues; // Detected issues
        std::vector<std::string> strengths; // Identified strengths
    };

    struct ValidationConfig {
        float min_acceptable_score = 0.7f;
        bool check_factuality = true;
        bool check_hallucination = true;
        bool check_completeness = true;
        bool check_redundancy = true;
        bool use_reference_summary = false;
        int max_summary_length = 500;
        int min_summary_length = 50;
        float length_penalty_factor = 0.1f;
        std::unordered_map<std::string, float> metric_weights;
    };

private:
    ValidationConfig config_;
    std::unique_ptr<class FactualityChecker> fact_checker_;
    std::unique_ptr<class CoherenceAnalyzer> coherence_analyzer_;
    std::unique_ptr<class SemanticSimilarityModel> similarity_model_;

    // CUDA resources
    cudaStream_t stream_;
    float* d_embeddings_;
    float* d_scores_;
    size_t embedding_buffer_size_;

    // Cache for embeddings
    std::unordered_map<std::string, std::vector<float>> embedding_cache_;

public:
    SummaryQualityValidator(const ValidationConfig& config = {})
        : config_(config) {
        InitializeDefaultWeights();
        InitializeCUDA();
        InitializeModels();
    }

    ~SummaryQualityValidator() {
        if (d_embeddings_) cudaFree(d_embeddings_);
        if (d_scores_) cudaFree(d_scores_);
        cudaStreamDestroy(stream_);
    }

    QualityMetrics ValidateSummary(const std::string& summary,
                                  const std::string& source_text,
                                  const std::string& reference_summary = "") {
        QualityMetrics metrics;

        // Compute individual metrics
        metrics.coverage_score = ComputeCoverageScore(summary, source_text);
        metrics.completeness_score = ComputeCompletenessScore(summary, source_text);
        metrics.relevance_score = ComputeRelevanceScore(summary, source_text);

        metrics.coherence_score = ComputeCoherenceScore(summary);
        metrics.fluency_score = ComputeFluencyScore(summary);
        metrics.grammaticality_score = ComputeGrammaticalityScore(summary);

        if (config_.check_factuality) {
            metrics.factuality_score = ComputeFactualityScore(summary, source_text);
        }

        if (config_.check_hallucination) {
            metrics.hallucination_score = 1.0f - DetectHallucination(summary, source_text);
        }

        metrics.consistency_score = ComputeConsistencyScore(summary);

        metrics.conciseness_score = ComputeConcisenessScore(summary, source_text);

        if (config_.check_redundancy) {
            metrics.redundancy_score = 1.0f - DetectRedundancy(summary);
        }

        metrics.informativeness_score = ComputeInformativenessScore(summary);

        // Compute semantic similarity metrics
        auto semantic_metrics = ComputeSemanticSimilarity(summary, source_text);
        metrics.semantic_similarity = semantic_metrics.similarity;
        metrics.bertscore_precision = semantic_metrics.precision;
        metrics.bertscore_recall = semantic_metrics.recall;
        metrics.bertscore_f1 = semantic_metrics.f1;

        // Compare with reference if available
        if (!reference_summary.empty() && config_.use_reference_summary) {
            AdjustMetricsWithReference(metrics, summary, reference_summary);
        }

        // Compute overall score
        metrics.overall_quality_score = ComputeOverallScore(metrics);

        // Assign grade
        metrics.quality_grade = AssignQualityGrade(metrics.overall_quality_score);

        // Identify issues and strengths
        IdentifyIssues(metrics);
        IdentifyStrengths(metrics);

        return metrics;
    }

    bool PassesQualityThreshold(const QualityMetrics& metrics) {
        return metrics.overall_quality_score >= config_.min_acceptable_score;
    }

    std::string GenerateQualityReport(const QualityMetrics& metrics) {
        std::stringstream report;

        report << "=== Summary Quality Report ===\n\n";

        report << "Overall Quality: " << metrics.quality_grade
               << " (" << std::fixed << std::setprecision(2)
               << metrics.overall_quality_score * 100 << "%)\n\n";

        report << "Detailed Metrics:\n";
        report << "├─ Content Coverage: " << FormatScore(metrics.coverage_score) << "\n";
        report << "├─ Completeness: " << FormatScore(metrics.completeness_score) << "\n";
        report << "├─ Relevance: " << FormatScore(metrics.relevance_score) << "\n";
        report << "├─ Coherence: " << FormatScore(metrics.coherence_score) << "\n";
        report << "├─ Fluency: " << FormatScore(metrics.fluency_score) << "\n";
        report << "├─ Factuality: " << FormatScore(metrics.factuality_score) << "\n";
        report << "├─ Conciseness: " << FormatScore(metrics.conciseness_score) << "\n";
        report << "├─ Informativeness: " << FormatScore(metrics.informativeness_score) << "\n";
        report << "└─ Semantic Similarity: " << FormatScore(metrics.semantic_similarity) << "\n";

        if (!metrics.issues.empty()) {
            report << "\nIdentified Issues:\n";
            for (const auto& issue : metrics.issues) {
                report << "⚠ " << issue << "\n";
            }
        }

        if (!metrics.strengths.empty()) {
            report << "\nStrengths:\n";
            for (const auto& strength : metrics.strengths) {
                report << "✓ " << strength << "\n";
            }
        }

        report << "\nRecommendations:\n";
        report << GenerateRecommendations(metrics);

        return report.str();
    }

private:
    void InitializeDefaultWeights() {
        config_.metric_weights = {
            {"coverage", 0.20f},
            {"completeness", 0.15f},
            {"relevance", 0.15f},
            {"coherence", 0.10f},
            {"fluency", 0.10f},
            {"factuality", 0.15f},
            {"conciseness", 0.10f},
            {"informativeness", 0.05f}
        };
    }

    void InitializeCUDA() {
        cudaStreamCreate(&stream_);

        embedding_buffer_size_ = 1024 * 768 * sizeof(float);  // Max 1024 sentences
        cudaMalloc(&d_embeddings_, embedding_buffer_size_ * 2);  // For source and summary
        cudaMalloc(&d_scores_, 1024 * 1024 * sizeof(float));    // Similarity matrix
    }

    void InitializeModels() {
        // Initialize sub-models for quality checking
        // These would be actual model implementations in production
    }

    float ComputeCoverageScore(const std::string& summary, const std::string& source) {
        // Extract key information from source
        auto key_info = ExtractKeyInformation(source);

        // Check how many key points are covered in summary
        int covered = 0;
        for (const auto& info : key_info) {
            if (IsInformationPresent(info, summary)) {
                covered++;
            }
        }

        return key_info.empty() ? 1.0f : float(covered) / key_info.size();
    }

    float ComputeCompletenessScore(const std::string& summary, const std::string& source) {
        // Check if summary covers beginning, middle, and end
        auto source_sections = SplitIntoSections(source, 3);
        auto summary_sentences = SplitIntoSentences(summary);

        float section_coverage = 0.0f;
        for (const auto& section : source_sections) {
            float best_match = 0.0f;
            for (const auto& sentence : summary_sentences) {
                float similarity = ComputeSentenceSimilarity(sentence, section);
                best_match = std::max(best_match, similarity);
            }
            section_coverage += best_match;
        }

        return section_coverage / source_sections.size();
    }

    float ComputeRelevanceScore(const std::string& summary, const std::string& source) {
        auto summary_sentences = SplitIntoSentences(summary);

        float total_relevance = 0.0f;
        for (const auto& sentence : summary_sentences) {
            float relevance = ComputeSentenceRelevance(sentence, source);
            total_relevance += relevance;
        }

        return summary_sentences.empty() ? 0.0f :
               total_relevance / summary_sentences.size();
    }

    float ComputeCoherenceScore(const std::string& summary) {
        auto sentences = SplitIntoSentences(summary);
        if (sentences.size() < 2) return 1.0f;

        float total_coherence = 0.0f;

        // Check coherence between consecutive sentences
        for (size_t i = 0; i < sentences.size() - 1; i++) {
            float coherence = ComputeSentencePairCoherence(sentences[i], sentences[i + 1]);
            total_coherence += coherence;
        }

        // Check for logical connectors
        float connector_score = CheckLogicalConnectors(summary);

        return (total_coherence / (sentences.size() - 1) + connector_score) / 2.0f;
    }

    float ComputeFluencyScore(const std::string& summary) {
        float perplexity_score = 1.0f - (ComputePerplexity(summary) / 100.0f);
        float readability_score = ComputeReadabilityScore(summary);
        float sentence_variety_score = ComputeSentenceVariety(summary);

        return (perplexity_score + readability_score + sentence_variety_score) / 3.0f;
    }

    float ComputeGrammaticalityScore(const std::string& summary) {
        // Simple grammar checks (would use language model in production)
        int errors = 0;

        // Check for basic grammar issues
        errors += CountGrammarErrors(summary);

        // Check for proper sentence structure
        auto sentences = SplitIntoSentences(summary);
        for (const auto& sentence : sentences) {
            if (!IsProperSentence(sentence)) errors++;
        }

        int total_sentences = sentences.size();
        return total_sentences == 0 ? 1.0f :
               1.0f - (float(errors) / total_sentences);
    }

    float ComputeFactualityScore(const std::string& summary, const std::string& source) {
        auto summary_facts = ExtractFacts(summary);
        auto source_facts = ExtractFacts(source);

        int correct_facts = 0;
        int total_facts = summary_facts.size();

        for (const auto& fact : summary_facts) {
            if (IsFactSupported(fact, source_facts)) {
                correct_facts++;
            }
        }

        return total_facts == 0 ? 1.0f : float(correct_facts) / total_facts;
    }

    float DetectHallucination(const std::string& summary, const std::string& source) {
        // Detect information in summary not present in source
        auto summary_entities = ExtractEntities(summary);
        auto source_entities = ExtractEntities(source);

        int hallucinated = 0;
        for (const auto& entity : summary_entities) {
            if (!IsEntityInSource(entity, source_entities)) {
                hallucinated++;
            }
        }

        // Check for unsupported claims
        auto claims = ExtractClaims(summary);
        for (const auto& claim : claims) {
            if (!IsClaimSupported(claim, source)) {
                hallucinated++;
            }
        }

        int total_checkable = summary_entities.size() + claims.size();
        return total_checkable == 0 ? 0.0f : float(hallucinated) / total_checkable;
    }

    float ComputeConsistencyScore(const std::string& summary) {
        // Check for internal contradictions
        auto sentences = SplitIntoSentences(summary);

        int contradictions = 0;
        for (size_t i = 0; i < sentences.size(); i++) {
            for (size_t j = i + 1; j < sentences.size(); j++) {
                if (AreContradictory(sentences[i], sentences[j])) {
                    contradictions++;
                }
            }
        }

        int total_pairs = (sentences.size() * (sentences.size() - 1)) / 2;
        return total_pairs == 0 ? 1.0f : 1.0f - (float(contradictions) / total_pairs);
    }

    float ComputeConcisenessScore(const std::string& summary, const std::string& source) {
        // Compression ratio
        float compression_ratio = float(summary.length()) / source.length();

        // Ideal compression ratio is around 0.15-0.25
        float ideal_ratio = 0.20f;
        float ratio_score = 1.0f - std::abs(compression_ratio - ideal_ratio) / ideal_ratio;

        // Check for verbose phrases
        float verbosity_score = 1.0f - DetectVerbosity(summary);

        return (ratio_score + verbosity_score) / 2.0f;
    }

    float DetectRedundancy(const std::string& summary) {
        auto sentences = SplitIntoSentences(summary);

        float total_redundancy = 0.0f;
        for (size_t i = 0; i < sentences.size(); i++) {
            for (size_t j = i + 1; j < sentences.size(); j++) {
                float similarity = ComputeSentenceSimilarity(sentences[i], sentences[j]);
                if (similarity > 0.8f) {  // High similarity indicates redundancy
                    total_redundancy += similarity - 0.8f;
                }
            }
        }

        int total_pairs = (sentences.size() * (sentences.size() - 1)) / 2;
        return total_pairs == 0 ? 0.0f : total_redundancy / total_pairs;
    }

    float ComputeInformativenessScore(const std::string& summary) {
        // Information density: unique information per word
        auto words = TokenizeWords(summary);
        std::unordered_set<std::string> unique_content_words;

        for (const auto& word : words) {
            if (IsContentWord(word)) {
                unique_content_words.insert(word);
            }
        }

        float density = words.empty() ? 0.0f :
                       float(unique_content_words.size()) / words.size();

        // Entropy-based informativeness
        float entropy = ComputeTextEntropy(summary);

        return (density + entropy / 10.0f) / 2.0f;  // Normalize entropy
    }

    struct SemanticMetrics {
        float similarity;
        float precision;
        float recall;
        float f1;
    };

    SemanticMetrics ComputeSemanticSimilarity(const std::string& summary,
                                             const std::string& source) {
        // Get embeddings
        auto summary_embeddings = GetSentenceEmbeddings(summary);
        auto source_embeddings = GetSentenceEmbeddings(source);

        // Flatten embeddings
        std::vector<float> flat_summary, flat_source;
        for (const auto& emb : summary_embeddings) {
            flat_summary.insert(flat_summary.end(), emb.begin(), emb.end());
        }
        for (const auto& emb : source_embeddings) {
            flat_source.insert(flat_source.end(), emb.begin(), emb.end());
        }

        // Copy to GPU
        cudaMemcpyAsync(d_embeddings_, flat_summary.data(),
                       flat_summary.size() * sizeof(float),
                       cudaMemcpyHostToDevice, stream_);
        cudaMemcpyAsync(d_embeddings_ + flat_summary.size(), flat_source.data(),
                       flat_source.size() * sizeof(float),
                       cudaMemcpyHostToDevice, stream_);

        // Compute BERTScore
        float precision, recall, f1;
        cuda::quality::compute_bertscore(
            d_embeddings_ + flat_summary.size(),  // Reference (source)
            d_embeddings_,                         // Candidate (summary)
            &precision, &recall, &f1,
            source_embeddings.size(),
            summary_embeddings.size(),
            768,  // Embedding dimension
            stream_
        );

        cudaStreamSynchronize(stream_);

        // Compute overall similarity
        float similarity = ComputeCosineSimilarity(
            ComputeMeanPooling(summary_embeddings),
            ComputeMeanPooling(source_embeddings)
        );

        return {similarity, precision, recall, f1};
    }

    void AdjustMetricsWithReference(QualityMetrics& metrics,
                                   const std::string& summary,
                                   const std::string& reference) {
        // Compare with reference summary
        auto ref_similarity = ComputeSemanticSimilarity(summary, reference);

        // Adjust scores based on reference similarity
        float adjustment_factor = ref_similarity.f1;

        metrics.coverage_score = (metrics.coverage_score +
            adjustment_factor * ComputeCoverageScore(summary, reference)) / 2.0f;

        metrics.relevance_score = (metrics.relevance_score +
            adjustment_factor * ComputeRelevanceScore(summary, reference)) / 2.0f;
    }

    float ComputeOverallScore(const QualityMetrics& metrics) {
        float weighted_sum = 0.0f;
        float total_weight = 0.0f;

        // Apply weights
        weighted_sum += metrics.coverage_score * config_.metric_weights["coverage"];
        weighted_sum += metrics.completeness_score * config_.metric_weights["completeness"];
        weighted_sum += metrics.relevance_score * config_.metric_weights["relevance"];
        weighted_sum += metrics.coherence_score * config_.metric_weights["coherence"];
        weighted_sum += metrics.fluency_score * config_.metric_weights["fluency"];
        weighted_sum += metrics.factuality_score * config_.metric_weights["factuality"];
        weighted_sum += metrics.conciseness_score * config_.metric_weights["conciseness"];
        weighted_sum += metrics.informativeness_score * config_.metric_weights["informativeness"];

        // Sum weights
        for (const auto& [metric, weight] : config_.metric_weights) {
            total_weight += weight;
        }

        return total_weight == 0.0f ? 0.0f : weighted_sum / total_weight;
    }

    std::string AssignQualityGrade(float score) {
        if (score >= 0.9f) return "A";
        if (score >= 0.8f) return "B";
        if (score >= 0.7f) return "C";
        if (score >= 0.6f) return "D";
        return "F";
    }

    void IdentifyIssues(QualityMetrics& metrics) {
        metrics.issues.clear();

        if (metrics.coverage_score < 0.6f) {
            metrics.issues.push_back("Incomplete coverage of source content");
        }

        if (metrics.coherence_score < 0.6f) {
            metrics.issues.push_back("Poor logical flow and structure");
        }

        if (metrics.factuality_score < 0.8f) {
            metrics.issues.push_back("Factual inconsistencies detected");
        }

        if (metrics.hallucination_score < 0.9f) {
            metrics.issues.push_back("Possible hallucinated information");
        }

        if (metrics.redundancy_score < 0.8f) {
            metrics.issues.push_back("Redundant information present");
        }

        if (metrics.conciseness_score < 0.6f) {
            metrics.issues.push_back("Summary is too verbose");
        }

        if (metrics.grammaticality_score < 0.8f) {
            metrics.issues.push_back("Grammar issues detected");
        }
    }

    void IdentifyStrengths(QualityMetrics& metrics) {
        metrics.strengths.clear();

        if (metrics.coverage_score >= 0.9f) {
            metrics.strengths.push_back("Excellent content coverage");
        }

        if (metrics.coherence_score >= 0.9f) {
            metrics.strengths.push_back("Very coherent and well-structured");
        }

        if (metrics.factuality_score >= 0.95f) {
            metrics.strengths.push_back("Highly factually accurate");
        }

        if (metrics.conciseness_score >= 0.9f) {
            metrics.strengths.push_back("Excellently concise");
        }

        if (metrics.fluency_score >= 0.9f) {
            metrics.strengths.push_back("Very fluent and readable");
        }

        if (metrics.semantic_similarity >= 0.85f) {
            metrics.strengths.push_back("Strong semantic alignment with source");
        }
    }

    std::string GenerateRecommendations(const QualityMetrics& metrics) {
        std::stringstream recommendations;

        if (metrics.coverage_score < 0.7f) {
            recommendations << "• Include more key information from the source\n";
        }

        if (metrics.coherence_score < 0.7f) {
            recommendations << "• Improve logical flow with better transitions\n";
        }

        if (metrics.conciseness_score < 0.7f) {
            recommendations << "• Remove redundant information and be more concise\n";
        }

        if (metrics.factuality_score < 0.85f) {
            recommendations << "• Verify facts against source material\n";
        }

        if (metrics.fluency_score < 0.7f) {
            recommendations << "• Improve sentence structure and readability\n";
        }

        if (recommendations.str().empty()) {
            recommendations << "• Summary meets quality standards\n";
        }

        return recommendations.str();
    }

    std::string FormatScore(float score) {
        std::stringstream ss;
        ss << std::fixed << std::setprecision(1) << (score * 100) << "%";
        return ss.str();
    }

    // Helper functions (simplified implementations)
    std::vector<std::string> ExtractKeyInformation(const std::string& text) {
        // Extract key sentences using TextRank or similar
        return {};
    }

    bool IsInformationPresent(const std::string& info, const std::string& text) {
        return text.find(info) != std::string::npos;
    }

    std::vector<std::string> SplitIntoSections(const std::string& text, int n) {
        std::vector<std::string> sections;
        int section_size = text.length() / n;

        for (int i = 0; i < n; i++) {
            int start = i * section_size;
            int end = (i == n - 1) ? text.length() : (i + 1) * section_size;
            sections.push_back(text.substr(start, end - start));
        }

        return sections;
    }

    std::vector<std::string> SplitIntoSentences(const std::string& text) {
        std::vector<std::string> sentences;
        std::regex sentence_regex(R"([.!?]+\s+)");
        std::sregex_token_iterator iter(text.begin(), text.end(), sentence_regex, -1);
        std::sregex_token_iterator end;

        for (; iter != end; ++iter) {
            std::string sentence = *iter;
            if (!sentence.empty()) {
                sentences.push_back(sentence);
            }
        }

        return sentences;
    }

    float ComputeSentenceSimilarity(const std::string& s1, const std::string& s2) {
        // Compute cosine similarity between sentence embeddings
        auto emb1 = GetSingleSentenceEmbedding(s1);
        auto emb2 = GetSingleSentenceEmbedding(s2);
        return ComputeCosineSimilarity(emb1, emb2);
    }

    float ComputeSentenceRelevance(const std::string& sentence, const std::string& source) {
        // Compute relevance based on keyword overlap and semantic similarity
        return 0.8f;  // Simplified
    }

    float ComputeSentencePairCoherence(const std::string& s1, const std::string& s2) {
        // Check for coherence indicators
        return 0.85f;  // Simplified
    }

    float CheckLogicalConnectors(const std::string& text) {
        std::vector<std::string> connectors = {
            "therefore", "however", "moreover", "furthermore",
            "additionally", "consequently", "nevertheless"
        };

        int count = 0;
        for (const auto& connector : connectors) {
            if (text.find(connector) != std::string::npos) count++;
        }

        return std::min(1.0f, count / 5.0f);
    }

    float ComputePerplexity(const std::string& text) {
        // Would use language model to compute perplexity
        return 20.0f;  // Simplified
    }

    float ComputeReadabilityScore(const std::string& text) {
        // Flesch-Kincaid or similar readability metric
        return 0.8f;  // Simplified
    }

    float ComputeSentenceVariety(const std::string& text) {
        auto sentences = SplitIntoSentences(text);

        // Check for varied sentence lengths
        std::vector<int> lengths;
        for (const auto& s : sentences) {
            lengths.push_back(s.length());
        }

        // Calculate standard deviation
        if (lengths.size() < 2) return 1.0f;

        float mean = 0.0f;
        for (int len : lengths) mean += len;
        mean /= lengths.size();

        float variance = 0.0f;
        for (int len : lengths) {
            variance += (len - mean) * (len - mean);
        }
        variance /= lengths.size();

        float std_dev = std::sqrt(variance);
        return std::min(1.0f, std_dev / mean);
    }

    int CountGrammarErrors(const std::string& text) {
        // Would use grammar checker
        return 0;  // Simplified
    }

    bool IsProperSentence(const std::string& sentence) {
        // Check for subject-verb structure
        return !sentence.empty() && std::isupper(sentence[0]);
    }

    std::vector<std::string> ExtractFacts(const std::string& text) {
        // Extract factual statements
        return {};
    }

    bool IsFactSupported(const std::string& fact, const std::vector<std::string>& source_facts) {
        // Check if fact is supported by source
        return true;
    }

    std::vector<std::string> ExtractEntities(const std::string& text) {
        // Named entity extraction
        return {};
    }

    bool IsEntityInSource(const std::string& entity, const std::vector<std::string>& source_entities) {
        return std::find(source_entities.begin(), source_entities.end(), entity) != source_entities.end();
    }

    std::vector<std::string> ExtractClaims(const std::string& text) {
        // Extract claims and assertions
        return {};
    }

    bool IsClaimSupported(const std::string& claim, const std::string& source) {
        // Check if claim is supported by source
        return true;
    }

    bool AreContradictory(const std::string& s1, const std::string& s2) {
        // Check for contradictions
        return false;
    }

    float DetectVerbosity(const std::string& text) {
        // Detect verbose phrases
        std::vector<std::string> verbose_phrases = {
            "in order to", "due to the fact that", "at this point in time",
            "in the event that", "for the purpose of"
        };

        int count = 0;
        for (const auto& phrase : verbose_phrases) {
            if (text.find(phrase) != std::string::npos) count++;
        }

        return count / 10.0f;  // Normalize
    }

    std::vector<std::string> TokenizeWords(const std::string& text) {
        std::vector<std::string> words;
        std::istringstream iss(text);
        std::string word;

        while (iss >> word) {
            words.push_back(word);
        }

        return words;
    }

    bool IsContentWord(const std::string& word) {
        std::unordered_set<std::string> stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "and", "or", "but", "if", "then", "this", "that", "these"
        };

        return stop_words.find(word) == stop_words.end() && word.length() > 2;
    }

    float ComputeTextEntropy(const std::string& text) {
        // Shannon entropy of text
        std::unordered_map<char, int> freq;
        for (char c : text) {
            freq[c]++;
        }

        float entropy = 0.0f;
        int total = text.length();

        for (const auto& [c, count] : freq) {
            float p = float(count) / total;
            entropy -= p * std::log2(p);
        }

        return entropy;
    }

    std::vector<std::vector<float>> GetSentenceEmbeddings(const std::string& text) {
        // Get cached or compute embeddings
        if (embedding_cache_.count(text)) {
            // Return cached embeddings reshaped
        }

        // Compute new embeddings
        auto sentences = SplitIntoSentences(text);
        std::vector<std::vector<float>> embeddings;

        for (const auto& sentence : sentences) {
            embeddings.push_back(GetSingleSentenceEmbedding(sentence));
        }

        return embeddings;
    }

    std::vector<float> GetSingleSentenceEmbedding(const std::string& sentence) {
        // Would use sentence transformer model
        return std::vector<float>(768, 0.1f);  // Simplified
    }

    std::vector<float> ComputeMeanPooling(const std::vector<std::vector<float>>& embeddings) {
        if (embeddings.empty()) return {};

        std::vector<float> mean(embeddings[0].size(), 0.0f);

        for (const auto& emb : embeddings) {
            for (size_t i = 0; i < emb.size(); i++) {
                mean[i] += emb[i];
            }
        }

        for (float& val : mean) {
            val /= embeddings.size();
        }

        return mean;
    }

    float ComputeCosineSimilarity(const std::vector<float>& v1, const std::vector<float>& v2) {
        if (v1.size() != v2.size() || v1.empty()) return 0.0f;

        float dot = 0.0f, norm1 = 0.0f, norm2 = 0.0f;

        for (size_t i = 0; i < v1.size(); i++) {
            dot += v1[i] * v2[i];
            norm1 += v1[i] * v1[i];
            norm2 += v2[i] * v2[i];
        }

        return dot / (std::sqrt(norm1) * std::sqrt(norm2) + 1e-8f);
    }
};

// Automatic summary improvement based on quality feedback
class SummaryImprover {
private:
    std::unique_ptr<SummaryQualityValidator> validator_;

public:
    SummaryImprover() {
        validator_ = std::make_unique<SummaryQualityValidator>();
    }

    std::string ImproveSummary(const std::string& initial_summary,
                              const std::string& source_text,
                              int max_iterations = 3) {
        std::string current_summary = initial_summary;

        for (int iter = 0; iter < max_iterations; iter++) {
            // Validate current summary
            auto metrics = validator_->ValidateSummary(current_summary, source_text);

            // Check if quality is acceptable
            if (validator_->PassesQualityThreshold(metrics)) {
                break;
            }

            // Generate improvement prompt based on issues
            std::string improvement_prompt = GenerateImprovementPrompt(
                current_summary, metrics, source_text
            );

            // Generate improved summary
            current_summary = GenerateImprovedSummary(improvement_prompt);
        }

        return current_summary;
    }

private:
    std::string GenerateImprovementPrompt(const std::string& summary,
                                         const SummaryQualityValidator::QualityMetrics& metrics,
                                         const std::string& source) {
        std::stringstream prompt;

        prompt << "Improve the following summary based on these issues:\n";

        for (const auto& issue : metrics.issues) {
            prompt << "- " << issue << "\n";
        }

        prompt << "\nOriginal summary:\n" << summary << "\n";
        prompt << "\nSource text:\n" << source << "\n";
        prompt << "\nProvide an improved summary that addresses these issues.";

        return prompt.str();
    }

    std::string GenerateImprovedSummary(const std::string& prompt) {
        // Call LLM with improvement prompt
        return "Improved summary";  // Placeholder
    }
};

// Global instances
static std::unique_ptr<SummaryQualityValidator> g_quality_validator;
static std::unique_ptr<SummaryImprover> g_summary_improver;

void InitializeQualityValidation() {
    g_quality_validator = std::make_unique<SummaryQualityValidator>();
    g_summary_improver = std::make_unique<SummaryImprover>();
}

SummaryQualityValidator::QualityMetrics ValidateSummaryQuality(
    const std::string& summary,
    const std::string& source_text) {

    if (!g_quality_validator) {
        InitializeQualityValidation();
    }

    return g_quality_validator->ValidateSummary(summary, source_text);
}

std::string ImproveSummaryQuality(const std::string& summary,
                                 const std::string& source_text) {
    if (!g_summary_improver) {
        InitializeQualityValidation();
    }

    return g_summary_improver->ImproveSummary(summary, source_text);
}

} // namespace summarization