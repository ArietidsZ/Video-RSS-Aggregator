#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <algorithm>
#include <regex>
#include <json/json.h>

namespace summarization {

class ContextAwareSummarizer {
public:
    enum class VideoContentType {
        Educational,        // Lectures, tutorials, courses
        News,              // News reports, current events
        Entertainment,     // Movies, shows, comedy
        Documentary,       // Documentaries, investigations
        Interview,         // Interviews, podcasts, talks
        Product_Review,    // Product reviews, unboxings
        Gaming,           // Gaming content, walkthroughs
        Sports,           // Sports highlights, analysis
        Tech_Talk,        // Technical presentations, demos
        Cooking,          // Cooking shows, recipes
        Travel,           // Travel vlogs, destinations
        Music,            // Music videos, concerts
        Financial,        // Financial analysis, market updates
        Medical,          // Medical content, health topics
        Conference,       // Conference talks, panels
        Vlog,            // Personal vlogs, daily life
        Tutorial,         // How-to guides, DIY
        Commentary,       // Commentary, reactions
        Live_Stream,      // Live streaming content
        Unknown
    };

    struct ContentContext {
        VideoContentType type;
        std::vector<std::string> key_topics;
        std::vector<std::string> entities;
        std::string primary_language;
        float technical_level;  // 0.0 (general) to 1.0 (expert)
        float formality_level;  // 0.0 (casual) to 1.0 (formal)
        int estimated_duration_minutes;
        bool has_visual_content;
        bool has_code_examples;
        bool has_data_visualization;
        std::vector<std::string> target_audience;
        std::unordered_map<std::string, float> topic_importance;
    };

    struct SummarizationStrategy {
        std::string focus_areas;
        int ideal_summary_length;
        std::string tone;
        std::vector<std::string> must_include_elements;
        std::vector<std::string> extraction_priorities;
        bool include_timestamps;
        bool extract_action_items;
        bool preserve_technical_details;
        std::string output_format;  // paragraph, bullet, structured
    };

private:
    // Content type classifiers
    std::unordered_map<VideoContentType, std::vector<std::string>> content_keywords_;
    std::unordered_map<VideoContentType, SummarizationStrategy> strategies_;

    // Model for content classification
    std::unique_ptr<class ContentClassifier> classifier_;

    // Template generators
    std::unordered_map<VideoContentType, std::function<std::string(const ContentContext&)>> template_generators_;

public:
    ContextAwareSummarizer() {
        InitializeContentKeywords();
        InitializeSummarizationStrategies();
        InitializeTemplateGenerators();
    }

    std::string GenerateContextAwareSummary(const std::string& transcript,
                                           const ContentContext& context) {
        // Select strategy based on content type
        auto strategy = GetStrategy(context.type);

        // Build context-aware prompt
        std::string prompt = BuildContextAwarePrompt(transcript, context, strategy);

        // Generate summary with appropriate model
        std::string summary = GenerateSummaryWithContext(prompt, context, strategy);

        // Post-process based on content type
        return PostProcessSummary(summary, context, strategy);
    }

    ContentContext AnalyzeContent(const std::string& transcript,
                                 const Json::Value& metadata = {}) {
        ContentContext context;

        // Detect content type
        context.type = DetectContentType(transcript, metadata);

        // Extract key topics
        context.key_topics = ExtractKeyTopics(transcript);

        // Identify entities
        context.entities = ExtractEntities(transcript);

        // Analyze technical level
        context.technical_level = AnalyzeTechnicalLevel(transcript);

        // Analyze formality
        context.formality_level = AnalyzeFormality(transcript);

        // Extract metadata
        if (!metadata.empty()) {
            context.estimated_duration_minutes = metadata.get("duration", 0).asInt() / 60;
            context.primary_language = metadata.get("language", "en").asString();
            context.has_visual_content = metadata.get("has_video", true).asBool();
        }

        // Detect specific content features
        context.has_code_examples = DetectCodeExamples(transcript);
        context.has_data_visualization = DetectDataVisualization(transcript);

        // Identify target audience
        context.target_audience = IdentifyTargetAudience(transcript, context);

        // Calculate topic importance
        context.topic_importance = CalculateTopicImportance(transcript, context.key_topics);

        return context;
    }

private:
    void InitializeContentKeywords() {
        content_keywords_[VideoContentType::Educational] = {
            "learn", "lesson", "chapter", "example", "exercise", "homework",
            "quiz", "test", "study", "course", "lecture", "professor",
            "student", "curriculum", "syllabus", "assignment"
        };

        content_keywords_[VideoContentType::News] = {
            "breaking", "report", "update", "announced", "confirmed",
            "sources", "correspondent", "developing", "statement",
            "official", "investigation", "incident", "happened"
        };

        content_keywords_[VideoContentType::Tech_Talk] = {
            "algorithm", "framework", "api", "implementation", "architecture",
            "performance", "optimization", "deployment", "scalability",
            "database", "backend", "frontend", "cloud", "microservices"
        };

        content_keywords_[VideoContentType::Product_Review] = {
            "review", "unboxing", "pros", "cons", "features", "price",
            "comparison", "rating", "quality", "performance", "value",
            "recommend", "verdict", "specifications", "warranty"
        };

        content_keywords_[VideoContentType::Interview] = {
            "question", "answer", "asked", "responded", "opinion",
            "perspective", "experience", "career", "journey", "advice",
            "thoughts", "believe", "feel", "guest", "host"
        };

        content_keywords_[VideoContentType::Documentary] = {
            "history", "investigation", "evidence", "discovery", "reveals",
            "explores", "uncovers", "documented", "archive", "witness",
            "expert", "analysis", "timeline", "background", "context"
        };

        content_keywords_[VideoContentType::Tutorial] = {
            "how to", "step by step", "guide", "instructions", "first",
            "next", "then", "finally", "make sure", "tip", "trick",
            "beginner", "advanced", "follow along", "demonstration"
        };

        content_keywords_[VideoContentType::Gaming] = {
            "game", "gameplay", "level", "boss", "quest", "mission",
            "character", "weapon", "strategy", "walkthrough", "tips",
            "multiplayer", "achievement", "speedrun", "easter egg"
        };

        content_keywords_[VideoContentType::Cooking] = {
            "recipe", "ingredients", "cook", "bake", "prepare", "mix",
            "stir", "heat", "oven", "minutes", "tablespoon", "teaspoon",
            "serve", "garnish", "season", "taste", "delicious"
        };

        content_keywords_[VideoContentType::Financial] = {
            "market", "stock", "investment", "portfolio", "earnings",
            "revenue", "profit", "loss", "trading", "analysis", "forecast",
            "economy", "inflation", "interest", "dividend", "volatility"
        };
    }

    void InitializeSummarizationStrategies() {
        // Educational content strategy
        strategies_[VideoContentType::Educational] = {
            .focus_areas = "key concepts, learning objectives, examples, takeaways",
            .ideal_summary_length = 300,
            .tone = "instructional",
            .must_include_elements = {"main topics", "key concepts", "examples", "conclusions"},
            .extraction_priorities = {"definitions", "formulas", "principles", "methodologies"},
            .include_timestamps = true,
            .extract_action_items = true,
            .preserve_technical_details = true,
            .output_format = "structured"
        };

        // News content strategy
        strategies_[VideoContentType::News] = {
            .focus_areas = "5 W's (who, what, when, where, why), impact, quotes",
            .ideal_summary_length = 200,
            .tone = "objective",
            .must_include_elements = {"headline", "key facts", "sources", "impact"},
            .extraction_priorities = {"facts", "quotes", "statistics", "timeline"},
            .include_timestamps = false,
            .extract_action_items = false,
            .preserve_technical_details = false,
            .output_format = "paragraph"
        };

        // Tech Talk strategy
        strategies_[VideoContentType::Tech_Talk] = {
            .focus_areas = "technical concepts, architecture, implementation, best practices",
            .ideal_summary_length = 400,
            .tone = "technical",
            .must_include_elements = {"problem", "solution", "architecture", "benefits"},
            .extraction_priorities = {"algorithms", "design patterns", "performance", "trade-offs"},
            .include_timestamps = true,
            .extract_action_items = true,
            .preserve_technical_details = true,
            .output_format = "structured"
        };

        // Product Review strategy
        strategies_[VideoContentType::Product_Review] = {
            .focus_areas = "features, pros/cons, performance, value, verdict",
            .ideal_summary_length = 250,
            .tone = "analytical",
            .must_include_elements = {"product name", "key features", "pros", "cons", "rating"},
            .extraction_priorities = {"specifications", "performance metrics", "price", "alternatives"},
            .include_timestamps = false,
            .extract_action_items = false,
            .preserve_technical_details = false,
            .output_format = "bullet"
        };

        // Interview strategy
        strategies_[VideoContentType::Interview] = {
            .focus_areas = "key insights, quotes, perspectives, revelations",
            .ideal_summary_length = 350,
            .tone = "conversational",
            .must_include_elements = {"participants", "main topics", "key quotes", "insights"},
            .extraction_priorities = {"opinions", "experiences", "advice", "anecdotes"},
            .include_timestamps = true,
            .extract_action_items = false,
            .preserve_technical_details = false,
            .output_format = "structured"
        };

        // Documentary strategy
        strategies_[VideoContentType::Documentary] = {
            .focus_areas = "narrative, evidence, findings, implications",
            .ideal_summary_length = 400,
            .tone = "informative",
            .must_include_elements = {"subject", "findings", "evidence", "conclusions"},
            .extraction_priorities = {"facts", "timeline", "expert opinions", "revelations"},
            .include_timestamps = true,
            .extract_action_items = false,
            .preserve_technical_details = false,
            .output_format = "paragraph"
        };

        // Tutorial strategy
        strategies_[VideoContentType::Tutorial] = {
            .focus_areas = "steps, requirements, tips, common mistakes",
            .ideal_summary_length = 300,
            .tone = "instructional",
            .must_include_elements = {"objective", "requirements", "steps", "tips"},
            .extraction_priorities = {"prerequisites", "materials", "procedure", "troubleshooting"},
            .include_timestamps = true,
            .extract_action_items = true,
            .preserve_technical_details = true,
            .output_format = "bullet"
        };

        // Gaming strategy
        strategies_[VideoContentType::Gaming] = {
            .focus_areas = "gameplay, strategies, tips, achievements",
            .ideal_summary_length = 250,
            .tone = "casual",
            .must_include_elements = {"game", "objectives", "strategies", "outcomes"},
            .extraction_priorities = {"tips", "strategies", "secrets", "achievements"},
            .include_timestamps = true,
            .extract_action_items = false,
            .preserve_technical_details = false,
            .output_format = "bullet"
        };

        // Financial strategy
        strategies_[VideoContentType::Financial] = {
            .focus_areas = "market analysis, data, predictions, recommendations",
            .ideal_summary_length = 350,
            .tone = "analytical",
            .must_include_elements = {"market conditions", "data points", "analysis", "outlook"},
            .extraction_priorities = {"numbers", "trends", "risks", "opportunities"},
            .include_timestamps = false,
            .extract_action_items = true,
            .preserve_technical_details = true,
            .output_format = "structured"
        };
    }

    void InitializeTemplateGenerators() {
        // Educational content template
        template_generators_[VideoContentType::Educational] = [](const ContentContext& ctx) {
            return R"(
## Learning Summary

**Topic**: [Main topic]
**Level**: [Beginner/Intermediate/Advanced]
**Duration**: [X minutes]

### Key Concepts
- [Concept 1]: [Explanation]
- [Concept 2]: [Explanation]

### Main Points
1. [Point 1]
2. [Point 2]

### Examples & Applications
- [Example 1]
- [Example 2]

### Key Takeaways
- [Takeaway 1]
- [Takeaway 2]

### Action Items
- [ ] [Action 1]
- [ ] [Action 2]
)";
        };

        // News content template
        template_generators_[VideoContentType::News] = [](const ContentContext& ctx) {
            return R"(
**Headline**: [Main story]

[Lead paragraph with who, what, when, where, why]

**Key Facts**:
- [Fact 1]
- [Fact 2]

**Impact**: [How this affects viewers/society]

**What's Next**: [Future developments]
)";
        };

        // Tech Talk template
        template_generators_[VideoContentType::Tech_Talk] = [](const ContentContext& ctx) {
            return R"(
## Technical Overview

**Technology**: [Name]
**Problem Solved**: [Description]

### Architecture & Design
[Technical architecture description]

### Implementation Details
- **Language/Framework**: [Details]
- **Key Components**: [List]
- **Performance**: [Metrics]

### Code Examples
```
[Code snippet if applicable]
```

### Best Practices
1. [Practice 1]
2. [Practice 2]

### Resources
- [Resource 1]
- [Resource 2]
)";
        };
    }

    VideoContentType DetectContentType(const std::string& transcript,
                                      const Json::Value& metadata) {
        // Score each content type based on keywords
        std::unordered_map<VideoContentType, float> scores;

        for (const auto& [type, keywords] : content_keywords_) {
            float score = 0.0f;
            std::string lower_transcript = transcript;
            std::transform(lower_transcript.begin(), lower_transcript.end(),
                         lower_transcript.begin(), ::tolower);

            for (const auto& keyword : keywords) {
                size_t count = 0;
                size_t pos = 0;
                while ((pos = lower_transcript.find(keyword, pos)) != std::string::npos) {
                    count++;
                    pos += keyword.length();
                }
                score += count * (1.0f / keywords.size());
            }

            scores[type] = score;
        }

        // Check metadata hints
        if (!metadata.empty() && metadata.isMember("category")) {
            std::string category = metadata["category"].asString();
            if (category == "Education") scores[VideoContentType::Educational] *= 2;
            if (category == "News & Politics") scores[VideoContentType::News] *= 2;
            if (category == "Science & Technology") scores[VideoContentType::Tech_Talk] *= 2;
            if (category == "Gaming") scores[VideoContentType::Gaming] *= 2;
        }

        // Find highest scoring type
        auto max_it = std::max_element(scores.begin(), scores.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });

        if (max_it != scores.end() && max_it->second > 1.0f) {
            return max_it->first;
        }

        return VideoContentType::Unknown;
    }

    std::vector<std::string> ExtractKeyTopics(const std::string& transcript) {
        std::vector<std::string> topics;

        // Simple keyword extraction (would use TF-IDF or TextRank in production)
        std::unordered_map<std::string, int> word_freq;
        std::istringstream iss(transcript);
        std::string word;

        while (iss >> word) {
            // Clean word
            word.erase(std::remove_if(word.begin(), word.end(), ::ispunct), word.end());
            std::transform(word.begin(), word.end(), word.begin(), ::tolower);

            if (word.length() > 4 && IsValidTopic(word)) {
                word_freq[word]++;
            }
        }

        // Get top topics
        std::vector<std::pair<std::string, int>> sorted_words(word_freq.begin(), word_freq.end());
        std::sort(sorted_words.begin(), sorted_words.end(),
                 [](const auto& a, const auto& b) { return a.second > b.second; });

        for (size_t i = 0; i < std::min(size_t(10), sorted_words.size()); i++) {
            topics.push_back(sorted_words[i].first);
        }

        return topics;
    }

    std::vector<std::string> ExtractEntities(const std::string& transcript) {
        std::vector<std::string> entities;

        // Simple named entity recognition (would use NER model in production)
        std::regex entity_regex(R"(\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b)");
        std::sregex_iterator iter(transcript.begin(), transcript.end(), entity_regex);
        std::sregex_iterator end;

        std::unordered_set<std::string> unique_entities;
        for (; iter != end; ++iter) {
            unique_entities.insert(iter->str());
        }

        entities.assign(unique_entities.begin(), unique_entities.end());
        return entities;
    }

    float AnalyzeTechnicalLevel(const std::string& transcript) {
        // Count technical terms and jargon
        std::vector<std::string> technical_terms = {
            "algorithm", "architecture", "framework", "api", "database",
            "optimization", "implementation", "protocol", "encryption",
            "compilation", "runtime", "latency", "throughput", "scalability"
        };

        int technical_count = 0;
        std::string lower_transcript = transcript;
        std::transform(lower_transcript.begin(), lower_transcript.end(),
                     lower_transcript.begin(), ::tolower);

        for (const auto& term : technical_terms) {
            if (lower_transcript.find(term) != std::string::npos) {
                technical_count++;
            }
        }

        // Normalize to 0-1 scale
        return std::min(1.0f, technical_count / 10.0f);
    }

    float AnalyzeFormality(const std::string& transcript) {
        // Analyze language formality
        std::vector<std::string> formal_indicators = {
            "therefore", "furthermore", "moreover", "consequently",
            "nevertheless", "accordingly", "respectively", "whereas"
        };

        std::vector<std::string> informal_indicators = {
            "gonna", "wanna", "yeah", "stuff", "thing", "like",
            "you know", "I mean", "kind of", "sort of"
        };

        int formal_count = 0, informal_count = 0;
        std::string lower_transcript = transcript;
        std::transform(lower_transcript.begin(), lower_transcript.end(),
                     lower_transcript.begin(), ::tolower);

        for (const auto& word : formal_indicators) {
            if (lower_transcript.find(word) != std::string::npos) formal_count++;
        }

        for (const auto& word : informal_indicators) {
            if (lower_transcript.find(word) != std::string::npos) informal_count++;
        }

        if (formal_count + informal_count == 0) return 0.5f;
        return float(formal_count) / (formal_count + informal_count);
    }

    bool DetectCodeExamples(const std::string& transcript) {
        // Detect code patterns
        std::vector<std::string> code_indicators = {
            "function", "class", "import", "return", "if else",
            "for loop", "while loop", "variable", "const", "let",
            "def ", "void ", "public ", "private ", "{}", "[]", "()"
        };

        for (const auto& indicator : code_indicators) {
            if (transcript.find(indicator) != std::string::npos) {
                return true;
            }
        }

        return false;
    }

    bool DetectDataVisualization(const std::string& transcript) {
        std::vector<std::string> viz_indicators = {
            "graph", "chart", "plot", "diagram", "visualization",
            "dashboard", "metrics", "statistics", "data shows"
        };

        for (const auto& indicator : viz_indicators) {
            if (transcript.find(indicator) != std::string::npos) {
                return true;
            }
        }

        return false;
    }

    std::vector<std::string> IdentifyTargetAudience(const std::string& transcript,
                                                   const ContentContext& context) {
        std::vector<std::string> audience;

        if (context.technical_level < 0.3) {
            audience.push_back("general");
            audience.push_back("beginners");
        } else if (context.technical_level < 0.7) {
            audience.push_back("intermediate");
            audience.push_back("practitioners");
        } else {
            audience.push_back("experts");
            audience.push_back("professionals");
        }

        // Check for specific audience mentions
        if (transcript.find("student") != std::string::npos) {
            audience.push_back("students");
        }
        if (transcript.find("developer") != std::string::npos) {
            audience.push_back("developers");
        }
        if (transcript.find("business") != std::string::npos) {
            audience.push_back("business_professionals");
        }

        return audience;
    }

    std::unordered_map<std::string, float> CalculateTopicImportance(
        const std::string& transcript,
        const std::vector<std::string>& topics) {

        std::unordered_map<std::string, float> importance;

        for (const auto& topic : topics) {
            // Calculate based on frequency and position
            int count = 0;
            float position_weight = 0.0f;
            size_t pos = 0;

            while ((pos = transcript.find(topic, pos)) != std::string::npos) {
                count++;
                // Weight earlier mentions higher
                position_weight += 1.0f - (float(pos) / transcript.length());
                pos += topic.length();
            }

            importance[topic] = count * position_weight;
        }

        // Normalize
        float max_importance = 0.0f;
        for (const auto& [topic, imp] : importance) {
            max_importance = std::max(max_importance, imp);
        }

        if (max_importance > 0) {
            for (auto& [topic, imp] : importance) {
                imp /= max_importance;
            }
        }

        return importance;
    }

    SummarizationStrategy GetStrategy(VideoContentType type) {
        if (strategies_.count(type)) {
            return strategies_[type];
        }
        // Return default strategy
        return {
            .focus_areas = "main points, key information",
            .ideal_summary_length = 300,
            .tone = "neutral",
            .must_include_elements = {"main topic", "key points", "conclusion"},
            .extraction_priorities = {"facts", "insights", "conclusions"},
            .include_timestamps = false,
            .extract_action_items = false,
            .preserve_technical_details = false,
            .output_format = "paragraph"
        };
    }

    std::string BuildContextAwarePrompt(const std::string& transcript,
                                       const ContentContext& context,
                                       const SummarizationStrategy& strategy) {
        std::stringstream prompt;

        // System instruction based on content type
        prompt << "You are an expert at summarizing " << GetContentTypeString(context.type)
               << " content. ";

        // Add context-specific instructions
        prompt << "Focus on: " << strategy.focus_areas << ".\n";

        if (context.technical_level > 0.7) {
            prompt << "Preserve technical accuracy and use appropriate terminology.\n";
        } else if (context.technical_level < 0.3) {
            prompt << "Use simple, accessible language avoiding jargon.\n";
        }

        // Add tone instruction
        prompt << "Tone: " << strategy.tone << "\n";

        // Add format instruction
        if (strategy.output_format == "bullet") {
            prompt << "Format as bullet points.\n";
        } else if (strategy.output_format == "structured") {
            prompt << "Use structured sections with headers.\n";
        }

        // Must include elements
        prompt << "Ensure you include: ";
        for (const auto& element : strategy.must_include_elements) {
            prompt << element << ", ";
        }
        prompt << "\n";

        // Add the transcript
        prompt << "\nTranscript to summarize:\n" << transcript << "\n\n";

        // Target length
        prompt << "Target length: approximately " << strategy.ideal_summary_length << " words.\n";

        return prompt.str();
    }

    std::string GenerateSummaryWithContext(const std::string& prompt,
                                          const ContentContext& context,
                                          const SummarizationStrategy& strategy) {
        // This would call the actual LLM with the context-aware prompt
        // For now, return a placeholder
        return "Context-aware summary based on " + GetContentTypeString(context.type) + " content.";
    }

    std::string PostProcessSummary(const std::string& summary,
                                  const ContentContext& context,
                                  const SummarizationStrategy& strategy) {
        std::string processed = summary;

        // Add metadata header if needed
        if (strategy.output_format == "structured") {
            std::stringstream final_summary;
            final_summary << "**Content Type**: " << GetContentTypeString(context.type) << "\n";
            final_summary << "**Duration**: " << context.estimated_duration_minutes << " minutes\n";

            if (!context.key_topics.empty()) {
                final_summary << "**Key Topics**: ";
                for (const auto& topic : context.key_topics) {
                    final_summary << topic << ", ";
                }
                final_summary << "\n";
            }

            final_summary << "\n" << processed;
            processed = final_summary.str();
        }

        // Extract and format action items if needed
        if (strategy.extract_action_items) {
            processed += "\n\n### Action Items\n";
            auto action_items = ExtractActionItems(summary);
            for (const auto& item : action_items) {
                processed += "- [ ] " + item + "\n";
            }
        }

        return processed;
    }

    std::vector<std::string> ExtractActionItems(const std::string& text) {
        std::vector<std::string> action_items;

        // Look for action-oriented phrases
        std::regex action_regex(R"((?:should|need to|must|have to|required to|recommended to)\s+([^.!?]+))");
        std::sregex_iterator iter(text.begin(), text.end(), action_regex);
        std::sregex_iterator end;

        for (; iter != end; ++iter) {
            action_items.push_back(iter->str(1));
        }

        return action_items;
    }

    bool IsValidTopic(const std::string& word) {
        // Filter out common stop words
        std::unordered_set<std::string> stop_words = {
            "the", "and", "for", "with", "from", "this", "that",
            "have", "been", "will", "would", "could", "should"
        };

        return stop_words.find(word) == stop_words.end();
    }

    std::string GetContentTypeString(VideoContentType type) {
        switch (type) {
            case VideoContentType::Educational: return "educational";
            case VideoContentType::News: return "news";
            case VideoContentType::Entertainment: return "entertainment";
            case VideoContentType::Documentary: return "documentary";
            case VideoContentType::Interview: return "interview";
            case VideoContentType::Product_Review: return "product review";
            case VideoContentType::Gaming: return "gaming";
            case VideoContentType::Sports: return "sports";
            case VideoContentType::Tech_Talk: return "technical";
            case VideoContentType::Cooking: return "cooking";
            case VideoContentType::Travel: return "travel";
            case VideoContentType::Music: return "music";
            case VideoContentType::Financial: return "financial";
            case VideoContentType::Medical: return "medical";
            case VideoContentType::Conference: return "conference";
            case VideoContentType::Vlog: return "vlog";
            case VideoContentType::Tutorial: return "tutorial";
            case VideoContentType::Commentary: return "commentary";
            case VideoContentType::Live_Stream: return "live stream";
            default: return "general";
        }
    }
};

// Global context-aware summarizer
static std::unique_ptr<ContextAwareSummarizer> g_context_summarizer;

void InitializeContextAwareSummarization() {
    g_context_summarizer = std::make_unique<ContextAwareSummarizer>();
}

std::string GenerateContextAwareSummary(const std::string& transcript,
                                       const Json::Value& metadata) {
    if (!g_context_summarizer) {
        InitializeContextAwareSummarization();
    }

    auto context = g_context_summarizer->AnalyzeContent(transcript, metadata);
    return g_context_summarizer->GenerateContextAwareSummary(transcript, context);
}

} // namespace summarization