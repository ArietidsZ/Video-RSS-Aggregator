#include <string>
#include <unordered_map>
#include <vector>
#include <regex>
#include <sstream>
#include <algorithm>
#include <chrono>
#include <json/json.h>

namespace summarization {

class PromptTemplateEngine {
public:
    enum class ContentType {
        TechTalk,
        Tutorial,
        NewsReport,
        Documentary,
        Interview,
        Podcast,
        Lecture,
        ProductReview,
        Entertainment,
        Financial,
        Medical,
        Legal,
        Gaming,
        Cooking,
        Sports,
        Travel,
        Conference,
        Webinar,
        ResearchPresentation,
        General
    };

    enum class SummaryStyle {
        Executive,      // Brief executive summary
        Technical,      // Detailed technical breakdown
        Educational,    // Learning-focused with key concepts
        Bullet,        // Bullet points
        Timeline,      // Chronological events
        Highlights,    // Key highlights only
        Comprehensive, // Detailed coverage
        Abstract,      // Academic abstract style
        Actionable,    // Focus on actionable items
        Narrative,     // Story-like narrative
        Q_And_A,       // Question and answer format
        KeyTakeaways   // Main takeaways only
    };

    struct PromptConfig {
        ContentType content_type;
        SummaryStyle summary_style;
        int target_length_words;
        std::string language;
        bool include_timestamps;
        bool extract_quotes;
        bool identify_speakers;
        bool extract_keywords;
        bool generate_title;
        bool create_chapters;
        float technical_level;  // 0.0 (beginner) to 1.0 (expert)
        std::vector<std::string> focus_topics;
        std::map<std::string, std::string> custom_instructions;
    };

private:
    struct PromptTemplate {
        std::string system_prompt;
        std::string user_prompt_template;
        std::string formatting_instructions;
        std::vector<std::string> example_outputs;
        std::map<std::string, std::string> variable_mappings;
    };

    std::unordered_map<ContentType, std::unordered_map<SummaryStyle, PromptTemplate>> templates_;
    std::unordered_map<std::string, std::string> language_instructions_;

public:
    PromptTemplateEngine() {
        InitializeTemplates();
        InitializeLanguageInstructions();
    }

    std::string GeneratePrompt(const std::string& transcript,
                              const PromptConfig& config) {
        // Select appropriate template
        auto template_ptr = GetTemplate(config.content_type, config.summary_style);

        // Build system prompt
        std::string system_prompt = BuildSystemPrompt(template_ptr, config);

        // Build user prompt
        std::string user_prompt = BuildUserPrompt(template_ptr, transcript, config);

        // Combine prompts
        return FormatFinalPrompt(system_prompt, user_prompt, config);
    }

    std::string GenerateChainOfThoughtPrompt(const std::string& transcript,
                                            const PromptConfig& config) {
        std::stringstream prompt;

        prompt << "You are an expert content analyzer. Follow this chain of thought process:\n\n";

        prompt << "Step 1: Content Analysis\n";
        prompt << "- Identify the main topic and subject matter\n";
        prompt << "- Determine the content type and structure\n";
        prompt << "- Note the key speakers or presenters\n\n";

        prompt << "Step 2: Key Information Extraction\n";
        prompt << "- Extract main arguments or points\n";
        prompt << "- Identify supporting evidence or examples\n";
        prompt << "- Note any conclusions or recommendations\n\n";

        prompt << "Step 3: Contextual Understanding\n";
        prompt << "- Consider the target audience\n";
        prompt << "- Identify the purpose or goal\n";
        prompt << "- Note any important context or background\n\n";

        prompt << "Step 4: Summary Generation\n";
        prompt << "Based on the above analysis, create a "
               << GetStyleDescription(config.summary_style)
               << " summary of approximately " << config.target_length_words
               << " words.\n\n";

        prompt << "Content to analyze:\n" << transcript << "\n\n";

        prompt << "Provide your analysis for each step, then generate the final summary.";

        return prompt.str();
    }

    std::string GenerateMultiPassPrompt(const std::string& transcript,
                                       const PromptConfig& config) {
        std::vector<std::string> passes;

        // Pass 1: Initial summary
        passes.push_back(GenerateInitialSummaryPrompt(transcript, config));

        // Pass 2: Refinement
        passes.push_back(GenerateRefinementPrompt(config));

        // Pass 3: Fact checking
        passes.push_back(GenerateFactCheckPrompt());

        // Pass 4: Final polish
        passes.push_back(GenerateFinalPolishPrompt(config));

        return JoinPasses(passes);
    }

private:
    void InitializeTemplates() {
        // Tech Talk Template
        templates_[ContentType::TechTalk][SummaryStyle::Technical] = {
            .system_prompt = R"(You are a technical content expert specializing in technology presentations and talks.
Your task is to create detailed technical summaries that capture complex concepts, architectures, and implementations.
Focus on technical accuracy, proper terminology, and preserving important technical details.)",

            .user_prompt_template = R"(Summarize this technical presentation with focus on:
1. Core technical concepts and innovations
2. Architecture and implementation details
3. Performance metrics and benchmarks
4. Technical challenges and solutions
5. Code examples or algorithms mentioned
6. Tools, frameworks, and technologies used
7. Best practices and recommendations

Transcript: {transcript}

Technical Summary:)",

            .formatting_instructions = R"(Format your response with clear sections:
- **Overview**: Brief introduction to the technology
- **Key Concepts**: Main technical concepts explained
- **Implementation**: How it works technically
- **Performance**: Metrics, benchmarks, or improvements
- **Use Cases**: Practical applications
- **Takeaways**: Key technical lessons)",

            .example_outputs = {
                "Example: A talk about distributed systems would include consensus algorithms, CAP theorem, partition tolerance strategies..."
            }
        };

        // Tutorial Template
        templates_[ContentType::Tutorial][SummaryStyle::Educational] = {
            .system_prompt = R"(You are an educational content specialist who excels at breaking down tutorials into clear, learnable segments.
Focus on step-by-step processes, prerequisites, and learning outcomes.)",

            .user_prompt_template = R"(Create an educational summary of this tutorial that includes:
1. Learning objectives and outcomes
2. Prerequisites and required knowledge
3. Step-by-step process overview
4. Key concepts and terminology explained
5. Common pitfalls and how to avoid them
6. Practice exercises or examples
7. Additional resources mentioned

Tutorial Content: {transcript}

Educational Summary:)",

            .formatting_instructions = R"(Structure as a learning guide:
- **What You'll Learn**: Clear learning objectives
- **Prerequisites**: What you need to know first
- **Steps Overview**: Numbered list of main steps
- **Key Concepts**: Important terms and ideas
- **Pro Tips**: Expert advice and shortcuts
- **Next Steps**: Where to go from here)"
        };

        // News Report Template
        templates_[ContentType::NewsReport][SummaryStyle::Executive] = {
            .system_prompt = R"(You are a news analyst who creates concise executive summaries of news reports.
Focus on the 5 W's (Who, What, When, Where, Why) and the impact of the news.)",

            .user_prompt_template = R"(Provide an executive summary of this news report covering:
- What happened (main event)
- Who is involved (key people/organizations)
- When and where it occurred
- Why it matters (impact and implications)
- Key quotes from officials or experts
- Background context if relevant
- What happens next

News Content: {transcript}

Executive Summary:)",

            .formatting_instructions = R"(Use inverted pyramid style:
Lead with most important information first.
Follow with supporting details.
End with background or less critical information.)"
        };

        // Interview Template
        templates_[ContentType::Interview][SummaryStyle::Q_And_A] = {
            .system_prompt = R"(You are skilled at summarizing interviews while preserving the conversational insights and key exchanges.
Maintain the Q&A structure while condensing to essential information.)",

            .user_prompt_template = R"(Summarize this interview in Q&A format:
- Identify the interviewer and interviewee(s)
- Extract the most important questions asked
- Provide condensed versions of the answers
- Highlight any surprising revelations
- Note any memorable quotes
- Capture the overall tone and key takeaways

Interview Transcript: {transcript}

Q&A Summary:)",

            .formatting_instructions = R"(Format as:
**Participants**: [Names and roles]
**Key Exchanges**:
Q1: [Question]
A1: [Condensed answer]
...
**Notable Quotes**: [Direct quotes]
**Key Insights**: [Main takeaways])"
        };

        // Research Presentation Template
        templates_[ContentType::ResearchPresentation][SummaryStyle::Abstract] = {
            .system_prompt = R"(You are an academic researcher skilled at creating scholarly abstracts.
Focus on research methodology, findings, and implications.)",

            .user_prompt_template = R"(Create an academic abstract for this research presentation including:
1. Research question or hypothesis
2. Methodology and approach
3. Key findings and results
4. Statistical significance if mentioned
5. Implications for the field
6. Limitations acknowledged
7. Future research directions

Research Content: {transcript}

Abstract:)",

            .formatting_instructions = R"(Follow academic abstract structure:
- **Objective**: Research goals
- **Methods**: How the research was conducted
- **Results**: What was discovered
- **Conclusions**: What it means
- **Keywords**: Relevant search terms)"
        };

        // Product Review Template
        templates_[ContentType::ProductReview][SummaryStyle::Comprehensive] = {
            .system_prompt = R"(You are a product analyst who creates detailed review summaries.
Balance pros and cons while maintaining objectivity.)",

            .user_prompt_template = R"(Summarize this product review covering:
1. Product name and category
2. Key features discussed
3. Pros and advantages
4. Cons and disadvantages
5. Performance metrics or tests
6. Comparison with competitors
7. Price and value assessment
8. Final verdict or rating

Review Content: {transcript}

Comprehensive Review Summary:)",

            .formatting_instructions = R"(Structure as:
- **Product Overview**: What it is
- **Pros**: Bullet list of advantages
- **Cons**: Bullet list of disadvantages
- **Performance**: Test results or metrics
- **Value**: Price vs. features analysis
- **Verdict**: Recommendation and rating)"
        };

        // Conference Template
        templates_[ContentType::Conference][SummaryStyle::Highlights] = {
            .system_prompt = R"(You are a conference correspondent who captures key highlights and announcements.
Focus on major reveals, important discussions, and actionable insights.)",

            .user_prompt_template = R"(Extract highlights from this conference session:
- Major announcements or reveals
- Key speakers and their main points
- Important statistics or data presented
- Upcoming releases or timelines
- Industry trends discussed
- Audience Q&A highlights

Conference Content: {transcript}

Conference Highlights:)",

            .formatting_instructions = R"(Format as news-style bullets:
• [Speaker]: [Key point]
• NEW: [Announcement]
• DATA: [Important statistic]
• COMING: [Future release])"
        };

        // Add more templates for other content types...
    }

    void InitializeLanguageInstructions() {
        language_instructions_["en"] = "Provide the summary in clear, professional English.";
        language_instructions_["zh"] = "请用简洁专业的中文提供摘要。";
        language_instructions_["es"] = "Proporcione el resumen en español claro y profesional.";
        language_instructions_["fr"] = "Fournissez le résumé dans un français clair et professionnel.";
        language_instructions_["de"] = "Geben Sie die Zusammenfassung in klarem, professionellem Deutsch an.";
        language_instructions_["ja"] = "明確で専門的な日本語で要約を提供してください。";
        language_instructions_["ko"] = "명확하고 전문적인 한국어로 요약을 제공하세요.";
        language_instructions_["ru"] = "Предоставьте резюме на понятном профессиональном русском языке.";
    }

    PromptTemplate GetTemplate(ContentType type, SummaryStyle style) {
        // Return template if exists, otherwise return general template
        if (templates_.count(type) && templates_[type].count(style)) {
            return templates_[type][style];
        }

        // Fallback to general template
        return GetGeneralTemplate(style);
    }

    PromptTemplate GetGeneralTemplate(SummaryStyle style) {
        PromptTemplate general;

        general.system_prompt = R"(You are an expert content summarizer capable of handling various types of content.
Create summaries that are accurate, well-structured, and tailored to the specified style.)";

        switch (style) {
            case SummaryStyle::Bullet:
                general.user_prompt_template = R"(Create a bullet-point summary of the following content:
{transcript}

Bullet Summary:)";
                break;

            case SummaryStyle::Executive:
                general.user_prompt_template = R"(Create an executive summary of the following content:
{transcript}

Executive Summary:)";
                break;

            default:
                general.user_prompt_template = R"(Summarize the following content:
{transcript}

Summary:)";
        }

        return general;
    }

    std::string BuildSystemPrompt(const PromptTemplate& tmpl,
                                 const PromptConfig& config) {
        std::stringstream prompt;

        prompt << tmpl.system_prompt << "\n\n";

        // Add language instruction
        if (language_instructions_.count(config.language)) {
            prompt << language_instructions_[config.language] << "\n";
        }

        // Add length constraint
        prompt << "Target length: approximately " << config.target_length_words << " words.\n";

        // Add technical level adjustment
        if (config.technical_level < 0.3) {
            prompt << "Use simple, non-technical language suitable for beginners.\n";
        } else if (config.technical_level > 0.7) {
            prompt << "Use technical terminology appropriate for experts.\n";
        }

        // Add custom instructions
        for (const auto& [key, value] : config.custom_instructions) {
            prompt << key << ": " << value << "\n";
        }

        return prompt.str();
    }

    std::string BuildUserPrompt(const PromptTemplate& tmpl,
                               const std::string& transcript,
                               const PromptConfig& config) {
        std::string prompt = tmpl.user_prompt_template;

        // Replace variables
        prompt = std::regex_replace(prompt, std::regex("\\{transcript\\}"), transcript);
        prompt = std::regex_replace(prompt, std::regex("\\{length\\}"),
                                   std::to_string(config.target_length_words));

        // Add additional requirements
        if (config.include_timestamps) {
            prompt += "\nInclude timestamps for key moments.";
        }

        if (config.extract_quotes) {
            prompt += "\nExtract important quotes with attribution.";
        }

        if (config.identify_speakers) {
            prompt += "\nIdentify and label different speakers.";
        }

        if (config.extract_keywords) {
            prompt += "\nList relevant keywords and topics.";
        }

        if (config.generate_title) {
            prompt += "\nSuggest an appropriate title.";
        }

        if (config.create_chapters) {
            prompt += "\nDivide into logical chapters or sections.";
        }

        // Add focus topics
        if (!config.focus_topics.empty()) {
            prompt += "\nPay special attention to these topics: ";
            for (size_t i = 0; i < config.focus_topics.size(); i++) {
                prompt += config.focus_topics[i];
                if (i < config.focus_topics.size() - 1) prompt += ", ";
            }
            prompt += "\n";
        }

        // Add formatting instructions
        if (!tmpl.formatting_instructions.empty()) {
            prompt += "\n" + tmpl.formatting_instructions;
        }

        return prompt;
    }

    std::string FormatFinalPrompt(const std::string& system_prompt,
                                 const std::string& user_prompt,
                                 const PromptConfig& config) {
        Json::Value prompt_json;
        prompt_json["system"] = system_prompt;
        prompt_json["user"] = user_prompt;
        prompt_json["temperature"] = 0.7;
        prompt_json["max_tokens"] = config.target_length_words * 2;  // Approximate

        Json::StreamWriterBuilder writer;
        return Json::writeString(writer, prompt_json);
    }

    std::string GetStyleDescription(SummaryStyle style) {
        switch (style) {
            case SummaryStyle::Executive: return "concise executive";
            case SummaryStyle::Technical: return "detailed technical";
            case SummaryStyle::Educational: return "educational learning-focused";
            case SummaryStyle::Bullet: return "bullet-point";
            case SummaryStyle::Timeline: return "chronological timeline";
            case SummaryStyle::Highlights: return "key highlights";
            case SummaryStyle::Comprehensive: return "comprehensive detailed";
            case SummaryStyle::Abstract: return "academic abstract";
            case SummaryStyle::Actionable: return "actionable items focused";
            case SummaryStyle::Narrative: return "narrative story-like";
            case SummaryStyle::Q_And_A: return "question and answer";
            case SummaryStyle::KeyTakeaways: return "key takeaways";
            default: return "standard";
        }
    }

    std::string GenerateInitialSummaryPrompt(const std::string& transcript,
                                            const PromptConfig& config) {
        return "First pass: Create an initial summary capturing main points.\n" + transcript;
    }

    std::string GenerateRefinementPrompt(const PromptConfig& config) {
        return "Second pass: Refine the summary for clarity and conciseness, target: " +
               std::to_string(config.target_length_words) + " words.";
    }

    std::string GenerateFactCheckPrompt() {
        return "Third pass: Verify facts, figures, and claims are accurately represented.";
    }

    std::string GenerateFinalPolishPrompt(const PromptConfig& config) {
        return "Final pass: Polish for readability and ensure " +
               GetStyleDescription(config.summary_style) + " style.";
    }

    std::string JoinPasses(const std::vector<std::string>& passes) {
        std::stringstream result;
        for (size_t i = 0; i < passes.size(); i++) {
            result << "=== Pass " << (i + 1) << " ===\n";
            result << passes[i] << "\n\n";
        }
        return result.str();
    }
};

// Content type detector
class ContentTypeDetector {
private:
    struct ContentSignals {
        std::vector<std::string> keywords;
        std::vector<std::string> phrases;
        float confidence_threshold;
    };

    std::unordered_map<PromptTemplateEngine::ContentType, ContentSignals> content_signals_;

public:
    ContentTypeDetector() {
        InitializeSignals();
    }

    PromptTemplateEngine::ContentType DetectContentType(const std::string& transcript) {
        std::unordered_map<PromptTemplateEngine::ContentType, float> scores;

        // Convert transcript to lowercase for matching
        std::string lower_transcript = transcript;
        std::transform(lower_transcript.begin(), lower_transcript.end(),
                      lower_transcript.begin(), ::tolower);

        // Score each content type
        for (const auto& [type, signals] : content_signals_) {
            float score = CalculateScore(lower_transcript, signals);
            scores[type] = score;
        }

        // Find highest scoring type
        auto max_it = std::max_element(scores.begin(), scores.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });

        if (max_it != scores.end() && max_it->second > 0.3f) {
            return max_it->first;
        }

        return PromptTemplateEngine::ContentType::General;
    }

private:
    void InitializeSignals() {
        content_signals_[PromptTemplateEngine::ContentType::TechTalk] = {
            .keywords = {"algorithm", "api", "framework", "deployment", "architecture",
                        "performance", "optimization", "cloud", "database", "microservices"},
            .phrases = {"let me show you", "in production", "best practices", "tech stack"},
            .confidence_threshold = 0.4f
        };

        content_signals_[PromptTemplateEngine::ContentType::Tutorial] = {
            .keywords = {"step", "tutorial", "how to", "guide", "learn", "example",
                        "exercise", "practice", "follow along", "prerequisite"},
            .phrases = {"first we need to", "next step", "make sure you", "let's start by"},
            .confidence_threshold = 0.5f
        };

        content_signals_[PromptTemplateEngine::ContentType::NewsReport] = {
            .keywords = {"breaking", "report", "announced", "statement", "official",
                        "sources", "correspondent", "developing", "update", "confirmed"},
            .phrases = {"according to", "reports say", "officials confirmed", "breaking news"},
            .confidence_threshold = 0.45f
        };

        // Add more signal definitions...
    }

    float CalculateScore(const std::string& transcript, const ContentSignals& signals) {
        float score = 0.0f;
        int total_signals = signals.keywords.size() + signals.phrases.size();

        // Check keywords
        for (const auto& keyword : signals.keywords) {
            if (transcript.find(keyword) != std::string::npos) {
                score += 1.0f / total_signals;
            }
        }

        // Check phrases (weighted higher)
        for (const auto& phrase : signals.phrases) {
            if (transcript.find(phrase) != std::string::npos) {
                score += 1.5f / total_signals;
            }
        }

        return score;
    }
};

// Global instances
static std::unique_ptr<PromptTemplateEngine> g_prompt_engine;
static std::unique_ptr<ContentTypeDetector> g_content_detector;

void InitializePromptEngine() {
    g_prompt_engine = std::make_unique<PromptTemplateEngine>();
    g_content_detector = std::make_unique<ContentTypeDetector>();
}

std::string GenerateSummarizationPrompt(const std::string& transcript,
                                       const std::string& style,
                                       int target_words) {
    if (!g_prompt_engine) InitializePromptEngine();

    // Detect content type
    auto content_type = g_content_detector->DetectContentType(transcript);

    // Build config
    PromptTemplateEngine::PromptConfig config;
    config.content_type = content_type;
    config.summary_style = PromptTemplateEngine::SummaryStyle::Executive;
    config.target_length_words = target_words;
    config.language = "en";
    config.extract_keywords = true;

    return g_prompt_engine->GeneratePrompt(transcript, config);
}

} // namespace summarization