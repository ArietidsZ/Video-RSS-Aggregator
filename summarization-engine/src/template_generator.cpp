#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <regex>
#include <sstream>
#include <json/json.h>
#include <chrono>

namespace summarization {

class TemplateSummaryGenerator {
public:
    enum class TemplateStyle {
        Professional,      // Business/professional style
        Academic,         // Academic paper style
        Journalistic,     // News article style
        Casual,          // Blog post style
        Technical,       // Technical documentation
        Educational,     // Educational material
        Social,         // Social media style
        Executive,      // Executive brief style
        Creative,       // Creative writing style
        Analytical      // Data analysis style
    };

    struct TemplateConfig {
        TemplateStyle style;
        int target_word_count;
        std::string tone;          // formal, neutral, casual, enthusiastic
        std::string perspective;   // first_person, third_person, passive
        bool include_metadata;
        bool include_timestamps;
        bool include_quotes;
        bool include_statistics;
        bool include_recommendations;
        bool include_conclusion;
        std::string date_format;    // ISO, US, EU
        std::string number_format;  // decimal, comma, scientific
        std::vector<std::string> required_sections;
        std::unordered_map<std::string, std::string> custom_fields;
    };

    struct TemplateSection {
        std::string name;
        std::string header;
        std::string content_pattern;
        int min_words;
        int max_words;
        bool is_required;
        std::vector<std::string> bullet_points;
        std::unordered_map<std::string, std::string> variables;
    };

private:
    std::unordered_map<TemplateStyle, std::vector<TemplateSection>> style_templates_;
    std::unordered_map<std::string, std::string> variable_mappings_;
    TemplateConfig current_config_;

public:
    TemplateSummaryGenerator() {
        InitializeTemplates();
        InitializeVariableMappings();
    }

    std::string GenerateTemplateSummary(const std::string& raw_summary,
                                       const TemplateConfig& config,
                                       const Json::Value& metadata = {}) {
        current_config_ = config;

        // Parse raw summary into components
        auto components = ParseSummaryComponents(raw_summary, metadata);

        // Select appropriate template
        auto template_sections = GetTemplate(config.style);

        // Fill template sections
        auto filled_sections = FillTemplateSections(template_sections, components);

        // Format final summary
        return FormatFinalSummary(filled_sections, config);
    }

    std::string GenerateBatchSummaries(const std::vector<std::string>& summaries,
                                      const TemplateConfig& config) {
        std::stringstream batch_output;

        for (size_t i = 0; i < summaries.size(); i++) {
            if (i > 0) batch_output << "\n---\n\n";

            Json::Value metadata;
            metadata["index"] = static_cast<int>(i + 1);
            metadata["total"] = static_cast<int>(summaries.size());

            batch_output << GenerateTemplateSummary(summaries[i], config, metadata);
        }

        return batch_output.str();
    }

private:
    void InitializeTemplates() {
        // Professional Template
        style_templates_[TemplateStyle::Professional] = {
            {
                .name = "executive_summary",
                .header = "Executive Summary",
                .content_pattern = "{key_findings}",
                .min_words = 50,
                .max_words = 100,
                .is_required = true
            },
            {
                .name = "background",
                .header = "Background",
                .content_pattern = "{context} {situation}",
                .min_words = 30,
                .max_words = 80,
                .is_required = false
            },
            {
                .name = "key_points",
                .header = "Key Points",
                .content_pattern = "bullet_list",
                .min_words = 40,
                .max_words = 150,
                .is_required = true
            },
            {
                .name = "implications",
                .header = "Implications",
                .content_pattern = "{impact} {consequences}",
                .min_words = 30,
                .max_words = 80,
                .is_required = true
            },
            {
                .name = "recommendations",
                .header = "Recommendations",
                .content_pattern = "numbered_list",
                .min_words = 30,
                .max_words = 100,
                .is_required = false
            }
        };

        // Academic Template
        style_templates_[TemplateStyle::Academic] = {
            {
                .name = "abstract",
                .header = "Abstract",
                .content_pattern = "{objective} {methodology} {findings} {conclusions}",
                .min_words = 150,
                .max_words = 250,
                .is_required = true
            },
            {
                .name = "introduction",
                .header = "Introduction",
                .content_pattern = "{background} {research_question}",
                .min_words = 50,
                .max_words = 100,
                .is_required = true
            },
            {
                .name = "methodology",
                .header = "Methodology",
                .content_pattern = "{approach} {data_sources}",
                .min_words = 40,
                .max_words = 80,
                .is_required = false
            },
            {
                .name = "findings",
                .header = "Key Findings",
                .content_pattern = "{results} {analysis}",
                .min_words = 60,
                .max_words = 120,
                .is_required = true
            },
            {
                .name = "discussion",
                .header = "Discussion",
                .content_pattern = "{interpretation} {limitations}",
                .min_words = 50,
                .max_words = 100,
                .is_required = true
            },
            {
                .name = "conclusion",
                .header = "Conclusion",
                .content_pattern = "{summary} {future_work}",
                .min_words = 40,
                .max_words = 80,
                .is_required = true
            }
        };

        // Journalistic Template
        style_templates_[TemplateStyle::Journalistic] = {
            {
                .name = "headline",
                .header = "",
                .content_pattern = "{main_story}",
                .min_words = 5,
                .max_words = 15,
                .is_required = true
            },
            {
                .name = "lead",
                .header = "",
                .content_pattern = "{who} {what} {when} {where} {why}",
                .min_words = 30,
                .max_words = 60,
                .is_required = true
            },
            {
                .name = "body",
                .header = "",
                .content_pattern = "{details} {quotes} {context}",
                .min_words = 100,
                .max_words = 200,
                .is_required = true
            },
            {
                .name = "conclusion",
                .header = "",
                .content_pattern = "{impact} {next_steps}",
                .min_words = 20,
                .max_words = 40,
                .is_required = false
            }
        };

        // Technical Template
        style_templates_[TemplateStyle::Technical] = {
            {
                .name = "overview",
                .header = "Technical Overview",
                .content_pattern = "{technology} {purpose}",
                .min_words = 30,
                .max_words = 60,
                .is_required = true
            },
            {
                .name = "architecture",
                .header = "Architecture & Design",
                .content_pattern = "{components} {interfaces}",
                .min_words = 40,
                .max_words = 100,
                .is_required = true
            },
            {
                .name = "implementation",
                .header = "Implementation Details",
                .content_pattern = "{code_structure} {algorithms}",
                .min_words = 50,
                .max_words = 120,
                .is_required = true
            },
            {
                .name = "performance",
                .header = "Performance Metrics",
                .content_pattern = "table_format",
                .min_words = 20,
                .max_words = 60,
                .is_required = false
            },
            {
                .name = "usage",
                .header = "Usage Examples",
                .content_pattern = "code_blocks",
                .min_words = 30,
                .max_words = 80,
                .is_required = false
            }
        };

        // Educational Template
        style_templates_[TemplateStyle::Educational] = {
            {
                .name = "learning_objectives",
                .header = "Learning Objectives",
                .content_pattern = "bullet_list",
                .min_words = 20,
                .max_words = 50,
                .is_required = true
            },
            {
                .name = "key_concepts",
                .header = "Key Concepts",
                .content_pattern = "{definitions} {explanations}",
                .min_words = 60,
                .max_words = 120,
                .is_required = true
            },
            {
                .name = "examples",
                .header = "Examples",
                .content_pattern = "{practical_examples}",
                .min_words = 40,
                .max_words = 100,
                .is_required = true
            },
            {
                .name = "practice",
                .header = "Practice Questions",
                .content_pattern = "numbered_list",
                .min_words = 30,
                .max_words = 60,
                .is_required = false
            },
            {
                .name = "summary",
                .header = "Summary",
                .content_pattern = "{key_takeaways}",
                .min_words = 30,
                .max_words = 60,
                .is_required = true
            }
        };

        // Executive Brief Template
        style_templates_[TemplateStyle::Executive] = {
            {
                .name = "situation",
                .header = "Situation",
                .content_pattern = "{current_state}",
                .min_words = 20,
                .max_words = 40,
                .is_required = true
            },
            {
                .name = "complication",
                .header = "Complication",
                .content_pattern = "{challenges} {risks}",
                .min_words = 30,
                .max_words = 60,
                .is_required = true
            },
            {
                .name = "question",
                .header = "Key Question",
                .content_pattern = "{decision_point}",
                .min_words = 10,
                .max_words = 25,
                .is_required = true
            },
            {
                .name = "answer",
                .header = "Recommendation",
                .content_pattern = "{solution} {rationale}",
                .min_words = 40,
                .max_words = 80,
                .is_required = true
            },
            {
                .name = "next_steps",
                .header = "Next Steps",
                .content_pattern = "action_items",
                .min_words = 20,
                .max_words = 50,
                .is_required = true
            }
        };
    }

    void InitializeVariableMappings() {
        variable_mappings_ = {
            {"{key_findings}", "extract_key_findings"},
            {"{context}", "extract_context"},
            {"{situation}", "extract_situation"},
            {"{impact}", "extract_impact"},
            {"{consequences}", "extract_consequences"},
            {"{objective}", "extract_objective"},
            {"{methodology}", "extract_methodology"},
            {"{findings}", "extract_findings"},
            {"{conclusions}", "extract_conclusions"},
            {"{who}", "extract_who"},
            {"{what}", "extract_what"},
            {"{when}", "extract_when"},
            {"{where}", "extract_where"},
            {"{why}", "extract_why"},
            {"{technology}", "extract_technology"},
            {"{purpose}", "extract_purpose"},
            {"{components}", "extract_components"},
            {"{current_state}", "extract_current_state"},
            {"{challenges}", "extract_challenges"},
            {"{solution}", "extract_solution"}
        };
    }

    struct SummaryComponents {
        std::string main_content;
        std::vector<std::string> key_points;
        std::vector<std::string> supporting_details;
        std::vector<std::string> quotes;
        std::vector<std::string> statistics;
        std::vector<std::string> recommendations;
        std::string conclusion;
        std::unordered_map<std::string, std::string> extracted_variables;
        Json::Value metadata;
    };

    SummaryComponents ParseSummaryComponents(const std::string& raw_summary,
                                            const Json::Value& metadata) {
        SummaryComponents components;
        components.metadata = metadata;

        // Extract main content
        components.main_content = ExtractMainContent(raw_summary);

        // Extract key points
        components.key_points = ExtractKeyPoints(raw_summary);

        // Extract supporting details
        components.supporting_details = ExtractSupportingDetails(raw_summary);

        // Extract quotes if present
        components.quotes = ExtractQuotes(raw_summary);

        // Extract statistics
        components.statistics = ExtractStatistics(raw_summary);

        // Extract recommendations
        components.recommendations = ExtractRecommendations(raw_summary);

        // Extract conclusion
        components.conclusion = ExtractConclusion(raw_summary);

        // Extract template variables
        for (const auto& [var_pattern, extractor] : variable_mappings_) {
            components.extracted_variables[var_pattern] =
                ExtractVariable(raw_summary, extractor);
        }

        return components;
    }

    std::vector<TemplateSection> GetTemplate(TemplateStyle style) {
        if (style_templates_.count(style)) {
            return style_templates_[style];
        }
        // Return default template
        return CreateDefaultTemplate();
    }

    std::vector<TemplateSection> CreateDefaultTemplate() {
        return {
            {
                .name = "summary",
                .header = "Summary",
                .content_pattern = "{main_content}",
                .min_words = 100,
                .max_words = 300,
                .is_required = true
            },
            {
                .name = "key_points",
                .header = "Key Points",
                .content_pattern = "bullet_list",
                .min_words = 50,
                .max_words = 150,
                .is_required = true
            }
        };
    }

    std::vector<TemplateSection> FillTemplateSections(
        const std::vector<TemplateSection>& template_sections,
        const SummaryComponents& components) {

        std::vector<TemplateSection> filled_sections;

        for (auto section : template_sections) {
            // Skip optional sections if no content
            if (!section.is_required && !HasContentForSection(section, components)) {
                continue;
            }

            // Fill section content
            if (section.content_pattern == "bullet_list") {
                section.bullet_points = GenerateBulletPoints(section.name, components);
            } else if (section.content_pattern == "numbered_list") {
                section.bullet_points = GenerateNumberedList(section.name, components);
            } else if (section.content_pattern == "table_format") {
                section.variables["table"] = GenerateTable(section.name, components);
            } else if (section.content_pattern == "code_blocks") {
                section.variables["code"] = GenerateCodeBlocks(section.name, components);
            } else if (section.content_pattern == "action_items") {
                section.bullet_points = GenerateActionItems(components);
            } else {
                // Replace variables in content pattern
                std::string content = section.content_pattern;
                for (const auto& [var, value] : components.extracted_variables) {
                    content = ReplaceVariable(content, var, value);
                }
                section.variables["content"] = content;
            }

            // Ensure word count constraints
            AdjustSectionLength(section, section.min_words, section.max_words);

            filled_sections.push_back(section);
        }

        return filled_sections;
    }

    std::string FormatFinalSummary(const std::vector<TemplateSection>& sections,
                                  const TemplateConfig& config) {
        std::stringstream formatted;

        // Add metadata header if configured
        if (config.include_metadata) {
            formatted << FormatMetadataHeader(config) << "\n\n";
        }

        // Format each section
        for (const auto& section : sections) {
            // Add section header
            if (!section.header.empty()) {
                formatted << FormatSectionHeader(section.header, config) << "\n";
            }

            // Add section content
            if (!section.bullet_points.empty()) {
                formatted << FormatBulletPoints(section.bullet_points, config) << "\n";
            } else if (section.variables.count("table")) {
                formatted << section.variables.at("table") << "\n";
            } else if (section.variables.count("code")) {
                formatted << section.variables.at("code") << "\n";
            } else if (section.variables.count("content")) {
                formatted << FormatParagraph(section.variables.at("content"), config) << "\n";
            }

            formatted << "\n";
        }

        // Add footer if needed
        if (config.include_conclusion && !HasConclusionSection(sections)) {
            formatted << FormatConclusion(config) << "\n";
        }

        return formatted.str();
    }

    std::string FormatMetadataHeader(const TemplateConfig& config) {
        std::stringstream header;

        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);

        header << "---\n";
        header << "Date: " << FormatDate(time_t, config.date_format) << "\n";
        header << "Style: " << GetStyleName(config.style) << "\n";
        header << "Word Count: " << config.target_word_count << "\n";

        for (const auto& [key, value] : config.custom_fields) {
            header << key << ": " << value << "\n";
        }

        header << "---";

        return header.str();
    }

    std::string FormatSectionHeader(const std::string& header,
                                   const TemplateConfig& config) {
        switch (config.style) {
            case TemplateStyle::Academic:
                return "## " + header;
            case TemplateStyle::Professional:
                return "**" + header + "**";
            case TemplateStyle::Technical:
                return "### " + header;
            default:
                return header + ":";
        }
    }

    std::string FormatBulletPoints(const std::vector<std::string>& points,
                                  const TemplateConfig& config) {
        std::stringstream formatted;

        for (const auto& point : points) {
            switch (config.style) {
                case TemplateStyle::Professional:
                case TemplateStyle::Executive:
                    formatted << "• " << point << "\n";
                    break;
                case TemplateStyle::Technical:
                    formatted << "- " << point << "\n";
                    break;
                default:
                    formatted << "* " << point << "\n";
            }
        }

        return formatted.str();
    }

    std::string FormatParagraph(const std::string& content,
                               const TemplateConfig& config) {
        std::string formatted = content;

        // Apply tone adjustments
        if (config.tone == "formal") {
            formatted = ApplyFormalTone(formatted);
        } else if (config.tone == "casual") {
            formatted = ApplyCasualTone(formatted);
        }

        // Apply perspective
        if (config.perspective == "first_person") {
            formatted = ConvertToFirstPerson(formatted);
        } else if (config.perspective == "passive") {
            formatted = ConvertToPassiveVoice(formatted);
        }

        return formatted;
    }

    // Content extraction functions
    std::string ExtractMainContent(const std::string& raw_summary) {
        // Extract the main body of the summary
        return raw_summary.substr(0, std::min(size_t(500), raw_summary.length()));
    }

    std::vector<std::string> ExtractKeyPoints(const std::string& raw_summary) {
        std::vector<std::string> points;

        // Simple extraction based on sentence importance
        auto sentences = SplitIntoSentences(raw_summary);

        for (const auto& sentence : sentences) {
            if (IsKeyPoint(sentence)) {
                points.push_back(sentence);
                if (points.size() >= 5) break;
            }
        }

        return points;
    }

    std::vector<std::string> ExtractSupportingDetails(const std::string& raw_summary) {
        // Extract supporting information
        return {};
    }

    std::vector<std::string> ExtractQuotes(const std::string& raw_summary) {
        std::vector<std::string> quotes;

        std::regex quote_regex(R"("([^"]+)")");
        std::sregex_iterator iter(raw_summary.begin(), raw_summary.end(), quote_regex);
        std::sregex_iterator end;

        for (; iter != end; ++iter) {
            quotes.push_back(iter->str(1));
        }

        return quotes;
    }

    std::vector<std::string> ExtractStatistics(const std::string& raw_summary) {
        std::vector<std::string> statistics;

        // Extract sentences containing numbers
        auto sentences = SplitIntoSentences(raw_summary);

        for (const auto& sentence : sentences) {
            if (ContainsStatistics(sentence)) {
                statistics.push_back(sentence);
            }
        }

        return statistics;
    }

    std::vector<std::string> ExtractRecommendations(const std::string& raw_summary) {
        std::vector<std::string> recommendations;

        // Look for action-oriented sentences
        auto sentences = SplitIntoSentences(raw_summary);

        for (const auto& sentence : sentences) {
            if (IsRecommendation(sentence)) {
                recommendations.push_back(sentence);
            }
        }

        return recommendations;
    }

    std::string ExtractConclusion(const std::string& raw_summary) {
        // Extract last paragraph or concluding statement
        auto sentences = SplitIntoSentences(raw_summary);

        if (!sentences.empty()) {
            return sentences.back();
        }

        return "";
    }

    std::string ExtractVariable(const std::string& raw_summary,
                               const std::string& extractor) {
        // Extract specific variable based on extractor type
        if (extractor == "extract_key_findings") {
            auto points = ExtractKeyPoints(raw_summary);
            return points.empty() ? "" : points[0];
        }
        // Add more extractors as needed

        return "";
    }

    // Helper functions
    bool HasContentForSection(const TemplateSection& section,
                             const SummaryComponents& components) {
        if (section.name == "quotes" && components.quotes.empty()) return false;
        if (section.name == "statistics" && components.statistics.empty()) return false;
        if (section.name == "recommendations" && components.recommendations.empty()) return false;

        return true;
    }

    std::vector<std::string> GenerateBulletPoints(const std::string& section_name,
                                                 const SummaryComponents& components) {
        if (section_name == "key_points") {
            return components.key_points;
        } else if (section_name == "learning_objectives") {
            return GenerateLearningObjectives(components);
        }

        return {};
    }

    std::vector<std::string> GenerateNumberedList(const std::string& section_name,
                                                 const SummaryComponents& components) {
        auto items = GenerateBulletPoints(section_name, components);

        // Convert to numbered format
        std::vector<std::string> numbered;
        for (size_t i = 0; i < items.size(); i++) {
            numbered.push_back(std::to_string(i + 1) + ". " + items[i]);
        }

        return numbered;
    }

    std::string GenerateTable(const std::string& section_name,
                             const SummaryComponents& components) {
        std::stringstream table;

        table << "| Metric | Value |\n";
        table << "|--------|-------|\n";

        // Add statistics as table rows
        for (const auto& stat : components.statistics) {
            auto parts = ParseStatistic(stat);
            if (parts.first.empty()) continue;
            table << "| " << parts.first << " | " << parts.second << " |\n";
        }

        return table.str();
    }

    std::string GenerateCodeBlocks(const std::string& section_name,
                                  const SummaryComponents& components) {
        // Generate code examples if applicable
        return "```\n// Example code\n```\n";
    }

    std::vector<std::string> GenerateActionItems(const SummaryComponents& components) {
        std::vector<std::string> action_items;

        for (const auto& rec : components.recommendations) {
            action_items.push_back("[ ] " + rec);
        }

        return action_items;
    }

    std::vector<std::string> GenerateLearningObjectives(const SummaryComponents& components) {
        std::vector<std::string> objectives;

        for (const auto& point : components.key_points) {
            objectives.push_back("Understand " + point);
            if (objectives.size() >= 3) break;
        }

        return objectives;
    }

    void AdjustSectionLength(TemplateSection& section, int min_words, int max_words) {
        // Adjust content to meet word count constraints
        // This is a simplified implementation
    }

    std::string ReplaceVariable(const std::string& content,
                               const std::string& variable,
                               const std::string& value) {
        std::string result = content;
        size_t pos = 0;

        while ((pos = result.find(variable, pos)) != std::string::npos) {
            result.replace(pos, variable.length(), value);
            pos += value.length();
        }

        return result;
    }

    bool HasConclusionSection(const std::vector<TemplateSection>& sections) {
        for (const auto& section : sections) {
            if (section.name == "conclusion") return true;
        }
        return false;
    }

    std::string FormatConclusion(const TemplateConfig& config) {
        return "In conclusion, this summary has presented the key information in a structured format.";
    }

    std::string FormatDate(std::time_t time, const std::string& format) {
        char buffer[100];
        std::strftime(buffer, sizeof(buffer), "%Y-%m-%d", std::localtime(&time));
        return std::string(buffer);
    }

    std::string GetStyleName(TemplateStyle style) {
        switch (style) {
            case TemplateStyle::Professional: return "Professional";
            case TemplateStyle::Academic: return "Academic";
            case TemplateStyle::Journalistic: return "Journalistic";
            case TemplateStyle::Technical: return "Technical";
            case TemplateStyle::Educational: return "Educational";
            case TemplateStyle::Executive: return "Executive Brief";
            default: return "Default";
        }
    }

    std::string ApplyFormalTone(const std::string& text) {
        std::string formal = text;

        // Replace informal phrases
        std::unordered_map<std::string, std::string> replacements = {
            {"can't", "cannot"},
            {"won't", "will not"},
            {"it's", "it is"},
            {"don't", "do not"}
        };

        for (const auto& [informal, formal_replacement] : replacements) {
            size_t pos = 0;
            while ((pos = formal.find(informal, pos)) != std::string::npos) {
                formal.replace(pos, informal.length(), formal_replacement);
                pos += formal_replacement.length();
            }
        }

        return formal;
    }

    std::string ApplyCasualTone(const std::string& text) {
        // Make text more casual
        return text;
    }

    std::string ConvertToFirstPerson(const std::string& text) {
        // Convert to first person perspective
        return text;
    }

    std::string ConvertToPassiveVoice(const std::string& text) {
        // Convert to passive voice
        return text;
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

    bool IsKeyPoint(const std::string& sentence) {
        // Check if sentence contains key point indicators
        std::vector<std::string> indicators = {
            "important", "key", "main", "primary", "significant",
            "crucial", "essential", "fundamental"
        };

        for (const auto& indicator : indicators) {
            if (sentence.find(indicator) != std::string::npos) {
                return true;
            }
        }

        return false;
    }

    bool ContainsStatistics(const std::string& sentence) {
        // Check for numbers and percentages
        std::regex number_regex(R"(\d+\.?\d*%?)");
        return std::regex_search(sentence, number_regex);
    }

    bool IsRecommendation(const std::string& sentence) {
        std::vector<std::string> action_words = {
            "should", "must", "recommend", "suggest", "advise",
            "consider", "implement", "adopt", "ensure"
        };

        for (const auto& word : action_words) {
            if (sentence.find(word) != std::string::npos) {
                return true;
            }
        }

        return false;
    }

    std::pair<std::string, std::string> ParseStatistic(const std::string& stat) {
        // Parse statistic into name and value
        size_t colon_pos = stat.find(':');
        if (colon_pos != std::string::npos) {
            return {stat.substr(0, colon_pos), stat.substr(colon_pos + 1)};
        }

        return {"", ""};
    }
};

// Consistency enforcer for batch summaries
class SummaryConsistencyEnforcer {
private:
    std::unordered_map<std::string, std::string> terminology_map_;
    std::unordered_map<std::string, std::string> style_rules_;

public:
    SummaryConsistencyEnforcer() {
        InitializeTerminologyMap();
        InitializeStyleRules();
    }

    std::string EnforceConsistency(const std::string& summary,
                                  const std::vector<std::string>& previous_summaries = {}) {
        std::string consistent_summary = summary;

        // Apply terminology consistency
        consistent_summary = ApplyTerminologyConsistency(consistent_summary);

        // Apply style consistency
        consistent_summary = ApplyStyleConsistency(consistent_summary);

        // Ensure consistency with previous summaries
        if (!previous_summaries.empty()) {
            consistent_summary = AlignWithPreviousSummaries(consistent_summary, previous_summaries);
        }

        return consistent_summary;
    }

private:
    void InitializeTerminologyMap() {
        // Standard terminology mappings
        terminology_map_ = {
            {"AI", "artificial intelligence"},
            {"ML", "machine learning"},
            {"API", "application programming interface"},
            {"UI", "user interface"},
            {"UX", "user experience"}
        };
    }

    void InitializeStyleRules() {
        style_rules_ = {
            {"date_format", "YYYY-MM-DD"},
            {"number_format", "comma_separated"},
            {"citation_style", "APA"},
            {"heading_style", "title_case"}
        };
    }

    std::string ApplyTerminologyConsistency(const std::string& text) {
        std::string result = text;

        for (const auto& [abbrev, full_form] : terminology_map_) {
            // Use consistent terminology throughout
            size_t pos = 0;
            while ((pos = result.find(abbrev, pos)) != std::string::npos) {
                // Check if it's a standalone word
                if ((pos == 0 || !std::isalnum(result[pos - 1])) &&
                    (pos + abbrev.length() >= result.length() ||
                     !std::isalnum(result[pos + abbrev.length()]))) {
                    result.replace(pos, abbrev.length(), full_form);
                    pos += full_form.length();
                } else {
                    pos += abbrev.length();
                }
            }
        }

        return result;
    }

    std::string ApplyStyleConsistency(const std::string& text) {
        // Apply consistent formatting rules
        return text;
    }

    std::string AlignWithPreviousSummaries(const std::string& summary,
                                          const std::vector<std::string>& previous) {
        // Ensure terminology and style match previous summaries
        return summary;
    }
};

// Global instances
static std::unique_ptr<TemplateSummaryGenerator> g_template_generator;
static std::unique_ptr<SummaryConsistencyEnforcer> g_consistency_enforcer;

void InitializeTemplateGeneration() {
    g_template_generator = std::make_unique<TemplateSummaryGenerator>();
    g_consistency_enforcer = std::make_unique<SummaryConsistencyEnforcer>();
}

std::string GenerateTemplateSummary(const std::string& raw_summary,
                                   TemplateSummaryGenerator::TemplateStyle style,
                                   int target_words) {
    if (!g_template_generator) {
        InitializeTemplateGeneration();
    }

    TemplateSummaryGenerator::TemplateConfig config;
    config.style = style;
    config.target_word_count = target_words;
    config.tone = "neutral";
    config.include_metadata = true;

    auto templated = g_template_generator->GenerateTemplateSummary(raw_summary, config);

    if (g_consistency_enforcer) {
        templated = g_consistency_enforcer->EnforceConsistency(templated);
    }

    return templated;
}

} // namespace summarization