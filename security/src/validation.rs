use ammonia::Builder;
use anyhow::{Context, Result};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, warn};
use validator::{Validate, ValidationError, ValidationErrors};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRule {
    pub field: String,
    pub rule_type: ValidationType,
    pub required: bool,
    pub error_message: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationType {
    Email,
    Url,
    Length { min: Option<usize>, max: Option<usize> },
    Regex { pattern: String },
    Range { min: Option<f64>, max: Option<f64> },
    CustomFunction { function_name: String },
    Whitelist { allowed_values: Vec<String> },
    Blacklist { forbidden_values: Vec<String> },
    NoXSS,
    NoSQLInjection,
    AlphaNumeric,
    Numeric,
    AlphaOnly,
    UUID,
    Base64,
    JSON,
    SafeFilename,
    IPAddress,
    DateFormat { format: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SanitizationRule {
    pub field: String,
    pub sanitizer_type: SanitizationType,
    pub preserve_html_tags: Vec<String>,
    pub preserve_attributes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SanitizationType {
    StripHTML,
    EscapeHTML,
    RemoveScripts,
    WhitelistHTML,
    TrimWhitespace,
    RemoveControlCharacters,
    NormalizeUnicode,
    SQLEscape,
    URLEncode,
    Base64Decode,
    JSONEscape,
    RemoveNulls,
    LimitLength { max_length: usize },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub valid: bool,
    pub errors: HashMap<String, Vec<String>>,
    pub sanitized_data: Option<serde_json::Value>,
    pub security_warnings: Vec<SecurityWarning>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityWarning {
    pub field: String,
    pub warning_type: SecurityWarningType,
    pub message: String,
    pub severity: SecuritySeverity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecurityWarningType {
    PossibleXSS,
    PossibleSQLInjection,
    SuspiciousPattern,
    ExcessiveLength,
    InvalidCharacters,
    MalformedData,
    RateLimitSuspicious,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecuritySeverity {
    Low,
    Medium,
    High,
    Critical,
}

pub struct ValidationService {
    xss_patterns: Vec<Regex>,
    sql_injection_patterns: Vec<Regex>,
    html_sanitizer: Builder<'static>,
    suspicious_patterns: Vec<(Regex, SecurityWarningType)>,
}

impl ValidationService {
    pub fn new() -> Self {
        let xss_patterns = Self::compile_xss_patterns();
        let sql_injection_patterns = Self::compile_sql_injection_patterns();
        let html_sanitizer = Self::create_html_sanitizer();
        let suspicious_patterns = Self::compile_suspicious_patterns();

        Self {
            xss_patterns,
            sql_injection_patterns,
            html_sanitizer,
            suspicious_patterns,
        }
    }

    fn compile_xss_patterns() -> Vec<Regex> {
        let patterns = vec![
            r"(?i)<script[^>]*>.*?</script>",
            r"(?i)javascript:",
            r"(?i)on\w+\s*=",
            r"(?i)<iframe[^>]*>",
            r"(?i)<object[^>]*>",
            r"(?i)<embed[^>]*>",
            r"(?i)<link[^>]*>",
            r"(?i)<meta[^>]*>",
            r"(?i)vbscript:",
            r"(?i)data:text/html",
            r"(?i)expression\s*\(",
            r"(?i)@import",
            r"(?i)<svg[^>]*onload",
            r"(?i)<img[^>]*onerror",
            r"(?i)<body[^>]*onload",
            r"(?i)<form[^>]*action\s*=\s*['\"]?javascript:",
        ];

        patterns
            .into_iter()
            .filter_map(|pattern| Regex::new(pattern).ok())
            .collect()
    }

    fn compile_sql_injection_patterns() -> Vec<Regex> {
        let patterns = vec![
            r"(?i)(\bUNION\b.*\bSELECT\b)",
            r"(?i)(\bSELECT\b.*\bFROM\b)",
            r"(?i)(\bINSERT\b.*\bINTO\b)",
            r"(?i)(\bUPDATE\b.*\bSET\b)",
            r"(?i)(\bDELETE\b.*\bFROM\b)",
            r"(?i)(\bDROP\b.*\bTABLE\b)",
            r"(?i)(\bCREATE\b.*\bTABLE\b)",
            r"(?i)(\bALTER\b.*\bTABLE\b)",
            r"(?i)(\bEXEC\b|\bEXECUTE\b)",
            r"(?i)(\bsp_\w+)",
            r"(?i)(\bxp_\w+)",
            r"(--\s)",
            r"(/\*.*\*/)",
            r"(;\s*--)",
            r"(;\s*/\*)",
            r"(?i)(\bOR\b\s+\d+\s*=\s*\d+)",
            r"(?i)(\bAND\b\s+\d+\s*=\s*\d+)",
            r"(?i)('.*\bOR\b.*')",
            r"(?i)(\".*\bOR\b.*\")",
            r"(?i)(\bHAVING\b.*\d+\s*=\s*\d+)",
        ];

        patterns
            .into_iter()
            .filter_map(|pattern| Regex::new(pattern).ok())
            .collect()
    }

    fn compile_suspicious_patterns() -> Vec<(Regex, SecurityWarningType)> {
        let patterns = vec![
            (r"(?i)(\.\.\/|\.\.\\)", SecurityWarningType::SuspiciousPattern),
            (r"(?i)(\/etc\/passwd|\/etc\/shadow)", SecurityWarningType::SuspiciousPattern),
            (r"(?i)(cmd\.exe|powershell\.exe|bash|sh)", SecurityWarningType::SuspiciousPattern),
            (r"(?i)(\$\{.*\})", SecurityWarningType::SuspiciousPattern),
            (r"(?i)(<%.*%>)", SecurityWarningType::SuspiciousPattern),
            (r"(\\x[0-9a-fA-F]{2})", SecurityWarningType::InvalidCharacters),
            (r"(%[0-9a-fA-F]{2}){5,}", SecurityWarningType::SuspiciousPattern),
            (r"(?i)(eval\s*\(|function\s*\()", SecurityWarningType::PossibleXSS),
            (r"(?i)(base64_decode|eval|exec|system)", SecurityWarningType::SuspiciousPattern),
            (r".{10000,}", SecurityWarningType::ExcessiveLength),
        ];

        patterns
            .into_iter()
            .filter_map(|(pattern, warning_type)| {
                Regex::new(pattern).ok().map(|regex| (regex, warning_type))
            })
            .collect()
    }

    fn create_html_sanitizer() -> Builder<'static> {
        Builder::default()
            .tags(hashset![
                "p", "br", "strong", "em", "u", "i", "b", "h1", "h2", "h3", "h4", "h5", "h6",
                "ul", "ol", "li", "blockquote", "code", "pre", "a", "img"
            ])
            .attributes(hashmap![
                "a" => hashset!["href", "title"],
                "img" => hashset!["src", "alt", "title", "width", "height"],
                "blockquote" => hashset!["cite"]
            ])
            .url_schemes(hashset!["http", "https", "mailto"])
            .link_rel(Some("nofollow noopener"))
            .generic_attributes(hashset!["class", "id"])
            .clean_content_tags(hashset!["script", "style"])
    }

    pub async fn validate_and_sanitize(
        &self,
        data: serde_json::Value,
        validation_rules: &[ValidationRule],
        sanitization_rules: &[SanitizationRule],
    ) -> Result<ValidationResult> {
        let mut errors = HashMap::new();
        let mut security_warnings = Vec::new();
        let mut sanitized_data = data.clone();

        // First, sanitize the data
        self.apply_sanitization(&mut sanitized_data, sanitization_rules, &mut security_warnings);

        // Then validate
        self.apply_validation(&sanitized_data, validation_rules, &mut errors, &mut security_warnings);

        // Perform security checks
        self.perform_security_checks(&sanitized_data, &mut security_warnings);

        let valid = errors.is_empty();

        Ok(ValidationResult {
            valid,
            errors,
            sanitized_data: if valid { Some(sanitized_data) } else { None },
            security_warnings,
        })
    }

    fn apply_sanitization(
        &self,
        data: &mut serde_json::Value,
        rules: &[SanitizationRule],
        warnings: &mut Vec<SecurityWarning>,
    ) {
        if let serde_json::Value::Object(ref mut map) = data {
            for rule in rules {
                if let Some(value) = map.get_mut(&rule.field) {
                    if let Some(string_value) = value.as_str() {
                        let sanitized = self.sanitize_string(string_value, &rule.sanitizer_type, &rule.preserve_html_tags, &rule.preserve_attributes);

                        if sanitized != string_value {
                            warnings.push(SecurityWarning {
                                field: rule.field.clone(),
                                warning_type: SecurityWarningType::MalformedData,
                                message: "Data was sanitized due to potentially unsafe content".to_string(),
                                severity: SecuritySeverity::Medium,
                            });
                        }

                        *value = serde_json::Value::String(sanitized);
                    }
                }
            }
        }
    }

    fn sanitize_string(
        &self,
        input: &str,
        sanitizer_type: &SanitizationType,
        preserve_tags: &[String],
        preserve_attributes: &[String],
    ) -> String {
        match sanitizer_type {
            SanitizationType::StripHTML => {
                html2text::from_read(input.as_bytes(), 80)
            },
            SanitizationType::EscapeHTML => {
                html_escape::encode_text(input).to_string()
            },
            SanitizationType::RemoveScripts => {
                let script_regex = Regex::new(r"(?i)<script[^>]*>.*?</script>").unwrap();
                script_regex.replace_all(input, "").to_string()
            },
            SanitizationType::WhitelistHTML => {
                let mut builder = self.html_sanitizer.clone();

                if !preserve_tags.is_empty() {
                    let tags: std::collections::HashSet<&str> = preserve_tags.iter().map(|s| s.as_str()).collect();
                    builder = builder.tags(tags);
                }

                builder.clean(input).to_string()
            },
            SanitizationType::TrimWhitespace => {
                input.trim().to_string()
            },
            SanitizationType::RemoveControlCharacters => {
                input.chars().filter(|c| !c.is_control() || *c == '\n' || *c == '\r' || *c == '\t').collect()
            },
            SanitizationType::NormalizeUnicode => {
                use unicode_normalization::UnicodeNormalization;
                input.nfc().collect::<String>()
            },
            SanitizationType::SQLEscape => {
                input.replace('\'', "''").replace('\\', "\\\\")
            },
            SanitizationType::URLEncode => {
                urlencoding::encode(input).to_string()
            },
            SanitizationType::Base64Decode => {
                match base64::decode(input) {
                    Ok(decoded) => String::from_utf8_lossy(&decoded).to_string(),
                    Err(_) => input.to_string(),
                }
            },
            SanitizationType::JSONEscape => {
                serde_json::to_string(input).unwrap_or_else(|_| input.to_string())
            },
            SanitizationType::RemoveNulls => {
                input.replace('\0', "")
            },
            SanitizationType::LimitLength { max_length } => {
                if input.len() > *max_length {
                    input.chars().take(*max_length).collect()
                } else {
                    input.to_string()
                }
            },
        }
    }

    fn apply_validation(
        &self,
        data: &serde_json::Value,
        rules: &[ValidationRule],
        errors: &mut HashMap<String, Vec<String>>,
        warnings: &mut Vec<SecurityWarning>,
    ) {
        if let serde_json::Value::Object(map) = data {
            for rule in rules {
                let field_value = map.get(&rule.field);

                // Check if required field is present
                if rule.required && (field_value.is_none() || field_value == Some(&serde_json::Value::Null)) {
                    errors.entry(rule.field.clone())
                        .or_insert_with(Vec::new)
                        .push(rule.error_message.clone().unwrap_or_else(|| "Field is required".to_string()));
                    continue;
                }

                // Skip validation if field is not present and not required
                if field_value.is_none() {
                    continue;
                }

                let value = field_value.unwrap();

                if let Err(validation_errors) = self.validate_field_value(value, &rule.rule_type) {
                    errors.entry(rule.field.clone())
                        .or_insert_with(Vec::new)
                        .extend(validation_errors);
                }

                // Additional security checks
                if let Some(string_value) = value.as_str() {
                    self.check_field_security(string_value, &rule.field, warnings);
                }
            }
        }
    }

    fn validate_field_value(&self, value: &serde_json::Value, rule_type: &ValidationType) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();

        match rule_type {
            ValidationType::Email => {
                if let Some(string_val) = value.as_str() {
                    if !self.is_valid_email(string_val) {
                        errors.push("Invalid email format".to_string());
                    }
                } else {
                    errors.push("Email must be a string".to_string());
                }
            },
            ValidationType::Url => {
                if let Some(string_val) = value.as_str() {
                    if url::Url::parse(string_val).is_err() {
                        errors.push("Invalid URL format".to_string());
                    }
                } else {
                    errors.push("URL must be a string".to_string());
                }
            },
            ValidationType::Length { min, max } => {
                if let Some(string_val) = value.as_str() {
                    let len = string_val.chars().count();
                    if let Some(min_len) = min {
                        if len < *min_len {
                            errors.push(format!("Must be at least {} characters long", min_len));
                        }
                    }
                    if let Some(max_len) = max {
                        if len > *max_len {
                            errors.push(format!("Must be at most {} characters long", max_len));
                        }
                    }
                } else {
                    errors.push("Value must be a string for length validation".to_string());
                }
            },
            ValidationType::Regex { pattern } => {
                if let Some(string_val) = value.as_str() {
                    if let Ok(regex) = Regex::new(pattern) {
                        if !regex.is_match(string_val) {
                            errors.push("Value does not match required pattern".to_string());
                        }
                    } else {
                        errors.push("Invalid regex pattern in validation rule".to_string());
                    }
                } else {
                    errors.push("Value must be a string for regex validation".to_string());
                }
            },
            ValidationType::Range { min, max } => {
                let num_val = if let Some(n) = value.as_f64() {
                    n
                } else if let Some(string_val) = value.as_str() {
                    string_val.parse::<f64>().map_err(|_| vec!["Value must be a number".to_string()])?
                } else {
                    return Err(vec!["Value must be a number".to_string()]);
                };

                if let Some(min_val) = min {
                    if num_val < *min_val {
                        errors.push(format!("Value must be at least {}", min_val));
                    }
                }
                if let Some(max_val) = max {
                    if num_val > *max_val {
                        errors.push(format!("Value must be at most {}", max_val));
                    }
                }
            },
            ValidationType::Whitelist { allowed_values } => {
                if let Some(string_val) = value.as_str() {
                    if !allowed_values.contains(&string_val.to_string()) {
                        errors.push("Value is not in the allowed list".to_string());
                    }
                } else {
                    errors.push("Value must be a string for whitelist validation".to_string());
                }
            },
            ValidationType::Blacklist { forbidden_values } => {
                if let Some(string_val) = value.as_str() {
                    if forbidden_values.contains(&string_val.to_string()) {
                        errors.push("Value is not allowed".to_string());
                    }
                }
            },
            ValidationType::NoXSS => {
                if let Some(string_val) = value.as_str() {
                    if self.contains_xss(string_val) {
                        errors.push("Value contains potentially dangerous content".to_string());
                    }
                }
            },
            ValidationType::NoSQLInjection => {
                if let Some(string_val) = value.as_str() {
                    if self.contains_sql_injection(string_val) {
                        errors.push("Value contains potentially dangerous SQL patterns".to_string());
                    }
                }
            },
            ValidationType::AlphaNumeric => {
                if let Some(string_val) = value.as_str() {
                    if !string_val.chars().all(|c| c.is_alphanumeric()) {
                        errors.push("Value must contain only alphanumeric characters".to_string());
                    }
                }
            },
            ValidationType::Numeric => {
                if let Some(string_val) = value.as_str() {
                    if string_val.parse::<f64>().is_err() {
                        errors.push("Value must be numeric".to_string());
                    }
                } else if !value.is_number() {
                    errors.push("Value must be numeric".to_string());
                }
            },
            ValidationType::AlphaOnly => {
                if let Some(string_val) = value.as_str() {
                    if !string_val.chars().all(|c| c.is_alphabetic()) {
                        errors.push("Value must contain only alphabetic characters".to_string());
                    }
                }
            },
            ValidationType::UUID => {
                if let Some(string_val) = value.as_str() {
                    if uuid::Uuid::parse_str(string_val).is_err() {
                        errors.push("Value must be a valid UUID".to_string());
                    }
                }
            },
            ValidationType::Base64 => {
                if let Some(string_val) = value.as_str() {
                    if base64::decode(string_val).is_err() {
                        errors.push("Value must be valid Base64".to_string());
                    }
                }
            },
            ValidationType::JSON => {
                if let Some(string_val) = value.as_str() {
                    if serde_json::from_str::<serde_json::Value>(string_val).is_err() {
                        errors.push("Value must be valid JSON".to_string());
                    }
                }
            },
            ValidationType::SafeFilename => {
                if let Some(string_val) = value.as_str() {
                    if !self.is_safe_filename(string_val) {
                        errors.push("Filename contains unsafe characters".to_string());
                    }
                }
            },
            ValidationType::IPAddress => {
                if let Some(string_val) = value.as_str() {
                    if string_val.parse::<std::net::IpAddr>().is_err() {
                        errors.push("Value must be a valid IP address".to_string());
                    }
                }
            },
            ValidationType::DateFormat { format } => {
                if let Some(string_val) = value.as_str() {
                    if chrono::NaiveDateTime::parse_from_str(string_val, format).is_err() {
                        errors.push(format!("Date must match format: {}", format));
                    }
                }
            },
            ValidationType::CustomFunction { function_name: _ } => {
                // Custom validation functions would be implemented here
                // For now, we'll just pass
            },
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    fn perform_security_checks(&self, data: &serde_json::Value, warnings: &mut Vec<SecurityWarning>) {
        if let serde_json::Value::Object(map) = data {
            for (field, value) in map {
                if let Some(string_value) = value.as_str() {
                    self.check_field_security(string_value, field, warnings);
                }
            }
        }
    }

    fn check_field_security(&self, value: &str, field: &str, warnings: &mut Vec<SecurityWarning>) {
        // Check for XSS patterns
        if self.contains_xss(value) {
            warnings.push(SecurityWarning {
                field: field.to_string(),
                warning_type: SecurityWarningType::PossibleXSS,
                message: "Field contains patterns that could be used for XSS attacks".to_string(),
                severity: SecuritySeverity::High,
            });
        }

        // Check for SQL injection patterns
        if self.contains_sql_injection(value) {
            warnings.push(SecurityWarning {
                field: field.to_string(),
                warning_type: SecurityWarningType::PossibleSQLInjection,
                message: "Field contains patterns that could be used for SQL injection".to_string(),
                severity: SecuritySeverity::High,
            });
        }

        // Check for suspicious patterns
        for (pattern, warning_type) in &self.suspicious_patterns {
            if pattern.is_match(value) {
                let severity = match warning_type {
                    SecurityWarningType::ExcessiveLength => SecuritySeverity::Medium,
                    SecurityWarningType::InvalidCharacters => SecuritySeverity::Low,
                    _ => SecuritySeverity::Medium,
                };

                warnings.push(SecurityWarning {
                    field: field.to_string(),
                    warning_type: warning_type.clone(),
                    message: format!("Field contains suspicious patterns: {}", pattern.as_str()),
                    severity,
                });
            }
        }
    }

    fn is_valid_email(&self, email: &str) -> bool {
        let email_regex = Regex::new(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$").unwrap();
        email_regex.is_match(email) && email.len() <= 254
    }

    fn contains_xss(&self, input: &str) -> bool {
        self.xss_patterns.iter().any(|pattern| pattern.is_match(input))
    }

    fn contains_sql_injection(&self, input: &str) -> bool {
        self.sql_injection_patterns.iter().any(|pattern| pattern.is_match(input))
    }

    fn is_safe_filename(&self, filename: &str) -> bool {
        let unsafe_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|', '\0'];
        let reserved_names = ["CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9"];

        if filename.is_empty() || filename.len() > 255 {
            return false;
        }

        if filename.starts_with('.') || filename.ends_with('.') || filename.ends_with(' ') {
            return false;
        }

        if filename.chars().any(|c| unsafe_chars.contains(&c)) {
            return false;
        }

        if reserved_names.contains(&filename.to_uppercase().as_str()) {
            return false;
        }

        true
    }

    pub fn create_common_validation_rules() -> Vec<ValidationRule> {
        vec![
            ValidationRule {
                field: "email".to_string(),
                rule_type: ValidationType::Email,
                required: true,
                error_message: Some("Please provide a valid email address".to_string()),
            },
            ValidationRule {
                field: "password".to_string(),
                rule_type: ValidationType::Length { min: Some(8), max: Some(128) },
                required: true,
                error_message: Some("Password must be 8-128 characters long".to_string()),
            },
            ValidationRule {
                field: "username".to_string(),
                rule_type: ValidationType::Regex { pattern: r"^[a-zA-Z0-9_]{3,30}$".to_string() },
                required: true,
                error_message: Some("Username must be 3-30 characters, alphanumeric and underscores only".to_string()),
            },
            ValidationRule {
                field: "url".to_string(),
                rule_type: ValidationType::Url,
                required: false,
                error_message: Some("Please provide a valid URL".to_string()),
            },
            ValidationRule {
                field: "age".to_string(),
                rule_type: ValidationType::Range { min: Some(13.0), max: Some(120.0) },
                required: false,
                error_message: Some("Age must be between 13 and 120".to_string()),
            },
        ]
    }

    pub fn create_common_sanitization_rules() -> Vec<SanitizationRule> {
        vec![
            SanitizationRule {
                field: "content".to_string(),
                sanitizer_type: SanitizationType::WhitelistHTML,
                preserve_html_tags: vec!["p".to_string(), "br".to_string(), "strong".to_string(), "em".to_string()],
                preserve_attributes: vec!["class".to_string()],
            },
            SanitizationRule {
                field: "title".to_string(),
                sanitizer_type: SanitizationType::EscapeHTML,
                preserve_html_tags: vec![],
                preserve_attributes: vec![],
            },
            SanitizationRule {
                field: "description".to_string(),
                sanitizer_type: SanitizationType::RemoveScripts,
                preserve_html_tags: vec![],
                preserve_attributes: vec![],
            },
            SanitizationRule {
                field: "search_query".to_string(),
                sanitizer_type: SanitizationType::LimitLength { max_length: 1000 },
                preserve_html_tags: vec![],
                preserve_attributes: vec![],
            },
        ]
    }

    pub async fn validate_request_data(&self, data: serde_json::Value, endpoint: &str) -> Result<ValidationResult> {
        let (validation_rules, sanitization_rules) = match endpoint {
            "/auth/register" => (
                vec![
                    ValidationRule {
                        field: "email".to_string(),
                        rule_type: ValidationType::Email,
                        required: true,
                        error_message: Some("Valid email is required".to_string()),
                    },
                    ValidationRule {
                        field: "password".to_string(),
                        rule_type: ValidationType::Length { min: Some(8), max: Some(128) },
                        required: true,
                        error_message: Some("Password must be 8-128 characters".to_string()),
                    },
                    ValidationRule {
                        field: "username".to_string(),
                        rule_type: ValidationType::Regex { pattern: r"^[a-zA-Z0-9_]{3,50}$".to_string() },
                        required: true,
                        error_message: Some("Username must be 3-50 chars, alphanumeric/underscore only".to_string()),
                    },
                ],
                vec![
                    SanitizationRule {
                        field: "first_name".to_string(),
                        sanitizer_type: SanitizationType::EscapeHTML,
                        preserve_html_tags: vec![],
                        preserve_attributes: vec![],
                    },
                    SanitizationRule {
                        field: "last_name".to_string(),
                        sanitizer_type: SanitizationType::EscapeHTML,
                        preserve_html_tags: vec![],
                        preserve_attributes: vec![],
                    },
                ]
            ),
            "/feeds" => (
                vec![
                    ValidationRule {
                        field: "url".to_string(),
                        rule_type: ValidationType::Url,
                        required: true,
                        error_message: Some("Valid RSS feed URL is required".to_string()),
                    },
                    ValidationRule {
                        field: "title".to_string(),
                        rule_type: ValidationType::Length { min: Some(1), max: Some(200) },
                        required: true,
                        error_message: Some("Title must be 1-200 characters".to_string()),
                    },
                ],
                vec![
                    SanitizationRule {
                        field: "title".to_string(),
                        sanitizer_type: SanitizationType::EscapeHTML,
                        preserve_html_tags: vec![],
                        preserve_attributes: vec![],
                    },
                    SanitizationRule {
                        field: "description".to_string(),
                        sanitizer_type: SanitizationType::WhitelistHTML,
                        preserve_html_tags: vec!["p".to_string(), "br".to_string()],
                        preserve_attributes: vec![],
                    },
                ]
            ),
            _ => (Self::create_common_validation_rules(), Self::create_common_sanitization_rules()),
        };

        self.validate_and_sanitize(data, &validation_rules, &sanitization_rules).await
    }
}