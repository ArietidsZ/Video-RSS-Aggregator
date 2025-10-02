use ammonia::Builder;
use anyhow::Result;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use html_escape;

pub type CustomValidatorFn = Arc<dyn Fn(&serde_json::Value) -> Result<(), String> + Send + Sync>;

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
    email_regex: Regex,
    url_regex: Regex,
    xss_patterns: Vec<Regex>,
    sql_injection_patterns: Vec<Regex>,
    html_sanitizer: Builder<'static>,
    suspicious_patterns: Vec<(Regex, SecurityWarningType)>,
    custom_validators: HashMap<String, CustomValidatorFn>,
}

impl ValidationService {
    pub fn new() -> Self {
        let email_regex = Regex::new(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$").unwrap();
        let url_regex = Regex::new(r"^https?://[^\s/$.?#].[^\s]*$").unwrap();
        let xss_patterns = Self::compile_xss_patterns();
        let sql_injection_patterns = Self::compile_sql_injection_patterns();
        let html_sanitizer = Self::create_html_sanitizer();
        let suspicious_patterns = Self::compile_suspicious_patterns();
        let custom_validators = Self::create_default_validators();

        Self {
            email_regex,
            url_regex,
            xss_patterns,
            sql_injection_patterns,
            html_sanitizer,
            suspicious_patterns,
            custom_validators,
        }
    }

    fn create_default_validators() -> HashMap<String, CustomValidatorFn> {
        let validators = HashMap::new();
        validators
    }

    pub fn register_custom_validator(&mut self, name: String, validator: CustomValidatorFn) {
        self.custom_validators.insert(name, validator);
    }

    fn compile_xss_patterns() -> Vec<Regex> {
        let patterns = vec![
            r"(?i)<script",
            r"(?i)javascript:",
            r"(?i)<iframe",
        ];

        patterns
            .into_iter()
            .filter_map(|pattern| Regex::new(pattern).ok())
            .collect()
    }

    fn compile_sql_injection_patterns() -> Vec<Regex> {
        let patterns = vec![
            r"(?i)(UNION.+SELECT)",
            r"(?i)(SELECT.+FROM)",
            r"(?i)(INSERT.+INTO)",
        ];

        patterns
            .into_iter()
            .filter_map(|pattern| Regex::new(pattern).ok())
            .collect()
    }

    fn compile_suspicious_patterns() -> Vec<(Regex, SecurityWarningType)> {
        let patterns = vec![
            (r"\.\.\/", SecurityWarningType::SuspiciousPattern),
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
    }

    pub async fn validate_and_sanitize(
        &self,
        mut data: serde_json::Value,
        validation_rules: &[ValidationRule],
        sanitization_rules: &[SanitizationRule],
    ) -> Result<ValidationResult> {
        let mut errors: HashMap<String, Vec<String>> = HashMap::new();
        let security_warnings = Vec::new();

        // Apply validation rules
        if let serde_json::Value::Object(ref obj) = data {
            for rule in validation_rules {
                let field_value = obj.get(&rule.field);

                // Check required fields
                if rule.required && field_value.is_none() {
                    errors.entry(rule.field.clone())
                        .or_insert_with(Vec::new)
                        .push(rule.error_message.clone()
                            .unwrap_or_else(|| format!("Field '{}' is required", rule.field)));
                    continue;
                }

                if let Some(value) = field_value {
                    let validation_errors = self.validate_field(value, &rule.rule_type)?;
                    if !validation_errors.is_empty() {
                        errors.insert(rule.field.clone(), validation_errors);
                    }
                }
            }
        }

        // Apply sanitization rules
        if let serde_json::Value::Object(ref mut obj) = data {
            for rule in sanitization_rules {
                if let Some(value) = obj.get_mut(&rule.field) {
                    if let serde_json::Value::String(s) = value {
                        let sanitized = self.sanitize_value(s, &rule.sanitizer_type, rule)?;
                        *value = serde_json::Value::String(sanitized);
                    }
                }
            }
        }

        let valid = errors.is_empty();

        Ok(ValidationResult {
            valid,
            errors,
            sanitized_data: Some(data),
            security_warnings,
        })
    }

    fn validate_field(&self, value: &serde_json::Value, rule_type: &ValidationType) -> Result<Vec<String>> {
        let mut errors = Vec::new();

        match rule_type {
            ValidationType::Email => {
                if let serde_json::Value::String(s) = value {
                    if !self.email_regex.is_match(s) {
                        errors.push("Invalid email format".to_string());
                    }
                }
            },
            ValidationType::Url => {
                if let serde_json::Value::String(s) = value {
                    if !self.url_regex.is_match(s) {
                        errors.push("Invalid URL format".to_string());
                    }
                }
            },
            ValidationType::Length { min, max } => {
                if let serde_json::Value::String(s) = value {
                    if let Some(min_len) = min {
                        if s.len() < *min_len {
                            errors.push(format!("Length must be at least {}", min_len));
                        }
                    }
                    if let Some(max_len) = max {
                        if s.len() > *max_len {
                            errors.push(format!("Length must not exceed {}", max_len));
                        }
                    }
                }
            },
            ValidationType::NoXSS => {
                if let serde_json::Value::String(s) = value {
                    if self.contains_xss(s) {
                        errors.push("Potential XSS detected".to_string());
                    }
                }
            },
            ValidationType::NoSQLInjection => {
                if let serde_json::Value::String(s) = value {
                    if self.contains_sql_injection(s) {
                        errors.push("Potential SQL injection detected".to_string());
                    }
                }
            },
            ValidationType::AlphaNumeric => {
                if let serde_json::Value::String(s) = value {
                    if !s.chars().all(|c| c.is_alphanumeric()) {
                        errors.push("Must contain only alphanumeric characters".to_string());
                    }
                }
            },
            ValidationType::UUID => {
                if let serde_json::Value::String(s) = value {
                    if uuid::Uuid::parse_str(s).is_err() {
                        errors.push("Invalid UUID format".to_string());
                    }
                }
            },
            _ => {} // Other validation types not implemented yet
        }

        Ok(errors)
    }

    fn sanitize_value(&self, value: &str, sanitizer_type: &SanitizationType, _rule: &SanitizationRule) -> Result<String> {
        match sanitizer_type {
            SanitizationType::StripHTML => {
                Ok(ammonia::clean(value))
            },
            SanitizationType::EscapeHTML => {
                Ok(html_escape::encode_text(value).to_string())
            },
            SanitizationType::TrimWhitespace => {
                Ok(value.trim().to_string())
            },
            SanitizationType::RemoveControlCharacters => {
                Ok(value.chars().filter(|c| !c.is_control()).collect())
            },
            SanitizationType::RemoveNulls => {
                Ok(value.replace('\0', ""))
            },
            SanitizationType::LimitLength { max_length } => {
                Ok(value.chars().take(*max_length).collect())
            },
            SanitizationType::WhitelistHTML => {
                // For now, use default ammonia cleaner
                // Custom tag whitelisting would require more complex setup
                Ok(ammonia::clean(value))
            },
            _ => Ok(value.to_string()) // Other sanitization types not fully implemented
        }
    }

    fn contains_xss(&self, input: &str) -> bool {
        self.xss_patterns.iter().any(|pattern| pattern.is_match(input))
    }

    fn contains_sql_injection(&self, input: &str) -> bool {
        self.sql_injection_patterns.iter().any(|pattern| pattern.is_match(input))
    }

    pub async fn validate_request_data(&self, data: serde_json::Value, endpoint: &str) -> Result<ValidationResult> {
        // Apply endpoint-specific validation rules
        let validation_rules = self.get_validation_rules_for_endpoint(endpoint);
        let sanitization_rules = self.get_sanitization_rules_for_endpoint(endpoint);

        self.validate_and_sanitize(data, &validation_rules, &sanitization_rules).await
    }

    fn get_validation_rules_for_endpoint(&self, endpoint: &str) -> Vec<ValidationRule> {
        // Define endpoint-specific validation rules
        match endpoint {
            "/auth/register" | "/auth/login" => vec![
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
            ],
            "/users" => vec![
                ValidationRule {
                    field: "username".to_string(),
                    rule_type: ValidationType::AlphaNumeric,
                    required: false,
                    error_message: Some("Username must be alphanumeric".to_string()),
                },
            ],
            _ => vec![] // Default: no specific rules
        }
    }

    fn get_sanitization_rules_for_endpoint(&self, endpoint: &str) -> Vec<SanitizationRule> {
        // Define endpoint-specific sanitization rules
        match endpoint {
            "/auth/register" | "/users" => vec![
                SanitizationRule {
                    field: "username".to_string(),
                    sanitizer_type: SanitizationType::TrimWhitespace,
                    preserve_html_tags: vec![],
                    preserve_attributes: vec![],
                },
                SanitizationRule {
                    field: "email".to_string(),
                    sanitizer_type: SanitizationType::TrimWhitespace,
                    preserve_html_tags: vec![],
                    preserve_attributes: vec![],
                },
            ],
            _ => vec![] // Default: no specific rules
        }
    }
}
