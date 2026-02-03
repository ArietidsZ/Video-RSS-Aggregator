use axum::http::HeaderMap;

#[derive(Clone, Debug)]
pub struct Auth {
    api_key: Option<String>,
}

impl Auth {
    pub fn new(api_key: Option<String>) -> Self {
        Self { api_key }
    }

    pub fn is_authorized(&self, headers: &HeaderMap) -> bool {
        let Some(expected) = &self.api_key else {
            return true;
        };

        let token = extract_token(headers);
        token.map(|value| value == *expected).unwrap_or(false)
    }
}

fn extract_token(headers: &HeaderMap) -> Option<String> {
    if let Some(value) = headers.get("authorization") {
        if let Ok(text) = value.to_str() {
            let parts: Vec<&str> = text.split_whitespace().collect();
            if parts.len() == 2 && parts[0].eq_ignore_ascii_case("bearer") {
                return Some(parts[1].to_string());
            }
        }
    }

    headers
        .get("x-api-key")
        .and_then(|value| value.to_str().ok())
        .map(|value| value.to_string())
}
