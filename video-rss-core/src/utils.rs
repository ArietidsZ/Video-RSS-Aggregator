use crate::types::Platform;
use regex::Regex;

pub fn extract_video_id(url: &str, platform: Platform) -> Option<String> {
    match platform {
        Platform::Bilibili => {
            let patterns = [
                r"bilibili\.com/video/([A-Za-z0-9]+)",
                r"bilibili\.com/video/av(\d+)",
                r"b23\.tv/([A-Za-z0-9]+)",
            ];

            for pattern in &patterns {
                if let Ok(re) = Regex::new(pattern) {
                    if let Some(captures) = re.captures(url) {
                        if let Some(id) = captures.get(1) {
                            return Some(id.as_str().to_string());
                        }
                    }
                }
            }
        }
        Platform::Douyin => {
            let patterns = [
                r"douyin\.com/video/(\d+)",
                r"v\.douyin\.com/([A-Za-z0-9]+)",
            ];

            for pattern in &patterns {
                if let Ok(re) = Regex::new(pattern) {
                    if let Some(captures) = re.captures(url) {
                        if let Some(id) = captures.get(1) {
                            return Some(id.as_str().to_string());
                        }
                    }
                }
            }
        }
        Platform::Kuaishou => {
            let patterns = [
                r"kuaishou\.com/profile/[^/]+/(\w+)",
                r"kuaishou\.com/short-video/(\w+)",
            ];

            for pattern in &patterns {
                if let Ok(re) = Regex::new(pattern) {
                    if let Some(captures) = re.captures(url) {
                        if let Some(id) = captures.get(1) {
                            return Some(id.as_str().to_string());
                        }
                    }
                }
            }
        }
        Platform::YouTube => {
            let patterns = [
                r"youtube\.com/watch\?v=([A-Za-z0-9_-]+)",
                r"youtu\.be/([A-Za-z0-9_-]+)",
            ];

            for pattern in &patterns {
                if let Ok(re) = Regex::new(pattern) {
                    if let Some(captures) = re.captures(url) {
                        if let Some(id) = captures.get(1) {
                            return Some(id.as_str().to_string());
                        }
                    }
                }
            }
        }
    }

    None
}

pub fn parse_duration(duration_str: &str) -> Option<u64> {
    if let Ok(seconds) = duration_str.parse::<u64>() {
        return Some(seconds);
    }

    if duration_str.contains(':') {
        let parts: Vec<&str> = duration_str.split(':').collect();
        match parts.len() {
            2 => {
                // MM:SS format
                if let (Ok(minutes), Ok(seconds)) = (parts[0].parse::<u64>(), parts[1].parse::<u64>()) {
                    return Some(minutes * 60 + seconds);
                }
            }
            3 => {
                // HH:MM:SS format
                if let (Ok(hours), Ok(minutes), Ok(seconds)) = (
                    parts[0].parse::<u64>(),
                    parts[1].parse::<u64>(),
                    parts[2].parse::<u64>(),
                ) {
                    return Some(hours * 3600 + minutes * 60 + seconds);
                }
            }
            _ => {}
        }
    }

    None
}

pub fn format_duration(seconds: u64) -> String {
    let hours = seconds / 3600;
    let minutes = (seconds % 3600) / 60;
    let secs = seconds % 60;

    if hours > 0 {
        format!("{:02}:{:02}:{:02}", hours, minutes, secs)
    } else {
        format!("{:02}:{:02}", minutes, secs)
    }
}

pub fn clean_html(text: &str) -> String {
    // Simple HTML tag removal
    let re = Regex::new(r"<[^>]*>").unwrap();
    re.replace_all(text, "").to_string()
}

pub fn truncate_text(text: &str, max_length: usize) -> String {
    if text.len() <= max_length {
        text.to_string()
    } else {
        format!("{}...", &text[..max_length])
    }
}

pub fn sanitize_xml(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_bilibili_id() {
        assert_eq!(
            extract_video_id("https://www.bilibili.com/video/BV1234567890", Platform::Bilibili),
            Some("BV1234567890".to_string())
        );

        assert_eq!(
            extract_video_id("https://www.bilibili.com/video/av12345", Platform::Bilibili),
            Some("12345".to_string())
        );
    }

    #[test]
    fn test_parse_duration() {
        assert_eq!(parse_duration("65"), Some(65));
        assert_eq!(parse_duration("1:05"), Some(65));
        assert_eq!(parse_duration("1:01:05"), Some(3665));
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(65), "01:05");
        assert_eq!(format_duration(3665), "01:01:05");
    }
}