use anyhow::Result;
use image::{DynamicImage, ImageBuffer, Luma, Rgba, RgbaImage};
use qrcode::{QrCode, EcLevel};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// QR Code generator for mobile RSS feed access
pub struct QRCodeGenerator {
    /// Default error correction level
    ec_level: EcLevel,

    /// Default module size (pixel size of each QR module)
    module_size: u32,

    /// Default margin (quiet zone) around QR code
    margin: u32,

    /// Brand settings
    brand_config: Option<BrandConfig>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BrandConfig {
    /// Logo image to embed in center
    pub logo_path: Option<String>,

    /// Brand colors
    pub foreground_color: [u8; 4], // RGBA
    pub background_color: [u8; 4], // RGBA

    /// Text to display below QR code
    pub caption: Option<String>,

    /// Font settings for caption
    pub font_size: u32,
}

impl Default for QRCodeGenerator {
    fn default() -> Self {
        Self {
            ec_level: EcLevel::M, // Medium error correction (15% recovery)
            module_size: 8,
            margin: 4,
            brand_config: None,
        }
    }
}

impl QRCodeGenerator {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_brand(mut self, config: BrandConfig) -> Self {
        self.brand_config = Some(config);
        self
    }

    pub fn with_error_correction(mut self, level: EcLevel) -> Self {
        self.ec_level = level;
        self
    }

    /// Generate a basic QR code for a URL
    pub fn generate(&self, url: &str) -> Result<Vec<u8>> {
        let code = QrCode::with_error_correction_level(url, self.ec_level)?;
        let image = self.render_qr_code(&code)?;

        // Convert to PNG bytes
        let mut buffer = Vec::new();
        image.write_to(
            &mut std::io::Cursor::new(&mut buffer),
            image::ImageOutputFormat::Png,
        )?;

        Ok(buffer)
    }

    /// Generate a branded QR code with custom styling
    pub fn generate_branded(&self, url: &str) -> Result<Vec<u8>> {
        let code = QrCode::with_error_correction_level(url, self.ec_level)?;
        let mut image = self.render_qr_code_colored(&code)?;

        if let Some(config) = &self.brand_config {
            // Add logo if specified
            if let Some(logo_path) = &config.logo_path {
                image = self.embed_logo(image, logo_path)?;
            }

            // Add caption if specified
            if let Some(caption) = &config.caption {
                image = self.add_caption(image, caption, config.font_size)?;
            }
        }

        // Convert to PNG bytes
        let mut buffer = Vec::new();
        image.write_to(
            &mut std::io::Cursor::new(&mut buffer),
            image::ImageOutputFormat::Png,
        )?;

        Ok(buffer)
    }

    /// Generate QR code with RSS feed metadata
    pub fn generate_feed_qr(
        &self,
        feed_url: &str,
        feed_title: &str,
        platform: &str,
    ) -> Result<Vec<u8>> {
        // Create a deep link URL with metadata
        let deep_link = format!(
            "rss://subscribe?url={}&title={}&platform={}",
            urlencoding::encode(feed_url),
            urlencoding::encode(feed_title),
            urlencoding::encode(platform)
        );

        let code = QrCode::with_error_correction_level(&deep_link, self.ec_level)?;
        let mut image = self.render_qr_code_colored(&code)?;

        // Add platform-specific branding
        image = self.add_platform_branding(image, platform)?;

        // Add feed title as caption
        let caption = format!("{} - {}", platform, feed_title);
        image = self.add_caption(image, &caption, 14)?;

        let mut buffer = Vec::new();
        image.write_to(
            &mut std::io::Cursor::new(&mut buffer),
            image::ImageOutputFormat::Png,
        )?;

        Ok(buffer)
    }

    /// Generate a batch of QR codes for multiple feeds
    pub async fn generate_batch(
        &self,
        feeds: Vec<(String, String, String)>, // (url, title, platform)
    ) -> Result<Vec<(String, Vec<u8>)>> {
        use futures::stream::{self, StreamExt};

        let results = stream::iter(feeds)
            .map(|(url, title, platform)| async move {
                let qr_data = self.generate_feed_qr(&url, &title, &platform)?;
                Ok::<_, anyhow::Error>((title, qr_data))
            })
            .buffer_unordered(4)
            .collect::<Vec<_>>()
            .await;

        results.into_iter().collect()
    }

    /// Generate a WiFi QR code (for mobile app configuration)
    pub fn generate_wifi_qr(
        &self,
        ssid: &str,
        password: &str,
        security: &str,
    ) -> Result<Vec<u8>> {
        let wifi_string = format!(
            "WIFI:T:{};S:{};P:{};;",
            security.to_uppercase(),
            ssid,
            password
        );

        self.generate(&wifi_string)
    }

    /// Generate vCard QR code (for contact sharing)
    pub fn generate_vcard_qr(
        &self,
        name: &str,
        phone: &str,
        email: &str,
        url: &str,
    ) -> Result<Vec<u8>> {
        let vcard = format!(
            "BEGIN:VCARD\nVERSION:3.0\nFN:{}\nTEL:{}\nEMAIL:{}\nURL:{}\nEND:VCARD",
            name, phone, email, url
        );

        self.generate(&vcard)
    }

    // Private helper methods

    fn render_qr_code(&self, code: &QrCode) -> Result<DynamicImage> {
        let size = code.width();
        let img_size = size * self.module_size + 2 * self.margin;

        let mut image = ImageBuffer::new(img_size, img_size);

        // Fill background
        for pixel in image.pixels_mut() {
            *pixel = Luma([255u8]);
        }

        // Draw QR modules
        for y in 0..size {
            for x in 0..size {
                if code[(x, y)] {
                    let start_x = x * self.module_size + self.margin;
                    let start_y = y * self.module_size + self.margin;

                    for dy in 0..self.module_size {
                        for dx in 0..self.module_size {
                            image.put_pixel(
                                start_x + dx,
                                start_y + dy,
                                Luma([0u8]),
                            );
                        }
                    }
                }
            }
        }

        Ok(DynamicImage::ImageLuma8(image))
    }

    fn render_qr_code_colored(&self, code: &QrCode) -> Result<DynamicImage> {
        let size = code.width();
        let img_size = size * self.module_size + 2 * self.margin;

        let mut image = RgbaImage::new(img_size, img_size);

        let (fg, bg) = if let Some(config) = &self.brand_config {
            (
                Rgba(config.foreground_color),
                Rgba(config.background_color),
            )
        } else {
            (
                Rgba([0, 0, 0, 255]),       // Black
                Rgba([255, 255, 255, 255]),  // White
            )
        };

        // Fill background
        for pixel in image.pixels_mut() {
            *pixel = bg;
        }

        // Draw QR modules with rounded corners
        for y in 0..size {
            for x in 0..size {
                if code[(x, y)] {
                    self.draw_rounded_module(
                        &mut image,
                        x * self.module_size + self.margin,
                        y * self.module_size + self.margin,
                        self.module_size,
                        fg,
                    );
                }
            }
        }

        Ok(DynamicImage::ImageRgba8(image))
    }

    fn draw_rounded_module(
        &self,
        image: &mut RgbaImage,
        x: u32,
        y: u32,
        size: u32,
        color: Rgba<u8>,
    ) {
        let radius = size / 4;

        for dy in 0..size {
            for dx in 0..size {
                // Check if pixel is within rounded corners
                let in_corner = (dx < radius && dy < radius &&
                                Self::distance(dx, dy, radius, radius) > radius as f32) ||
                               (dx >= size - radius && dy < radius &&
                                Self::distance(dx, dy, size - radius - 1, radius) > radius as f32) ||
                               (dx < radius && dy >= size - radius &&
                                Self::distance(dx, dy, radius, size - radius - 1) > radius as f32) ||
                               (dx >= size - radius && dy >= size - radius &&
                                Self::distance(dx, dy, size - radius - 1, size - radius - 1) > radius as f32);

                if !in_corner {
                    image.put_pixel(x + dx, y + dy, color);
                }
            }
        }
    }

    fn distance(x1: u32, y1: u32, x2: u32, y2: u32) -> f32 {
        let dx = x1 as f32 - x2 as f32;
        let dy = y1 as f32 - y2 as f32;
        (dx * dx + dy * dy).sqrt()
    }

    fn embed_logo(&self, mut image: DynamicImage, logo_path: &str) -> Result<DynamicImage> {
        let logo = image::open(logo_path)?;

        // Calculate logo size (10% of QR code)
        let qr_size = image.width().min(image.height());
        let logo_size = qr_size / 10;

        // Resize logo
        let logo = logo.resize_exact(
            logo_size,
            logo_size,
            image::imageops::FilterType::Lanczos3,
        );

        // Calculate position (center)
        let x = (image.width() - logo_size) / 2;
        let y = (image.height() - logo_size) / 2;

        // Add white background behind logo
        let padding = 4;
        for dy in 0..logo_size + padding * 2 {
            for dx in 0..logo_size + padding * 2 {
                if x >= padding && y >= padding {
                    image.put_pixel(
                        x + dx - padding,
                        y + dy - padding,
                        Rgba([255, 255, 255, 255]),
                    );
                }
            }
        }

        // Overlay logo
        image::imageops::overlay(&mut image, &logo, x as i64, y as i64);

        Ok(image)
    }

    fn add_caption(&self, image: DynamicImage, caption: &str, font_size: u32) -> Result<DynamicImage> {
        // For simplicity, we'll just add padding and return
        // In production, you'd use a text rendering library like rusttype

        let width = image.width();
        let caption_height = font_size * 2;
        let new_height = image.height() + caption_height;

        let mut new_image = RgbaImage::new(width, new_height);

        // Fill background
        for pixel in new_image.pixels_mut() {
            *pixel = Rgba([255, 255, 255, 255]);
        }

        // Copy original image
        image::imageops::overlay(&mut DynamicImage::ImageRgba8(new_image), &image, 0, 0);

        // In production, render text here using rusttype
        // For now, we just have the white space for the caption

        Ok(DynamicImage::ImageRgba8(new_image.clone()))
    }

    fn add_platform_branding(&self, mut image: DynamicImage, platform: &str) -> Result<DynamicImage> {
        // Add platform-specific color overlay or watermark
        let color = match platform.to_lowercase().as_str() {
            "youtube" => Rgba([255, 0, 0, 50]),      // Red tint
            "bilibili" => Rgba([0, 161, 214, 50]),   // Blue tint
            "douyin" => Rgba([255, 0, 85, 50]),      // Pink tint
            "kuaishou" => Rgba([255, 119, 0, 50]),   // Orange tint
            _ => Rgba([0, 0, 0, 0]),                 // No tint
        };

        // Apply subtle color overlay
        if color[3] > 0 {
            for pixel in image.as_mut_rgba8().unwrap().pixels_mut() {
                let Rgba([r, g, b, a]) = *pixel;
                if a > 0 {
                    // Blend with platform color
                    let alpha = color[3] as f32 / 255.0;
                    pixel[0] = ((r as f32 * (1.0 - alpha)) + (color[0] as f32 * alpha)) as u8;
                    pixel[1] = ((g as f32 * (1.0 - alpha)) + (color[1] as f32 * alpha)) as u8;
                    pixel[2] = ((b as f32 * (1.0 - alpha)) + (color[2] as f32 * alpha)) as u8;
                }
            }
        }

        Ok(image)
    }
}

/// Generate QR code for a feed URL
pub fn generate_qr_code(url: &str) -> Result<Vec<u8>> {
    let generator = QRCodeGenerator::new();
    generator.generate(url)
}

/// Generate branded QR code with custom configuration
pub fn generate_branded_qr(url: &str, config: BrandConfig) -> Result<Vec<u8>> {
    let generator = QRCodeGenerator::new().with_brand(config);
    generator.generate_branded(url)
}

/// Generate QR code optimized for mobile RSS readers
pub fn generate_mobile_qr(feed_url: &str, feed_title: &str, platform: &str) -> Result<Vec<u8>> {
    let generator = QRCodeGenerator::new()
        .with_error_correction(EcLevel::H); // High error correction for mobile scanning

    generator.generate_feed_qr(feed_url, feed_title, platform)
}

/// Service for managing QR codes
pub struct QRCodeService {
    generator: Arc<QRCodeGenerator>,
    cache: Arc<crate::cache::CacheManager>,
}

impl QRCodeService {
    pub fn new(cache: Arc<crate::cache::CacheManager>) -> Self {
        Self {
            generator: Arc::new(QRCodeGenerator::new()),
            cache,
        }
    }

    /// Get or generate QR code for a feed
    pub async fn get_qr_code(&self, feed_id: &str, feed_url: &str) -> Result<Vec<u8>> {
        let cache_key = format!("qr:{}", feed_id);

        // Check cache
        if let Some(cached) = self.cache.get_feed(&cache_key).await {
            // Decode base64 cached data
            return Ok(base64::decode(cached)?);
        }

        // Generate new QR code
        let qr_data = self.generator.generate(feed_url)?;

        // Cache as base64 (QR codes don't change)
        let encoded = base64::encode(&qr_data);
        self.cache.set_feed(&cache_key, &encoded, 86400).await; // Cache for 24 hours

        Ok(qr_data)
    }

    /// Generate QR codes for multiple feeds
    pub async fn generate_batch(&self, feeds: Vec<(String, String)>) -> Result<Vec<Vec<u8>>> {
        let mut results = Vec::new();

        for (feed_id, feed_url) in feeds {
            let qr_data = self.get_qr_code(&feed_id, &feed_url).await?;
            results.push(qr_data);
        }

        Ok(results)
    }

    /// Generate a QR code sheet with multiple feeds
    pub async fn generate_qr_sheet(
        &self,
        feeds: Vec<(String, String, String)>, // (id, url, title)
        columns: u32,
    ) -> Result<Vec<u8>> {
        let qr_size = 200;
        let margin = 20;
        let title_height = 30;

        let rows = (feeds.len() as f32 / columns as f32).ceil() as u32;
        let sheet_width = columns * (qr_size + margin * 2);
        let sheet_height = rows * (qr_size + margin * 2 + title_height);

        let mut sheet = RgbaImage::new(sheet_width, sheet_height);

        // Fill white background
        for pixel in sheet.pixels_mut() {
            *pixel = Rgba([255, 255, 255, 255]);
        }

        // Generate and place QR codes
        for (idx, (feed_id, feed_url, title)) in feeds.iter().enumerate() {
            let col = (idx as u32) % columns;
            let row = (idx as u32) / columns;

            let x = col * (qr_size + margin * 2) + margin;
            let y = row * (qr_size + margin * 2 + title_height) + margin;

            // Generate QR code
            let qr_data = self.get_qr_code(feed_id, feed_url).await?;
            let qr_image = image::load_from_memory(&qr_data)?;

            // Resize to fit
            let qr_image = qr_image.resize_exact(
                qr_size,
                qr_size,
                image::imageops::FilterType::Lanczos3,
            );

            // Place on sheet
            image::imageops::overlay(
                &mut DynamicImage::ImageRgba8(sheet.clone()),
                &qr_image,
                x as i64,
                y as i64,
            );

            // Add title text (simplified - in production use rusttype)
            // For now, we just have space allocated for it
        }

        // Convert to PNG
        let mut buffer = Vec::new();
        DynamicImage::ImageRgba8(sheet).write_to(
            &mut std::io::Cursor::new(&mut buffer),
            image::ImageOutputFormat::Png,
        )?;

        Ok(buffer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qr_generation() {
        let generator = QRCodeGenerator::new();
        let result = generator.generate("https://example.com/feed.xml");
        assert!(result.is_ok());

        let data = result.unwrap();
        assert!(!data.is_empty());
    }

    #[test]
    fn test_wifi_qr() {
        let generator = QRCodeGenerator::new();
        let result = generator.generate_wifi_qr("MyNetwork", "password123", "WPA");
        assert!(result.is_ok());
    }

    #[test]
    fn test_vcard_qr() {
        let generator = QRCodeGenerator::new();
        let result = generator.generate_vcard_qr(
            "John Doe",
            "+1234567890",
            "john@example.com",
            "https://example.com"
        );
        assert!(result.is_ok());
    }
}