use anyhow::{Result, Context};
use ffmpeg_next as ffmpeg;
use std::path::Path;
use tokio::task;
use tracing::{info, debug};

use crate::gpu::VideoFrame;

pub struct VideoData {
    pub frames: Vec<VideoFrame>,
    pub audio_buffer: Vec<f32>,
    pub duration: f64,
    pub fps: f64,
    pub resolution: (u32, u32),
}

pub async fn extract_video_data(video_path: &Path, max_frames: usize) -> Result<VideoData> {
    let path_str = video_path.to_string_lossy().to_string();

    task::spawn_blocking(move || {
        extract_video_data_sync(&path_str, max_frames)
    })
    .await
    .context("Failed to spawn blocking task")?
}

fn extract_video_data_sync(video_path: &str, max_frames: usize) -> Result<VideoData> {
    ffmpeg::init().context("Failed to initialize FFmpeg")?;

    let mut input_context = ffmpeg::format::input(&video_path)
        .context("Failed to open video file")?;

    let video_stream = input_context
        .streams()
        .best(ffmpeg::media::Type::Video)
        .ok_or_else(|| anyhow::anyhow!("No video stream found"))?;

    let video_stream_index = video_stream.index();

    let audio_stream = input_context
        .streams()
        .best(ffmpeg::media::Type::Audio);

    let audio_stream_index = audio_stream.as_ref().map(|s| s.index());

    // Get video metadata
    let fps = video_stream.avg_frame_rate();
    let fps_f64 = fps.0 as f64 / fps.1.max(1) as f64;
    let duration = video_stream.duration() as f64 *
        f64::from(video_stream.time_base().0) / f64::from(video_stream.time_base().1);

    // Create decoders
    let context_decoder = ffmpeg::codec::context::Context::from_parameters(video_stream.parameters())?;
    let mut video_decoder = context_decoder.decoder().video()?;

    let resolution = (video_decoder.width(), video_decoder.height());

    let mut audio_decoder = if let Some(audio_stream) = audio_stream {
        let context = ffmpeg::codec::context::Context::from_parameters(audio_stream.parameters())?;
        Some(context.decoder().audio()?)
    } else {
        None
    };

    // Extract frames and audio
    let mut frames = Vec::new();
    let mut audio_buffer = Vec::new();
    let frame_skip = (video_stream.frames() as usize / max_frames).max(1);
    let mut frame_count = 0;

    info!(
        "Extracting video data: {}x{} @ {:.2} fps, duration: {:.2}s",
        resolution.0, resolution.1, fps_f64, duration
    );

    for (stream, packet) in input_context.packets() {
        if stream.index() == video_stream_index {
            video_decoder.send_packet(&packet)?;

            let mut decoded = ffmpeg::util::frame::video::Video::empty();
            while video_decoder.receive_frame(&mut decoded).is_ok() {
                if frame_count % frame_skip == 0 && frames.len() < max_frames {
                    frames.push(convert_frame_to_rgb(&decoded)?);
                }
                frame_count += 1;
            }
        } else if Some(stream.index()) == audio_stream_index {
            if let Some(ref mut decoder) = audio_decoder {
                decoder.send_packet(&packet)?;

                let mut decoded = ffmpeg::util::frame::audio::Audio::empty();
                while decoder.receive_frame(&mut decoded).is_ok() {
                    audio_buffer.extend(convert_audio_to_f32(&decoded)?);
                }
            }
        }
    }

    // Flush decoders
    video_decoder.send_eof()?;
    let mut decoded = ffmpeg::util::frame::video::Video::empty();
    while video_decoder.receive_frame(&mut decoded).is_ok() {
        if frame_count % frame_skip == 0 && frames.len() < max_frames {
            frames.push(convert_frame_to_rgb(&decoded)?);
        }
        frame_count += 1;
    }

    if let Some(ref mut decoder) = audio_decoder {
        decoder.send_eof()?;
        let mut decoded = ffmpeg::util::frame::audio::Audio::empty();
        while decoder.receive_frame(&mut decoded).is_ok() {
            audio_buffer.extend(convert_audio_to_f32(&decoded)?);
        }
    }

    info!(
        "Extracted {} frames and {} audio samples",
        frames.len(),
        audio_buffer.len()
    );

    Ok(VideoData {
        frames,
        audio_buffer,
        duration,
        fps: fps_f64,
        resolution,
    })
}

fn convert_frame_to_rgb(frame: &ffmpeg::util::frame::video::Video) -> Result<VideoFrame> {
    let mut rgb_frame = ffmpeg::util::frame::video::Video::empty();

    // Create scaler to convert to RGB
    let mut scaler = ffmpeg::software::scaling::context::Context::get(
        frame.format(),
        frame.width(),
        frame.height(),
        ffmpeg::format::Pixel::RGB24,
        frame.width(),
        frame.height(),
        ffmpeg::software::scaling::flag::Flags::BILINEAR,
    )?;

    scaler.run(frame, &mut rgb_frame)?;

    // Convert to Vec<u8>
    let data = rgb_frame.data(0).to_vec();
    let timestamp = frame.timestamp().unwrap_or(0) as f64;

    Ok(VideoFrame {
        data,
        width: frame.width(),
        height: frame.height(),
        timestamp,
    })
}

fn convert_audio_to_f32(frame: &ffmpeg::util::frame::audio::Audio) -> Result<Vec<f32>> {
    let mut resampler = ffmpeg::software::resampling::context::Context::get(
        frame.format(),
        frame.channel_layout(),
        frame.rate(),
        ffmpeg::format::Sample::F32(ffmpeg::format::sample::Type::Planar),
        ffmpeg::channel_layout::ChannelLayout::MONO,
        16000, // Whisper expects 16kHz
    )?;

    let mut resampled = ffmpeg::util::frame::audio::Audio::empty();
    resampler.run(frame, &mut resampled)?;

    // Convert to f32 vector
    let plane = resampled.plane::<f32>(0);
    Ok(plane.to_vec())
}

// Hardware-accelerated video decoding support
pub struct HardwareDecoder {
    hw_type: HardwareAccelType,
}

#[derive(Debug, Clone, Copy)]
pub enum HardwareAccelType {
    Cuda,
    Vaapi,
    Vdpau,
    Qsv,
    Dxva2,
    D3d11va,
    Videotoolbox,
}

impl HardwareDecoder {
    pub fn new(hw_type: HardwareAccelType) -> Self {
        Self { hw_type }
    }

    pub fn is_available(&self) -> bool {
        // Check if hardware acceleration is available
        match self.hw_type {
            HardwareAccelType::Cuda => check_cuda_available(),
            HardwareAccelType::Videotoolbox => cfg!(target_os = "macos"),
            _ => false,
        }
    }

    pub fn get_hw_config_string(&self) -> &str {
        match self.hw_type {
            HardwareAccelType::Cuda => "cuda",
            HardwareAccelType::Vaapi => "vaapi",
            HardwareAccelType::Vdpau => "vdpau",
            HardwareAccelType::Qsv => "qsv",
            HardwareAccelType::Dxva2 => "dxva2",
            HardwareAccelType::D3d11va => "d3d11va",
            HardwareAccelType::Videotoolbox => "videotoolbox",
        }
    }
}

fn check_cuda_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        // Check for NVDEC availability
        true // Simplified - would check actual CUDA availability
    }
    #[cfg(not(feature = "cuda"))]
    false
}

// Batch frame processing for efficiency
pub fn process_frames_batch(frames: &[VideoFrame], batch_size: usize) -> Vec<Vec<VideoFrame>> {
    frames
        .chunks(batch_size)
        .map(|chunk| chunk.to_vec())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hardware_decoder_availability() {
        let cuda_decoder = HardwareDecoder::new(HardwareAccelType::Cuda);
        // This will depend on the system
        let _ = cuda_decoder.is_available();

        let vt_decoder = HardwareDecoder::new(HardwareAccelType::Videotoolbox);
        #[cfg(target_os = "macos")]
        assert!(vt_decoder.is_available());
        #[cfg(not(target_os = "macos"))]
        assert!(!vt_decoder.is_available());
    }

    #[test]
    fn test_batch_processing() {
        let frames = vec![
            VideoFrame {
                data: vec![0; 100],
                width: 10,
                height: 10,
                timestamp: 0.0,
            },
            VideoFrame {
                data: vec![1; 100],
                width: 10,
                height: 10,
                timestamp: 1.0,
            },
            VideoFrame {
                data: vec![2; 100],
                width: 10,
                height: 10,
                timestamp: 2.0,
            },
        ];

        let batches = process_frames_batch(&frames, 2);
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].len(), 2);
        assert_eq!(batches[1].len(), 1);
    }
}