use anyhow::Result;
use clap::{Parser, Subcommand};
use rust_video_core::{
    VideoProcessor, VideoAnalysisConfig, GpuBackendType
};
use std::path::PathBuf;
use std::time::Instant;
use serde_json;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Process a video file
    Process {
        /// Path to the video file
        #[arg(short, long)]
        video: PathBuf,
        
        /// GPU backend to use (auto, cuda, rocm, openvino, metal, cpu)
        #[arg(short, long, default_value = "auto")]
        backend: String,
        
        /// Maximum number of frames to analyze
        #[arg(short, long, default_value = "100")]
        max_frames: usize,
        
        /// Output format (json, pretty)
        #[arg(short, long, default_value = "pretty")]
        output: String,
    },
    
    /// Transcribe audio from a video
    Transcribe {
        /// Path to the video file
        #[arg(short, long)]
        video: PathBuf,
        
        /// Language code (auto-detect if not specified)
        #[arg(short, long, default_value = "auto")]
        language: String,
    },
    
    /// List available GPU backends
    Backends,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    env_logger::init();
    
    let cli = Cli::parse();
    
    match cli.command {
        Commands::Process { video, backend, max_frames, output } => {
            println!("🎬 Processing video: {}", video.display());
            println!("🖥️  Using backend: {}", backend);
            
            let config = VideoAnalysisConfig {
                gpu_backend: parse_backend(&backend),
                max_frames,
                ..Default::default()
            };
            
            let start = Instant::now();
            let processor = VideoProcessor::new(config).await?;
            
            println!("⏳ Analyzing video...");
            let mut result = processor.process_video(&video).await?;
            
            let elapsed = start.elapsed();
            result.processing_time_ms = elapsed.as_millis() as u64;
            
            match output.as_str() {
                "json" => {
                    println!("{}", serde_json::to_string(&result)?);
                }
                _ => {
                    println!("\n✅ Analysis Complete\n");
                    println!("📹 Video: {}", result.video_path);
                    println!("⏱️  Duration: {:.2}s", result.duration);
                    println!("🎞️  FPS: {:.2}", result.fps);
                    println!("📐 Resolution: {}x{}", result.resolution.0, result.resolution.1);
                    println!("⚡ Processing time: {:.2}s", elapsed.as_secs_f64());
                    
                    println!("\n🎨 Visual Features:");
                    println!("  - Scenes detected: {}", result.visual_features.scene_count);
                    println!("  - Objects identified: {}", result.visual_features.detected_objects.len());
                    println!("  - Motion intensity: {:.2}", result.visual_features.motion_intensity);
                    
                    if let Some(transcript) = &result.transcript {
                        println!("\n🎤 Transcript:");
                        println!("  - Language: {}", transcript.language);
                        println!("  - Word count: {}", transcript.text.split_whitespace().count());
                        println!("  - Segments: {}", transcript.segments.len());
                        
                        if !transcript.text.is_empty() {
                            println!("\n📝 Text (first 500 chars):");
                            let preview: String = transcript.text.chars().take(500).collect();
                            println!("  {}", preview);
                            if transcript.text.len() > 500 {
                                println!("  ...");
                            }
                        }
                    }
                    
                    println!("\n📊 Summary:");
                    println!("  {}", result.summary);
                }
            }
        }
        
        Commands::Transcribe { video, language } => {
            println!("🎤 Transcribing audio from: {}", video.display());
            
            let config = VideoAnalysisConfig::default();
            let processor = VideoProcessor::new(config).await?;
            
            // Extract audio and transcribe
            let video_data = rust_video_core::video::extract_video_data(&video, 1).await?;
            
            if video_data.audio_buffer.is_empty() {
                println!("❌ No audio found in video");
                return Ok(());
            }
            
            let start = Instant::now();
            let transcript = processor.transcriber.transcribe(&video_data.audio_buffer).await?;
            let elapsed = start.elapsed();
            
            println!("\n✅ Transcription Complete\n");
            println!("🌐 Language: {}", transcript.language);
            println!("📝 Word count: {}", transcript.text.split_whitespace().count());
            println!("🎯 Segments: {}", transcript.segments.len());
            println!("⚡ Processing time: {:.2}s", elapsed.as_secs_f64());
            
            println!("\n📄 Full Transcript:\n");
            for segment in &transcript.segments {
                println!("[{:.2}s - {:.2}s] {}", 
                    segment.start, 
                    segment.end, 
                    segment.text
                );
            }
        }
        
        Commands::Backends => {
            println!("🖥️  Available GPU Backends:\n");
            
            #[cfg(feature = "cuda")]
            println!("  ✅ CUDA (NVIDIA GPUs)");
            
            #[cfg(feature = "rocm")]
            println!("  ✅ ROCm (AMD GPUs)");
            
            #[cfg(feature = "openvino")]
            println!("  ✅ OpenVINO (Intel GPUs/CPUs)");
            
            #[cfg(target_os = "macos")]
            println!("  ✅ Metal (Apple Silicon)");
            
            println!("  ✅ CPU (Fallback)");
            println!("  ✅ Auto (Automatic selection)");
            
            println!("\n💡 Use --backend flag to specify which backend to use");
        }
    }
    
    Ok(())
}

fn parse_backend(backend: &str) -> GpuBackendType {
    match backend.to_lowercase().as_str() {
        "cuda" => GpuBackendType::Cuda,
        "rocm" => GpuBackendType::Rocm,
        "openvino" => GpuBackendType::OpenVino,
        "metal" => GpuBackendType::Metal,
        "cpu" => GpuBackendType::Cpu,
        _ => GpuBackendType::Auto,
    }
}