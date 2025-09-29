use crate::{types::{HardwareProfile, PerformanceTier, GpuVendor, ThermalState}, HardwareError, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimalConfig {
    pub transcription: TranscriptionConfig,
    pub summarization: SummarizationConfig,
    pub memory: MemoryConfig,
    pub gpu: GpuConfig,
    pub performance: PerformanceConfig,
    pub video_processing: VideoProcessingConfig,
    pub ai_acceleration: AIAccelerationConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionConfig {
    pub model_size: ModelSize,
    pub model_variant: ModelVariant,
    pub quantization: QuantizationType,
    pub batch_size: u32,
    pub chunk_length_ms: u32,
    pub enable_cuda_graphs: bool,
    pub enable_kernel_fusion: bool,
    pub enable_flash_attention: bool,
    pub language_optimization: LanguageOptimization,
    pub beam_size: u32,
    pub vad_threshold: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummarizationConfig {
    pub model_name: String,
    pub quantization: QuantizationType,
    pub max_length: u32,
    pub enable_multimodal: bool,
    pub context_window: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub repetition_penalty: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    pub transcription_buffer_mb: u32,
    pub summarization_buffer_mb: u32,
    pub cache_size_mb: u32,
    pub enable_memory_mapping: bool,
    pub enable_unified_memory: bool,
    pub swap_threshold_mb: u32,
    pub prefetch_size_mb: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuConfig {
    pub device_id: Option<u32>,
    pub memory_fraction: f32,
    pub enable_mixed_precision: bool,
    pub optimization_level: OptimizationLevel,
    pub tensor_rt_enabled: bool,
    pub cudnn_benchmark: bool,
    pub memory_growth: bool,
    pub compute_mode: ComputeMode,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    pub worker_threads: u32,
    pub enable_parallel_processing: bool,
    pub priority_class: PriorityClass,
    pub thermal_throttling: bool,
    pub power_limit_watts: Option<u32>,
    pub cpu_governor: CpuGovernor,
    pub io_priority: IoPriority,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoProcessingConfig {
    pub enable_hardware_decode: bool,
    pub decoder_type: DecoderType,
    pub enable_gpu_preprocessing: bool,
    pub frame_sampling_rate: u32,
    pub enable_scene_detection: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AIAccelerationConfig {
    pub enable_tensorcore: bool,
    pub enable_xnnpack: bool,
    pub enable_onnx_runtime: bool,
    pub enable_openvino: bool,
    pub enable_coreml: bool,
    pub optimization_passes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelSize {
    Tiny,
    Base,
    Small,
    Medium,
    Large,
    Turbo,
    Ultra,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelVariant {
    WhisperStandard,
    WhisperTurbo,
    FasterWhisper,
    Paraformer,
    FunASR,
    Qwen3ASR,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantizationType {
    None,
    FP32,
    FP16,
    BF16,
    INT8,
    FP8,
    INT4,
    Dynamic,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationLevel {
    Conservative,
    Balanced,
    Aggressive,
    Maximum,
    Experimental,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PriorityClass {
    Low,
    Normal,
    High,
    RealTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LanguageOptimization {
    None,
    English,
    Chinese,
    Multilingual,
    AutoDetect,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComputeMode {
    Default,
    Exclusive,
    ProhibitedThreads,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CpuGovernor {
    Powersave,
    Ondemand,
    Performance,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IoPriority {
    Low,
    Normal,
    High,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DecoderType {
    Software,
    NVDEC,
    QuickSync,
    VideoToolbox,
    VAAPI,
    AMF,
}

pub struct ConfigGenerator;

impl ConfigGenerator {
    pub fn generate_optimal_config(profile: &HardwareProfile) -> Result<OptimalConfig> {
        // Calculate comprehensive hardware score for better decision making
        let hw_score = profile.calculate_score();

        let transcription = Self::generate_transcription_config(profile, hw_score)?;
        let summarization = Self::generate_summarization_config(profile, hw_score)?;
        let memory = Self::generate_memory_config(profile, hw_score)?;
        let gpu = Self::generate_gpu_config(profile, hw_score)?;
        let performance = Self::generate_performance_config(profile, hw_score)?;
        let video_processing = Self::generate_video_processing_config(profile)?;
        let ai_acceleration = Self::generate_ai_acceleration_config(profile)?;

        Ok(OptimalConfig {
            transcription,
            summarization,
            memory,
            gpu,
            performance,
            video_processing,
            ai_acceleration,
        })
    }

    fn generate_transcription_config(profile: &HardwareProfile, hw_score: f64) -> Result<TranscriptionConfig> {
        // Advanced model selection based on hardware score and capabilities
        let (model_size, model_variant, quantization) = Self::select_optimal_transcription_model(profile, hw_score);

        // Use intelligent batch size optimization
        let batch_size = Self::optimize_batch_size(profile, &model_variant);

        // Determine chunk length based on memory and processing power
        let chunk_length_ms = if hw_score > 80.0 { 10000 } // 10 seconds for high-end
                            else if hw_score > 40.0 { 5000 }  // 5 seconds for mid-range
                            else { 3000 }; // 3 seconds for low-end

        // CUDA-specific optimizations
        let (enable_cuda_graphs, enable_kernel_fusion) = Self::determine_cuda_optimizations(profile);

        // Flash Attention for transformer models (requires Ampere+ or Apple Silicon)
        let enable_flash_attention = Self::supports_flash_attention(profile);

        // Language-specific optimizations
        let language_optimization = Self::determine_language_optimization(profile);

        // Beam search size based on hardware capability
        let beam_size = if hw_score > 60.0 { 5 } else if hw_score > 30.0 { 3 } else { 1 };

        // VAD threshold tuning
        let vad_threshold = 0.5; // Standard threshold, could be tuned based on use case

        Ok(TranscriptionConfig {
            model_size,
            model_variant,
            quantization,
            batch_size,
            chunk_length_ms,
            enable_cuda_graphs,
            enable_kernel_fusion,
            enable_flash_attention,
            language_optimization,
            beam_size,
            vad_threshold,
        })
    }

    fn select_optimal_transcription_model(profile: &HardwareProfile, hw_score: f64) -> (ModelSize, ModelVariant, QuantizationType) {
        // First, check GPU capabilities for model variant selection
        let has_nvidia_gpu = profile.gpus.iter().any(|gpu| matches!(gpu.vendor, GpuVendor::Nvidia));
        let has_tensor_cores = profile.gpus.iter().any(|gpu| gpu.tensor_cores.unwrap_or(0) > 0);
        let gpu_memory_gb = profile.gpus.first()
            .map(|gpu| gpu.memory_total as f64 / (1024.0 * 1024.0 * 1024.0))
            .unwrap_or(0.0);

        // Model variant selection based on hardware
        let model_variant = if hw_score > 100.0 && has_nvidia_gpu {
            ModelVariant::WhisperTurbo // Fastest for high-end NVIDIA
        } else if hw_score > 60.0 {
            ModelVariant::FasterWhisper // Good balance
        } else {
            ModelVariant::WhisperStandard // Conservative choice
        };

        // Model size and quantization selection based on score and memory
        let (model_size, quantization) = match hw_score {
            score if score > 150.0 => {
                // Ultra high-end: Can run largest models with minimal quantization
                if gpu_memory_gb >= 24.0 {
                    (ModelSize::Ultra, QuantizationType::FP16)
                } else if gpu_memory_gb >= 16.0 {
                    (ModelSize::Large, QuantizationType::BF16)
                } else {
                    (ModelSize::Large, QuantizationType::INT8)
                }
            },
            score if score > 100.0 => {
                // High-end: Large models with smart quantization
                if gpu_memory_gb >= 12.0 {
                    (ModelSize::Large, if has_tensor_cores { QuantizationType::FP16 } else { QuantizationType::INT8 })
                } else {
                    (ModelSize::Turbo, QuantizationType::INT8)
                }
            },
            score if score > 60.0 => {
                // Mid-high: Turbo models with INT8 quantization
                (ModelSize::Turbo, QuantizationType::INT8)
            },
            score if score > 40.0 => {
                // Mid-range: Medium models with aggressive quantization
                (ModelSize::Medium, QuantizationType::INT8)
            },
            score if score > 20.0 => {
                // Low-mid: Small models with INT8
                (ModelSize::Small, QuantizationType::INT8)
            },
            _ => {
                // Low-end: Tiny models with maximum quantization
                (ModelSize::Tiny, QuantizationType::FP8)
            }
        };

        (model_size, model_variant, quantization)
    }

    fn generate_summarization_config(profile: &HardwareProfile, hw_score: f64) -> Result<SummarizationConfig> {
        // Select summarization model based on hardware capabilities
        let (model_name, quantization, max_length) = Self::select_optimal_summarization_model(profile, hw_score);

        // Enable multimodal for high-performance systems
        let enable_multimodal = hw_score > 60.0 && profile.gpus.iter()
            .any(|gpu| gpu.memory_total > 8 * 1024 * 1024 * 1024);

        // Context window based on available memory
        let context_window = if hw_score > 80.0 { max_length * 4 }
                           else if hw_score > 40.0 { max_length * 2 }
                           else { max_length };

        // Generation parameters tuning
        let temperature = 0.7; // Standard temperature for balanced creativity
        let top_p = 0.9; // Nucleus sampling threshold
        let repetition_penalty = 1.2; // Avoid repetitive text

        Ok(SummarizationConfig {
            model_name,
            quantization,
            max_length,
            enable_multimodal,
            context_window,
            temperature,
            top_p,
            repetition_penalty,
        })
    }

    fn select_optimal_summarization_model(profile: &HardwareProfile, hw_score: f64) -> (String, QuantizationType, u32) {
        let gpu_memory_gb = profile.gpus.first()
            .map(|gpu| gpu.memory_total as f64 / (1024.0 * 1024.0 * 1024.0))
            .unwrap_or(0.0);

        match hw_score {
            score if score > 100.0 && gpu_memory_gb >= 16.0 => {
                // High-end: Large multilingual models
                ("google/mt5-xl".to_string(), QuantizationType::FP16, 2048)
            },
            score if score > 80.0 && gpu_memory_gb >= 12.0 => {
                // High-performance: Large models with optimization
                ("google/mt5-large".to_string(), QuantizationType::BF16, 1024)
            },
            score if score > 60.0 => {
                // Mid-high: Base models with INT8
                ("google/mt5-base".to_string(), QuantizationType::INT8, 512)
            },
            score if score > 40.0 => {
                // Mid-range: Small models quantized
                ("google/mt5-small".to_string(), QuantizationType::INT8, 256)
            },
            _ => {
                // Low-end: Smallest models with aggressive quantization
                ("google/mt5-small".to_string(), QuantizationType::FP8, 128)
            }
        }
    }

    fn generate_memory_config(profile: &HardwareProfile, hw_score: f64) -> Result<MemoryConfig> {
        // Determine model variant for memory calculation
        let has_nvidia_gpu = profile.gpus.iter().any(|gpu| matches!(gpu.vendor, GpuVendor::Nvidia));
        let model_variant = if hw_score > 100.0 && has_nvidia_gpu {
            ModelVariant::WhisperTurbo
        } else if hw_score > 60.0 {
            ModelVariant::FasterWhisper
        } else {
            ModelVariant::WhisperStandard
        };

        // Get optimized batch size
        let batch_size = Self::optimize_batch_size(profile, &model_variant);

        // Use intelligent memory allocation
        let memory_config = Self::calculate_memory_allocation(profile, &model_variant, batch_size);

        Ok(memory_config)
    }

    fn generate_gpu_config(profile: &HardwareProfile, hw_score: f64) -> Result<GpuConfig> {
        let primary_gpu = profile.gpus.first();
        let device_id = primary_gpu.map(|_| 0u32);

        // Memory fraction based on score and thermal state
        let base_memory_fraction = match hw_score {
            score if score > 100.0 => 0.95,
            score if score > 60.0 => 0.85,
            score if score > 40.0 => 0.75,
            _ => 0.65,
        };

        // Adjust for thermal conditions
        let memory_fraction = match profile.thermal_info.thermal_state {
            crate::types::ThermalState::Hot | crate::types::ThermalState::Critical => base_memory_fraction * 0.9,
            _ => base_memory_fraction,
        };

        // Mixed precision for compatible GPUs
        let enable_mixed_precision = primary_gpu
            .map(|gpu| {
                matches!(gpu.vendor, GpuVendor::Nvidia) &&
                gpu.tensor_cores.unwrap_or(0) > 0 &&
                gpu.memory_total > 6 * 1024 * 1024 * 1024
            })
            .unwrap_or(false);

        // Optimization level based on hardware capability
        let optimization_level = match hw_score {
            score if score > 150.0 => OptimizationLevel::Experimental,
            score if score > 100.0 => OptimizationLevel::Maximum,
            score if score > 60.0 => OptimizationLevel::Aggressive,
            score if score > 40.0 => OptimizationLevel::Balanced,
            _ => OptimizationLevel::Conservative,
        };

        // TensorRT for NVIDIA GPUs with sufficient capabilities
        let tensor_rt_enabled = primary_gpu
            .map(|gpu| matches!(gpu.vendor, GpuVendor::Nvidia) && gpu.memory_total > 4 * 1024 * 1024 * 1024)
            .unwrap_or(false);

        // cuDNN benchmark for finding optimal convolution algorithms
        let cudnn_benchmark = tensor_rt_enabled && hw_score > 60.0;

        // Memory growth for dynamic allocation
        let memory_growth = hw_score < 60.0; // Enable for lower-end to avoid OOM

        // Compute mode for dedicated GPU usage
        let compute_mode = if hw_score > 100.0 { ComputeMode::Exclusive }
                         else { ComputeMode::Default };

        Ok(GpuConfig {
            device_id,
            memory_fraction,
            enable_mixed_precision,
            optimization_level,
            tensor_rt_enabled,
            cudnn_benchmark,
            memory_growth,
            compute_mode,
        })
    }

    fn generate_performance_config(profile: &HardwareProfile, hw_score: f64) -> Result<PerformanceConfig> {
        // Smart thread allocation based on workload characteristics
        let worker_threads = Self::calculate_optimal_thread_count(profile, hw_score);

        // Parallel processing for multi-core systems
        let enable_parallel_processing = profile.cpu.cores_logical > 4 && hw_score > 30.0;

        // Priority class based on performance tier and thermal state
        let priority_class = match (hw_score, profile.thermal_info.thermal_state) {
            (score, _) if score > 100.0 => PriorityClass::High,
            (score, crate::types::ThermalState::Hot) if score > 60.0 => PriorityClass::Normal,
            (score, _) if score > 60.0 => PriorityClass::High,
            _ => PriorityClass::Normal,
        };

        // Thermal throttling detection
        let thermal_throttling = matches!(profile.thermal_info.thermal_state,
            crate::types::ThermalState::Hot | crate::types::ThermalState::Critical);

        // Power limit based on thermal state and cooling capability
        let power_limit_watts = match (profile.thermal_info.thermal_state, profile.thermal_info.cooling_capability) {
            (crate::types::ThermalState::Critical, _) => Some(100),
            (crate::types::ThermalState::Hot, crate::types::CoolingCapability::Passive) => Some(150),
            (crate::types::ThermalState::Hot, _) => Some(200),
            _ => None,
        };

        // CPU governor for power management
        let cpu_governor = if hw_score > 80.0 { CpuGovernor::Performance }
                         else if hw_score > 40.0 { CpuGovernor::Ondemand }
                         else { CpuGovernor::Powersave };

        // I/O priority for disk operations
        let io_priority = if hw_score > 60.0 { IoPriority::High }
                        else { IoPriority::Normal };

        Ok(PerformanceConfig {
            worker_threads,
            enable_parallel_processing,
            priority_class,
            thermal_throttling,
            power_limit_watts,
            cpu_governor,
            io_priority,
        })
    }

    fn generate_video_processing_config(profile: &HardwareProfile) -> Result<VideoProcessingConfig> {
        // Determine hardware decode support
        let (enable_hardware_decode, decoder_type) = Self::determine_hardware_decoder(profile);

        // GPU preprocessing for video frames
        let enable_gpu_preprocessing = profile.gpus.iter()
            .any(|gpu| gpu.memory_total > 4 * 1024 * 1024 * 1024);

        // Frame sampling rate based on performance
        let frame_sampling_rate = match profile.performance_tier {
            PerformanceTier::Ultra => 30,
            PerformanceTier::High => 15,
            PerformanceTier::Medium => 10,
            PerformanceTier::Low => 5,
        };

        // Scene detection for intelligent chunking
        let enable_scene_detection = matches!(profile.performance_tier,
            PerformanceTier::Ultra | PerformanceTier::High);

        Ok(VideoProcessingConfig {
            enable_hardware_decode,
            decoder_type,
            enable_gpu_preprocessing,
            frame_sampling_rate,
            enable_scene_detection,
        })
    }

    fn generate_ai_acceleration_config(profile: &HardwareProfile) -> Result<AIAccelerationConfig> {
        let primary_gpu = profile.gpus.first();

        // TensorCore acceleration for NVIDIA GPUs
        let enable_tensorcore = primary_gpu
            .map(|gpu| matches!(gpu.vendor, GpuVendor::Nvidia) && gpu.tensor_cores.unwrap_or(0) > 0)
            .unwrap_or(false);

        // XNNPACK for CPU optimization
        let enable_xnnpack = !enable_tensorcore; // Use when no GPU acceleration

        // ONNX Runtime for cross-platform inference
        let enable_onnx_runtime = true; // Generally beneficial

        // OpenVINO for Intel hardware
        let enable_openvino = primary_gpu
            .map(|gpu| matches!(gpu.vendor, GpuVendor::Intel))
            .unwrap_or(false) || profile.cpu.brand.contains("Intel");

        // CoreML for Apple Silicon
        let enable_coreml = profile.cpu.brand.contains("Apple");

        // Optimization passes based on hardware capabilities
        let mut optimization_passes = Vec::new();
        if enable_tensorcore {
            optimization_passes.push("tensorcore_fusion".to_string());
        }
        if profile.calculate_score() > 60.0 {
            optimization_passes.push("graph_optimization".to_string());
            optimization_passes.push("constant_folding".to_string());
            optimization_passes.push("dead_code_elimination".to_string());
        }

        Ok(AIAccelerationConfig {
            enable_tensorcore,
            enable_xnnpack,
            enable_onnx_runtime,
            enable_openvino,
            enable_coreml,
            optimization_passes,
        })
    }


    fn calculate_optimal_thread_count(profile: &HardwareProfile, hw_score: f64) -> u32 {
        let logical_cores = profile.cpu.cores_logical as u32;
        let physical_cores = profile.cpu.cores_physical as u32;

        // Consider hyperthreading effectiveness
        let has_efficient_ht = logical_cores > physical_cores && hw_score > 60.0;

        let thread_count = match hw_score {
            score if score > 100.0 => {
                if has_efficient_ht {
                    logical_cores.min(32) // Use all logical cores up to 32
                } else {
                    physical_cores.min(24) // Use physical cores
                }
            },
            score if score > 60.0 => {
                if has_efficient_ht {
                    (logical_cores * 3 / 4).min(16)
                } else {
                    (physical_cores * 3 / 4).min(12)
                }
            },
            score if score > 40.0 => {
                (physical_cores / 2).max(2).min(8)
            },
            _ => {
                (physical_cores / 4).max(2).min(4)
            }
        };

        thread_count
    }

    fn determine_cuda_optimizations(profile: &HardwareProfile) -> (bool, bool) {
        let primary_gpu = profile.gpus.first();

        let has_cuda = primary_gpu
            .map(|gpu| matches!(gpu.vendor, GpuVendor::Nvidia))
            .unwrap_or(false);

        if !has_cuda {
            return (false, false);
        }

        let gpu_memory = primary_gpu.map(|gpu| gpu.memory_total).unwrap_or(0);
        let has_tensor_cores = primary_gpu.map(|gpu| gpu.tensor_cores.unwrap_or(0) > 0).unwrap_or(false);

        // CUDA graphs require sufficient memory and modern GPU
        let enable_cuda_graphs = gpu_memory > 6 * 1024 * 1024 * 1024 && has_tensor_cores;

        // Kernel fusion for reducing kernel launch overhead
        let enable_kernel_fusion = gpu_memory > 4 * 1024 * 1024 * 1024;

        (enable_cuda_graphs, enable_kernel_fusion)
    }

    fn supports_flash_attention(profile: &HardwareProfile) -> bool {
        profile.gpus.iter().any(|gpu| {
            match gpu.vendor {
                GpuVendor::Nvidia => {
                    // Requires Ampere (30xx) or newer
                    gpu.model.contains("30") || gpu.model.contains("40") ||
                    gpu.model.contains("A100") || gpu.model.contains("H100")
                },
                GpuVendor::Apple => true, // Apple Silicon supports optimized attention
                _ => false,
            }
        })
    }

    fn determine_language_optimization(profile: &HardwareProfile) -> LanguageOptimization {
        // For now, default to auto-detect
        // Could be enhanced with user preferences or content analysis
        LanguageOptimization::AutoDetect
    }

    fn determine_hardware_decoder(profile: &HardwareProfile) -> (bool, DecoderType) {
        let primary_gpu = profile.gpus.first();

        if let Some(gpu) = primary_gpu {
            match gpu.vendor {
                GpuVendor::Nvidia if gpu.memory_total > 2 * 1024 * 1024 * 1024 => {
                    (true, DecoderType::NVDEC)
                },
                GpuVendor::Intel => {
                    (true, DecoderType::QuickSync)
                },
                GpuVendor::Amd => {
                    (true, DecoderType::AMF)
                },
                GpuVendor::Apple => {
                    (true, DecoderType::VideoToolbox)
                },
                _ => (false, DecoderType::Software),
            }
        } else if cfg!(target_os = "linux") {
            // Try VAAPI on Linux
            (true, DecoderType::VAAPI)
        } else {
            (false, DecoderType::Software)
        }
    }

    pub fn optimize_batch_size(profile: &HardwareProfile, model_variant: &ModelVariant) -> u32 {
        // Calculate optimal batch size based on hardware capabilities
        // Goal: maximize throughput while ensuring single-stream GPU saturation

        let base_batch_size = match model_variant {
            ModelVariant::WhisperStandard => 4,
            ModelVariant::WhisperTurbo => 1,  // Turbo optimized for single-stream
            ModelVariant::FasterWhisper => 2,
            ModelVariant::Paraformer => 8,    // Efficient batch processing
            ModelVariant::FunASR => 6,
            ModelVariant::Qwen3ASR => 4,
        };

        // Adjust based on GPU memory
        let gpu_memory_gb = profile.gpus.first()
            .map(|gpu| gpu.memory_total as f64 / (1024.0 * 1024.0 * 1024.0))
            .unwrap_or(0.0);

        let memory_multiplier = if gpu_memory_gb >= 24.0 {
            2.0  // High-end GPU
        } else if gpu_memory_gb >= 12.0 {
            1.5  // Mid-range GPU
        } else if gpu_memory_gb >= 8.0 {
            1.0  // Entry-level GPU
        } else {
            0.5  // CPU-only or very limited GPU
        };

        // Consider CPU cores for parallel preprocessing
        let cpu_multiplier = if profile.cpu.cores_physical >= 16 {
            1.25
        } else if profile.cpu.cores_physical >= 8 {
            1.0
        } else {
            0.75
        };

        // Apply thermal constraints
        let thermal_multiplier = match profile.thermal_info.thermal_state {
            ThermalState::Optimal => 1.0,
            ThermalState::Warm => 0.8,
            ThermalState::Hot => 0.5,
            ThermalState::Critical => 0.25,
        };

        let optimal_batch = (base_batch_size as f64 * memory_multiplier * cpu_multiplier * thermal_multiplier) as u32;

        // Clamp to reasonable limits
        optimal_batch.max(1).min(32)
    }

    pub fn calculate_memory_allocation(profile: &HardwareProfile, model_variant: &ModelVariant, batch_size: u32) -> MemoryConfig {
        let total_memory_mb = (profile.memory.total_bytes / (1024 * 1024)) as u32;
        let available_memory_mb = (profile.memory.available_bytes / (1024 * 1024)) as u32;

        // Reserve 10% for OS and other processes
        let usable_memory_mb = ((available_memory_mb as f32) * 0.9) as u32;

        // Model memory requirements (estimated in MB)
        let model_base_memory = match model_variant {
            ModelVariant::WhisperStandard => 1500,
            ModelVariant::WhisperTurbo => 800,
            ModelVariant::FasterWhisper => 1200,
            ModelVariant::Paraformer => 1000,
            ModelVariant::FunASR => 900,
            ModelVariant::Qwen3ASR => 2000,
        };

        // Scale by batch size
        let transcription_memory = model_base_memory * batch_size;

        // Summarization typically needs 30% of transcription memory
        let summarization_memory = (transcription_memory as f32 * 0.3) as u32;

        // Cache should be 20% of available memory for optimal performance
        let cache_memory = (usable_memory_mb as f32 * 0.2) as u32;

        // Prefetch buffer for streaming
        let prefetch_memory = if profile.memory.total_bytes > 16 * 1024 * 1024 * 1024 {
            256  // Large prefetch for high-memory systems
        } else {
            128  // Standard prefetch
        };

        // Calculate swap threshold (when to start using disk)
        let swap_threshold = total_memory_mb * 10;  // Allow up to 10x overcommit

        // Enable memory mapping for large files on systems with sufficient RAM
        let enable_memory_mapping = profile.memory.total_bytes > 8 * 1024 * 1024 * 1024;

        // Unified memory for NVIDIA GPUs with sufficient VRAM
        let enable_unified = profile.gpus.iter().any(|gpu| {
            matches!(gpu.vendor, GpuVendor::Nvidia) && gpu.memory_total > 8 * 1024 * 1024 * 1024
        });

        MemoryConfig {
            transcription_buffer_mb: transcription_memory.min(usable_memory_mb / 2),
            summarization_buffer_mb: summarization_memory.min(usable_memory_mb / 3),
            cache_size_mb: cache_memory,
            enable_memory_mapping,
            enable_unified_memory: enable_unified,
            swap_threshold_mb: swap_threshold,
            prefetch_size_mb: prefetch_memory,
        }
    }

    pub fn validate_config(config: &OptimalConfig, profile: &HardwareProfile) -> Result<()> {
        // Validate memory requirements with dynamic allocation
        let required_memory = config.memory.transcription_buffer_mb +
            config.memory.summarization_buffer_mb +
            config.memory.cache_size_mb +
            config.memory.prefetch_size_mb;

        let available_memory_mb = profile.memory.available_bytes / (1024 * 1024);

        // Allow overcommit up to swap threshold
        let max_allowed = config.memory.swap_threshold_mb.min(available_memory_mb as u32 * 2);

        if required_memory > max_allowed {
            return Err(HardwareError::ConfigGeneration(
                format!("Memory requirement ({} MB) exceeds maximum allowed ({} MB)",
                    required_memory, max_allowed)
            ));
        }

        // Validate GPU memory requirements
        if let Some(gpu) = profile.gpus.first() {
            let gpu_memory_mb = gpu.memory_total / (1024 * 1024);
            let required_gpu_memory = (gpu_memory_mb as f32 * config.gpu.memory_fraction) as u64;

            if required_gpu_memory > gpu_memory_mb {
                return Err(HardwareError::ConfigGeneration(
                    format!("GPU memory requirement ({} MB) exceeds available GPU memory ({} MB)",
                        required_gpu_memory, gpu_memory_mb)
                ));
            }
        }

        // Validate worker thread count
        if config.performance.worker_threads > profile.cpu.cores_logical as u32 * 2 {
            return Err(HardwareError::ConfigGeneration(
                format!("Worker threads ({}) exceeds reasonable limit for {} logical cores",
                    config.performance.worker_threads, profile.cpu.cores_logical)
            ));
        }

        // Validate quantization compatibility
        if matches!(config.transcription.quantization, QuantizationType::FP8 | QuantizationType::INT4) {
            let has_advanced_quantization = profile.gpus.iter().any(|gpu| {
                matches!(gpu.vendor, GpuVendor::Nvidia) && gpu.tensor_cores.unwrap_or(0) > 0
            });

            if !has_advanced_quantization {
                return Err(HardwareError::ConfigGeneration(
                    "Advanced quantization (FP8/INT4) requires NVIDIA GPU with tensor cores".to_string()
                ));
            }
        }

        Ok(())
    }

    pub fn save_config_to_file(config: &OptimalConfig, path: &str) -> Result<()> {
        let json = serde_json::to_string_pretty(config)
            .map_err(|e| HardwareError::ConfigGeneration(format!("Failed to serialize config: {}", e)))?;

        std::fs::write(path, json)
            .map_err(|e| HardwareError::ConfigGeneration(format!("Failed to write config file: {}", e)))?;

        Ok(())
    }

    pub fn load_config_from_file(path: &str) -> Result<OptimalConfig> {
        let json = std::fs::read_to_string(path)
            .map_err(|e| HardwareError::ConfigGeneration(format!("Failed to read config file: {}", e)))?;

        let config: OptimalConfig = serde_json::from_str(&json)
            .map_err(|e| HardwareError::ConfigGeneration(format!("Failed to parse config: {}", e)))?;

        Ok(config)
    }
}