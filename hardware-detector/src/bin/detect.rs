use hardware_detector::{HardwareProfiler, config::ConfigGenerator, PerformanceAdjuster};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "hardware_detector=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    println!("🔍 Video RSS Aggregator Hardware Detection");
    println!("=========================================");

    let mut profiler = HardwareProfiler::new();

    println!("🔎 Detecting hardware configuration...");
    let profile = profiler.detect_hardware()?;

    println!("\n📊 Hardware Profile:");
    println!("-------------------");

    // CPU Information
    println!("🖥️  CPU: {}", profile.cpu.brand);
    println!("   Cores: {} physical, {} logical", profile.cpu.cores_physical, profile.cpu.cores_logical);
    println!("   Frequency: {} MHz", profile.cpu.frequency_mhz);
    println!("   Cache: L1: {}KB, L2: {}KB, L3: {}KB",
        profile.cpu.cache_l1 / 1024,
        profile.cpu.cache_l2 / 1024,
        profile.cpu.cache_l3 / 1024
    );
    println!("   Architecture: {}", profile.cpu.architecture);
    println!("   Features: {}", profile.cpu.features.join(", "));

    // Memory Information
    println!("\n💾 Memory:");
    println!("   Total: {:.1} GB", profile.memory.total_bytes as f64 / (1024.0 * 1024.0 * 1024.0));
    println!("   Available: {:.1} GB", profile.memory.available_bytes as f64 / (1024.0 * 1024.0 * 1024.0));
    println!("   Type: {}", profile.memory.memory_type);
    if let Some(speed) = profile.memory.speed_mhz {
        println!("   Speed: {} MHz", speed);
    }
    if let Some(channels) = profile.memory.channels {
        println!("   Channels: {}", channels);
    }

    // GPU Information
    println!("\n🎮 GPUs ({}):", profile.gpus.len());
    for (i, gpu) in profile.gpus.iter().enumerate() {
        println!("   GPU {}: {}", i, gpu.model);
        println!("      Vendor: {:?}", gpu.vendor);
        if gpu.memory_total > 0 {
            println!("      Memory: {:.1} GB", gpu.memory_total as f64 / (1024.0 * 1024.0 * 1024.0));
        }
        if let Some(compute) = &gpu.compute_capability {
            println!("      Compute Capability: {}", compute);
        }
        if let Some(cuda_cores) = gpu.cuda_cores {
            println!("      CUDA Cores: {}", cuda_cores);
        }
        if let Some(tensor_cores) = gpu.tensor_cores {
            println!("      Tensor Cores: {}", tensor_cores);
        }
        if let Some(temp) = gpu.temperature {
            println!("      Temperature: {:.1}°C", temp);
        }
        if let Some(util) = gpu.utilization {
            println!("      Utilization: {:.1}%", util);
        }
    }

    // Performance Information
    println!("\n⚡ Performance:");
    println!("   Tier: {:?}", profile.performance_tier);
    println!("   Score: {:.2}", profile.calculate_score());
    println!("   Thermal State: {:?}", profile.thermal_info.thermal_state);
    println!("   Cooling: {:?}", profile.thermal_info.cooling_capability);

    // System Metrics
    println!("\n📈 Current Metrics:");
    println!("   CPU Usage: {:.1}%", profile.system_metrics.cpu_usage);
    println!("   Memory Usage: {:.1}%", profile.system_metrics.memory_usage);
    if let Some(gpu_usage) = profile.system_metrics.gpu_usage {
        println!("   GPU Usage: {:.1}%", gpu_usage);
    }
    if let Some(power) = profile.system_metrics.power_consumption {
        println!("   Estimated Power: {:.0}W", power);
    }

    // Generate optimal configuration
    println!("\n⚙️  Generating optimal configuration...");
    let config = ConfigGenerator::generate_optimal_config(&profile)?;

    println!("\n🎯 Optimal Configuration:");
    println!("========================");

    // Transcription Configuration
    println!("\n📝 Transcription:");
    println!("   Model: {:?} ({:?})", config.transcription.model_size, config.transcription.model_variant);
    println!("   Quantization: {:?}", config.transcription.quantization);
    println!("   Batch Size: {}", config.transcription.batch_size);
    println!("   Chunk Length: {}ms", config.transcription.chunk_length_ms);
    println!("   Language: {:?}", config.transcription.language_optimization);
    println!("   Beam Size: {}", config.transcription.beam_size);
    println!("   CUDA Graphs: {}", config.transcription.enable_cuda_graphs);
    println!("   Kernel Fusion: {}", config.transcription.enable_kernel_fusion);
    println!("   Flash Attention: {}", config.transcription.enable_flash_attention);

    // Summarization Configuration
    println!("\n📄 Summarization:");
    println!("   Model: {}", config.summarization.model_name);
    println!("   Quantization: {:?}", config.summarization.quantization);
    println!("   Max Length: {}", config.summarization.max_length);
    println!("   Context Window: {}", config.summarization.context_window);
    println!("   Multimodal: {}", config.summarization.enable_multimodal);
    println!("   Temperature: {}", config.summarization.temperature);
    println!("   Top-P: {}", config.summarization.top_p);

    // Memory Configuration
    println!("\n💾 Memory:");
    println!("   Transcription Buffer: {} MB", config.memory.transcription_buffer_mb);
    println!("   Summarization Buffer: {} MB", config.memory.summarization_buffer_mb);
    println!("   Cache Size: {} MB", config.memory.cache_size_mb);
    println!("   Prefetch Size: {} MB", config.memory.prefetch_size_mb);
    println!("   Memory Mapping: {}", config.memory.enable_memory_mapping);
    println!("   Unified Memory: {}", config.memory.enable_unified_memory);

    // GPU Configuration
    println!("\n🎮 GPU:");
    if let Some(device_id) = config.gpu.device_id {
        println!("   Device ID: {}", device_id);
    }
    println!("   Memory Fraction: {:.1}%", config.gpu.memory_fraction * 100.0);
    println!("   Mixed Precision: {}", config.gpu.enable_mixed_precision);
    println!("   Optimization: {:?}", config.gpu.optimization_level);
    println!("   TensorRT: {}", config.gpu.tensor_rt_enabled);
    println!("   cuDNN Benchmark: {}", config.gpu.cudnn_benchmark);
    println!("   Compute Mode: {:?}", config.gpu.compute_mode);

    // Performance Configuration
    println!("\n⚡ Performance:");
    println!("   Worker Threads: {}", config.performance.worker_threads);
    println!("   Parallel Processing: {}", config.performance.enable_parallel_processing);
    println!("   Priority: {:?}", config.performance.priority_class);
    println!("   CPU Governor: {:?}", config.performance.cpu_governor);
    println!("   I/O Priority: {:?}", config.performance.io_priority);
    println!("   Thermal Throttling: {}", config.performance.thermal_throttling);
    if let Some(power_limit) = config.performance.power_limit_watts {
        println!("   Power Limit: {}W", power_limit);
    }

    // Video Processing Configuration
    println!("\n🎬 Video Processing:");
    println!("   Hardware Decode: {}", config.video_processing.enable_hardware_decode);
    println!("   Decoder: {:?}", config.video_processing.decoder_type);
    println!("   GPU Preprocessing: {}", config.video_processing.enable_gpu_preprocessing);
    println!("   Frame Rate: {} fps", config.video_processing.frame_sampling_rate);
    println!("   Scene Detection: {}", config.video_processing.enable_scene_detection);

    // AI Acceleration Configuration
    println!("\n🚀 AI Acceleration:");
    println!("   TensorCore: {}", config.ai_acceleration.enable_tensorcore);
    println!("   XNNPACK: {}", config.ai_acceleration.enable_xnnpack);
    println!("   ONNX Runtime: {}", config.ai_acceleration.enable_onnx_runtime);
    println!("   OpenVINO: {}", config.ai_acceleration.enable_openvino);
    println!("   CoreML: {}", config.ai_acceleration.enable_coreml);
    if !config.ai_acceleration.optimization_passes.is_empty() {
        println!("   Optimization Passes: {}", config.ai_acceleration.optimization_passes.join(", "));
    }

    // Validate configuration
    println!("\n✅ Validating configuration...");
    match ConfigGenerator::validate_config(&config, &profile) {
        Ok(()) => println!("   Configuration is valid!"),
        Err(e) => println!("   ⚠️  Configuration warning: {}", e),
    }

    // Save configuration
    let config_path = "optimal_config.json";
    println!("\n💾 Saving configuration to {}...", config_path);
    ConfigGenerator::save_config_to_file(&config, config_path)?;
    println!("   Configuration saved successfully!");

    // Performance benchmark
    println!("\n🚀 Running performance benchmark...");
    let benchmark_score = profiler.benchmark_performance()?;
    println!("   Benchmark Score: {:.2}", benchmark_score);

    // Demonstrate thermal monitoring and dynamic adjustment
    println!("\n🌡️  Testing thermal monitoring and dynamic adjustment...");
    let mut performance_adjuster = PerformanceAdjuster::new();

    // Get initial thermal state and adjustments
    let adjustments = performance_adjuster.update(&profile)?;

    println!("\n📊 Thermal Monitoring:");
    println!("-------------------");

    let thermal_monitor = performance_adjuster.get_thermal_monitor();
    if let Some(last_reading) = thermal_monitor.get_thermal_history().last() {
        println!("   CPU Temperature: {:.1}°C", last_reading.cpu_temp);
        if let Some(gpu_temp) = last_reading.gpu_temp {
            println!("   GPU Temperature: {:.1}°C", gpu_temp);
        }
        println!("   Thermal State: {:?}", last_reading.thermal_state);
        println!("   Throttle Level: {:.0}%", last_reading.throttle_percentage * 100.0);
    }

    println!("\n⚙️  Dynamic Adjustments:");
    println!("   Batch Size: {:.0}%", adjustments.batch_size_multiplier * 100.0);
    println!("   Frequency: {:.0}%", adjustments.frequency_multiplier * 100.0);
    println!("   Memory: {:.0}%", adjustments.memory_multiplier * 100.0);
    println!("   Threads: {:.0}%", adjustments.thread_multiplier * 100.0);
    println!("   Quality: {:.0}%", adjustments.quality_multiplier * 100.0);
    println!("   Reason: {}", adjustments.reason);

    // Example of applying adjustments to configuration
    println!("\n🔧 Applied Adjustments Example:");
    let base_batch_size = config.transcription.batch_size;
    let adjusted_batch_size = performance_adjuster.apply_to_batch_size(base_batch_size);
    println!("   Base Batch Size: {} → Adjusted: {}", base_batch_size, adjusted_batch_size);

    let base_threads = config.performance.worker_threads;
    let adjusted_threads = performance_adjuster.apply_to_thread_count(base_threads);
    println!("   Base Threads: {} → Adjusted: {}", base_threads, adjusted_threads);

    let base_memory = config.memory.transcription_buffer_mb;
    let adjusted_memory = performance_adjuster.apply_to_memory_allocation(base_memory);
    println!("   Base Memory: {} MB → Adjusted: {} MB", base_memory, adjusted_memory);

    println!("   Quality Level: {:?}", performance_adjuster.get_quality_level());

    println!("\n✅ Hardware detection and configuration completed!");
    println!("   Detection took: {}ms",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs() - profile.detection_timestamp
    );

    Ok(())
}