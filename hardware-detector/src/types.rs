use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuInfo {
    pub brand: String,
    pub cores_physical: usize,
    pub cores_logical: usize,
    pub frequency_mhz: u64,
    pub cache_l1: u64,
    pub cache_l2: u64,
    pub cache_l3: u64,
    pub architecture: String,
    pub features: Vec<String>,
    pub thermal_max: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryInfo {
    pub total_bytes: u64,
    pub available_bytes: u64,
    pub swap_total: u64,
    pub swap_available: u64,
    pub memory_type: String,
    pub speed_mhz: Option<u64>,
    pub channels: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    pub vendor: GpuVendor,
    pub model: String,
    pub memory_total: u64,
    pub memory_available: u64,
    pub compute_capability: Option<String>,
    pub cuda_cores: Option<u32>,
    pub tensor_cores: Option<u32>,
    pub pcie_bandwidth: Option<u64>,
    pub power_limit: Option<u32>,
    pub temperature: Option<f32>,
    pub utilization: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GpuVendor {
    Nvidia,
    Amd,
    Intel,
    Apple,
    Unknown(String),
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum PerformanceTier {
    Low,
    Medium,
    High,
    Ultra,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalInfo {
    pub cpu_temperature: Option<f32>,
    pub gpu_temperature: Option<f32>,
    pub thermal_state: ThermalState,
    pub cooling_capability: CoolingCapability,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ThermalState {
    Optimal,
    Warm,
    Hot,
    Critical,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CoolingCapability {
    Passive,
    Active,
    Liquid,
    Custom,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    pub cpu_usage: f32,
    pub memory_usage: f32,
    pub gpu_usage: Option<f32>,
    pub thermal_info: ThermalInfo,
    pub power_consumption: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareProfile {
    pub cpu: CpuInfo,
    pub memory: MemoryInfo,
    pub gpus: Vec<GpuInfo>,
    pub performance_tier: PerformanceTier,
    pub thermal_info: ThermalInfo,
    pub system_metrics: SystemMetrics,
    pub detection_timestamp: u64,
}

impl HardwareProfile {
    /// Calculate comprehensive hardware performance score for AI video processing workloads
    /// Score ranges: 0-100+ (higher is better)
    /// Optimized for transcription, summarization, and video analysis tasks
    pub fn calculate_score(&self) -> f64 {
        let cpu_score = self.calculate_cpu_score();
        let memory_score = self.calculate_memory_score();
        let gpu_score = self.calculate_gpu_score();
        let thermal_penalty = self.calculate_thermal_penalty();
        let efficiency_bonus = self.calculate_efficiency_bonus();

        // Dynamic weighting based on workload requirements:
        // GPU-heavy workloads (AI inference): GPU 60%, CPU 25%, Memory 15%
        // CPU workloads get adjusted weighting when GPU is weak
        let gpu_weight = if gpu_score > 10.0 { 0.60 } else { 0.30 };
        let cpu_weight = if gpu_score > 10.0 { 0.25 } else { 0.50 };
        let memory_weight = if gpu_score > 10.0 { 0.15 } else { 0.20 };

        let base_score = (cpu_score * cpu_weight) + (memory_score * memory_weight) + (gpu_score * gpu_weight);
        let final_score = base_score * thermal_penalty + efficiency_bonus;

        // Normalize to reasonable range (0-100+)
        final_score.max(0.0).min(200.0)
    }

    /// Enhanced CPU scoring considering architecture, features, and cache hierarchy
    fn calculate_cpu_score(&self) -> f64 {
        // Base performance calculation
        let base_flops = (self.cpu.cores_logical as f64 * self.cpu.frequency_mhz as f64) / 100.0;

        // Architecture bonus
        let arch_multiplier = self.get_cpu_architecture_multiplier();

        // Cache scoring (L1 + L2 + L3 with diminishing returns)
        let cache_score = self.calculate_cache_score();

        // Feature bonus for SIMD instructions
        let feature_bonus = self.calculate_cpu_feature_bonus();

        // Physical vs logical core penalty for hyperthreading overhead
        let ht_penalty = if self.cpu.cores_physical < self.cpu.cores_logical {
            0.85 // 15% penalty for hyperthreading in CPU-intensive workloads
        } else {
            1.0
        };

        (base_flops * arch_multiplier + cache_score + feature_bonus) * ht_penalty
    }

    fn get_cpu_architecture_multiplier(&self) -> f64 {
        let brand_lower = self.cpu.brand.to_lowercase();

        // Apple Silicon (excellent performance per core)
        if brand_lower.contains("apple") {
            if brand_lower.contains("m3") { 2.2 }
            else if brand_lower.contains("m2") { 2.0 }
            else if brand_lower.contains("m1") { 1.8 }
            else { 1.5 }
        }
        // AMD Zen architectures
        else if brand_lower.contains("amd") || brand_lower.contains("ryzen") || brand_lower.contains("epyc") {
            if brand_lower.contains("7000") || brand_lower.contains("zen 4") { 1.7 }
            else if brand_lower.contains("5000") || brand_lower.contains("zen 3") { 1.6 }
            else if brand_lower.contains("3000") || brand_lower.contains("zen 2") { 1.4 }
            else { 1.2 }
        }
        // Intel architectures
        else if brand_lower.contains("intel") {
            if brand_lower.contains("13th gen") || brand_lower.contains("raptor") { 1.6 }
            else if brand_lower.contains("12th gen") || brand_lower.contains("alder") { 1.5 }
            else if brand_lower.contains("11th gen") || brand_lower.contains("rocket") { 1.3 }
            else if brand_lower.contains("10th gen") || brand_lower.contains("comet") { 1.2 }
            else { 1.0 }
        }
        else { 1.0 }
    }

    fn calculate_cache_score(&self) -> f64 {
        let l1_score = (self.cpu.cache_l1 as f64 / 1024.0).sqrt(); // KB to score
        let l2_score = (self.cpu.cache_l2 as f64 / 1024.0 / 1024.0).sqrt(); // MB to score
        let l3_score = (self.cpu.cache_l3 as f64 / 1024.0 / 1024.0).sqrt(); // MB to score

        // Weighted cache importance: L1 > L2 > L3 for latency-sensitive tasks
        (l1_score * 0.5) + (l2_score * 0.3) + (l3_score * 0.2)
    }

    fn calculate_cpu_feature_bonus(&self) -> f64 {
        let mut bonus = 0.0;
        for feature in &self.cpu.features {
            match feature.as_str() {
                "AVX512" => bonus += 3.0, // Excellent for AI workloads
                "AVX2" => bonus += 2.0,   // Good SIMD support
                "AVX" => bonus += 1.0,    // Basic SIMD
                "SSE4.2" => bonus += 0.5, // Legacy but useful
                "BMI2" => bonus += 0.5,   // Bit manipulation
                "AES" => bonus += 0.3,    // Hardware crypto
                _ => {}
            }
        }
        bonus
    }

    /// Enhanced memory scoring considering bandwidth, latency, and capacity
    fn calculate_memory_score(&self) -> f64 {
        let total_gb = self.memory.total_bytes as f64 / (1024.0 * 1024.0 * 1024.0);

        // Memory type scoring
        let type_multiplier = match self.memory.memory_type.as_str() {
            "DDR5" => 1.8,      // Latest generation
            "LPDDR5" => 1.9,    // Mobile, very efficient
            "DDR4" => 1.0,      // Current standard
            "LPDDR4" => 1.1,    // Mobile DDR4
            "DDR3" => 0.6,      // Older generation
            "Unified" => 2.2,   // Apple Silicon unified memory
            _ => 0.8,
        };

        // Speed factor (normalized to DDR4-2400)
        let speed_factor = self.memory.speed_mhz.unwrap_or(2400) as f64 / 2400.0;

        // Channel bonus for memory bandwidth
        let channel_bonus = match self.memory.channels.unwrap_or(2) {
            1 => 1.0,  // Single channel
            2 => 1.3,  // Dual channel (+30%)
            4 => 1.5,  // Quad channel (+50%)
            _ => 1.6,  // More channels
        };

        // Capacity scoring with sweet spots for AI workloads
        let capacity_score = if total_gb >= 64.0 {
            total_gb * 0.8  // Diminishing returns above 64GB
        } else if total_gb >= 32.0 {
            total_gb * 1.0  // Optimal range
        } else if total_gb >= 16.0 {
            total_gb * 1.2  // Good for most tasks
        } else {
            total_gb * 0.8  // Below recommended
        };

        capacity_score * speed_factor * channel_bonus * type_multiplier
    }

    /// Comprehensive GPU scoring for AI inference workloads
    fn calculate_gpu_score(&self) -> f64 {
        if self.gpus.is_empty() {
            return 0.0;
        }

        // Score the best GPU (most workloads use single GPU)
        self.gpus.iter()
            .map(|gpu| self.score_individual_gpu(gpu))
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0)
    }

    fn score_individual_gpu(&self, gpu: &GpuInfo) -> f64 {
        let memory_gb = gpu.memory_total as f64 / (1024.0 * 1024.0 * 1024.0);

        // Base compute score from CUDA cores or equivalent
        let compute_score = self.calculate_gpu_compute_score(gpu);

        // Memory score with bandwidth considerations
        let memory_score = self.calculate_gpu_memory_score(gpu, memory_gb);

        // Vendor-specific optimizations
        let vendor_multiplier = self.get_gpu_vendor_multiplier(gpu);

        // Architecture and generation bonuses
        let arch_bonus = self.get_gpu_architecture_bonus(gpu);

        // Thermal and power efficiency
        let efficiency_factor = self.get_gpu_efficiency_factor(gpu);

        (compute_score + memory_score + arch_bonus) * vendor_multiplier * efficiency_factor
    }

    fn calculate_gpu_compute_score(&self, gpu: &GpuInfo) -> f64 {
        match gpu.vendor {
            GpuVendor::Nvidia => {
                let cuda_cores = gpu.cuda_cores.unwrap_or(0) as f64;
                let tensor_cores = gpu.tensor_cores.unwrap_or(0) as f64;

                // Tensor cores are much more valuable for AI workloads
                (cuda_cores / 100.0) + (tensor_cores * 5.0)
            },
            GpuVendor::Apple => {
                // Apple Silicon GPU performance estimation based on model
                if gpu.model.contains("M3") { 25.0 }
                else if gpu.model.contains("M2 Max") { 20.0 }
                else if gpu.model.contains("M2 Pro") { 15.0 }
                else if gpu.model.contains("M2") { 12.0 }
                else if gpu.model.contains("M1 Max") { 18.0 }
                else if gpu.model.contains("M1 Pro") { 14.0 }
                else if gpu.model.contains("M1") { 10.0 }
                else { 8.0 }
            },
            GpuVendor::Amd => {
                // AMD compute units estimation
                if gpu.model.to_lowercase().contains("7900") { 35.0 }
                else if gpu.model.to_lowercase().contains("7800") { 30.0 }
                else if gpu.model.to_lowercase().contains("6900") { 25.0 }
                else { 15.0 }
            },
            GpuVendor::Intel => {
                // Intel Arc or integrated graphics
                if gpu.model.to_lowercase().contains("arc") { 12.0 }
                else { 5.0 } // Integrated graphics
            },
            GpuVendor::Unknown(_) => 3.0,
        }
    }

    fn calculate_gpu_memory_score(&self, gpu: &GpuInfo, memory_gb: f64) -> f64 {
        // Memory capacity score with AI workload requirements
        let capacity_score = if memory_gb >= 24.0 {
            memory_gb * 2.0  // Excellent for large models
        } else if memory_gb >= 16.0 {
            memory_gb * 2.5  // Sweet spot for most models
        } else if memory_gb >= 8.0 {
            memory_gb * 2.0  // Good for smaller models
        } else if memory_gb >= 4.0 {
            memory_gb * 1.5  // Limited utility
        } else {
            memory_gb * 0.5  // Very limited
        };

        // Memory bandwidth estimation based on GPU class
        let bandwidth_bonus = match gpu.vendor {
            GpuVendor::Nvidia => {
                if gpu.model.contains("4090") { 8.0 }
                else if gpu.model.contains("4080") { 6.0 }
                else if gpu.model.contains("A100") { 12.0 }
                else { 4.0 }
            },
            GpuVendor::Apple => 10.0, // Unified memory high bandwidth
            _ => 3.0,
        };

        capacity_score + bandwidth_bonus
    }

    fn get_gpu_vendor_multiplier(&self, gpu: &GpuInfo) -> f64 {
        match gpu.vendor {
            GpuVendor::Nvidia => 1.0,     // Reference standard
            GpuVendor::Apple => 1.1,      // Excellent efficiency
            GpuVendor::Amd => 0.75,       // Good but less AI optimization
            GpuVendor::Intel => 0.6,      // Newer to AI market
            GpuVendor::Unknown(_) => 0.4,
        }
    }

    fn get_gpu_architecture_bonus(&self, gpu: &GpuInfo) -> f64 {
        let model_lower = gpu.model.to_lowercase();

        // NVIDIA generation bonuses
        if model_lower.contains("50") || model_lower.contains("blackwell") { 5.0 }
        else if model_lower.contains("40") || model_lower.contains("ada") { 3.0 }
        else if model_lower.contains("30") || model_lower.contains("ampere") { 2.0 }
        else if model_lower.contains("h100") || model_lower.contains("hopper") { 8.0 }
        else if model_lower.contains("a100") { 6.0 }
        else { 0.0 }
    }

    fn get_gpu_efficiency_factor(&self, gpu: &GpuInfo) -> f64 {
        // Temperature penalty
        let temp_factor = match gpu.temperature {
            Some(temp) if temp > 85.0 => 0.8,  // Thermal throttling likely
            Some(temp) if temp > 75.0 => 0.9,  // Running hot
            Some(temp) if temp > 65.0 => 0.95, // Warm but acceptable
            _ => 1.0, // Cool or unknown
        };

        // Power efficiency bonus for modern architectures
        let efficiency_bonus = match gpu.vendor {
            GpuVendor::Apple => 1.2,    // Excellent power efficiency
            GpuVendor::Nvidia if gpu.model.contains("40") => 1.1, // Ada Lovelace efficiency
            _ => 1.0,
        };

        temp_factor * efficiency_bonus
    }

    /// Calculate thermal penalty based on system thermal state
    fn calculate_thermal_penalty(&self) -> f64 {
        match self.thermal_info.thermal_state {
            ThermalState::Optimal => 1.0,
            ThermalState::Warm => 0.95,
            ThermalState::Hot => 0.85,
            ThermalState::Critical => 0.7,
        }
    }

    /// Calculate efficiency bonus for well-cooled systems
    fn calculate_efficiency_bonus(&self) -> f64 {
        let cooling_bonus = match self.thermal_info.cooling_capability {
            CoolingCapability::Liquid => 3.0,
            CoolingCapability::Active => 1.0,
            CoolingCapability::Passive => 0.0,
            CoolingCapability::Custom => 2.0,
        };

        // Power efficiency bonus
        let power_efficiency = match self.system_metrics.power_consumption {
            Some(power) if power < 50.0 => 2.0,  // Very efficient
            Some(power) if power < 100.0 => 1.0, // Normal
            Some(power) if power < 200.0 => 0.5, // High power
            _ => 0.0, // Unknown or very high power
        };

        cooling_bonus + power_efficiency
    }
}