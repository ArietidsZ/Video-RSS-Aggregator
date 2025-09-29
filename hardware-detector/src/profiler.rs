use crate::{
    cpu::CpuDetector,
    memory::MemoryDetector,
    gpu::GpuDetector,
    types::{HardwareProfile, PerformanceTier, ThermalInfo, ThermalState, CoolingCapability, SystemMetrics},
    HardwareError, Result,
};
use sysinfo::System;
use std::time::{SystemTime, UNIX_EPOCH};

pub struct HardwareProfiler {
    cpu_detector: CpuDetector,
    memory_detector: MemoryDetector,
    gpu_detector: GpuDetector,
    system: System,
}

impl HardwareProfiler {
    pub fn new() -> Self {
        let mut system = System::new_all();
        system.refresh_all();

        Self {
            cpu_detector: CpuDetector::new(),
            memory_detector: MemoryDetector::new(),
            gpu_detector: GpuDetector::new(),
            system,
        }
    }

    pub fn detect_hardware(&mut self) -> Result<HardwareProfile> {
        tracing::info!("Starting hardware detection...");

        // Detect CPU information
        let cpu = self.cpu_detector.detect_cpu_info()
            .map_err(|e| HardwareError::CpuDetection(format!("CPU detection failed: {}", e)))?;
        tracing::debug!("CPU detected: {} cores, {} MHz", cpu.cores_logical, cpu.frequency_mhz);

        // Detect memory information
        let memory = self.memory_detector.detect_memory_info()
            .map_err(|e| HardwareError::MemoryDetection(format!("Memory detection failed: {}", e)))?;
        tracing::debug!("Memory detected: {} GB total", memory.total_bytes / (1024 * 1024 * 1024));

        // Detect GPU information
        let gpus = self.gpu_detector.detect_gpus()
            .map_err(|e| HardwareError::GpuDetection(format!("GPU detection failed: {}", e)))?;
        tracing::debug!("GPUs detected: {}", gpus.len());

        // Get thermal information
        let thermal_info = self.get_thermal_info(&cpu, &gpus)?;

        // Get current system metrics
        let system_metrics = self.get_system_metrics(&thermal_info)?;

        // Calculate performance tier
        let profile = HardwareProfile {
            cpu: cpu.clone(),
            memory: memory.clone(),
            gpus: gpus.clone(),
            performance_tier: PerformanceTier::Medium, // Temporary
            thermal_info: thermal_info.clone(),
            system_metrics,
            detection_timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };

        let performance_tier = self.calculate_performance_tier(&profile);

        let final_profile = HardwareProfile {
            performance_tier,
            ..profile
        };

        tracing::info!("Hardware detection completed. Performance tier: {:?}", final_profile.performance_tier);

        Ok(final_profile)
    }

    fn get_thermal_info(&mut self, cpu: &crate::types::CpuInfo, gpus: &[crate::types::GpuInfo]) -> Result<ThermalInfo> {
        self.system.refresh_cpu_all();

        // Get CPU temperature (platform-specific)
        let cpu_temperature = self.get_cpu_temperature();

        // Get GPU temperature from the first GPU
        let gpu_temperature = gpus.first().and_then(|gpu| gpu.temperature);

        // Determine thermal state
        let thermal_state = self.determine_thermal_state(cpu_temperature, gpu_temperature);

        // Estimate cooling capability
        let cooling_capability = self.estimate_cooling_capability(&thermal_state, cpu, gpus);

        Ok(ThermalInfo {
            cpu_temperature,
            gpu_temperature,
            thermal_state,
            cooling_capability,
        })
    }

    fn get_cpu_temperature(&mut self) -> Option<f32> {
        // Platform-specific temperature reading
        #[cfg(target_os = "linux")]
        {
            // Try reading from thermal zone
            if let Ok(temp_str) = std::fs::read_to_string("/sys/class/thermal/thermal_zone0/temp") {
                if let Ok(temp_millidegrees) = temp_str.trim().parse::<i32>() {
                    return Some(temp_millidegrees as f32 / 1000.0);
                }
            }

            // Try sensors command
            if let Ok(output) = std::process::Command::new("sensors")
                .args(&["-u"])
                .output()
            {
                let output_str = String::from_utf8_lossy(&output.stdout);
                return self.parse_sensors_output(&output_str);
            }
        }

        #[cfg(target_os = "macos")]
        {
            // macOS temperature reading would require IOKit or external tools
            // For now, return None
        }

        #[cfg(target_os = "windows")]
        {
            // Windows temperature reading would require WMI
            // For now, return None
        }

        None
    }

    #[cfg(target_os = "linux")]
    fn parse_sensors_output(&self, output: &str) -> Option<f32> {
        for line in output.lines() {
            if line.contains("_input:") && (line.contains("temp") || line.contains("Core")) {
                let parts: Vec<&str> = line.split(':').collect();
                if parts.len() >= 2 {
                    if let Ok(temp) = parts[1].trim().parse::<f32>() {
                        return Some(temp);
                    }
                }
            }
        }
        None
    }

    fn determine_thermal_state(&self, cpu_temp: Option<f32>, gpu_temp: Option<f32>) -> ThermalState {
        let max_temp = match (cpu_temp, gpu_temp) {
            (Some(cpu), Some(gpu)) => cpu.max(gpu),
            (Some(cpu), None) => cpu,
            (None, Some(gpu)) => gpu,
            (None, None) => return ThermalState::Optimal, // Unknown, assume optimal
        };

        if max_temp > 85.0 {
            ThermalState::Critical
        } else if max_temp > 75.0 {
            ThermalState::Hot
        } else if max_temp > 65.0 {
            ThermalState::Warm
        } else {
            ThermalState::Optimal
        }
    }

    fn estimate_cooling_capability(&self, thermal_state: &ThermalState, cpu: &crate::types::CpuInfo, gpus: &[crate::types::GpuInfo]) -> CoolingCapability {
        // Estimate based on hardware characteristics and thermal state
        let high_performance_system = cpu.cores_logical > 8 ||
            gpus.iter().any(|gpu| gpu.memory_total > 8 * 1024 * 1024 * 1024); // > 8GB VRAM

        match thermal_state {
            ThermalState::Optimal => {
                if high_performance_system {
                    CoolingCapability::Liquid
                } else {
                    CoolingCapability::Active
                }
            }
            ThermalState::Warm => CoolingCapability::Active,
            ThermalState::Hot => CoolingCapability::Passive,
            ThermalState::Critical => CoolingCapability::Passive,
        }
    }

    fn get_system_metrics(&mut self, thermal_info: &ThermalInfo) -> Result<SystemMetrics> {
        self.system.refresh_cpu_all();
        self.system.refresh_memory();

        // Calculate average CPU usage
        let cpu_usage = self.system.cpus().iter()
            .map(|cpu| cpu.cpu_usage())
            .sum::<f32>() / self.system.cpus().len() as f32;

        // Calculate memory usage
        let memory_usage = ((self.system.total_memory() - self.system.available_memory()) as f32 / self.system.total_memory() as f32) * 100.0;

        // GPU usage would come from GPU-specific APIs
        let gpu_usage = None; // TODO: Implement GPU utilization reading

        // Power consumption estimation (very rough)
        let power_consumption = self.estimate_power_consumption(cpu_usage, memory_usage, gpu_usage);

        Ok(SystemMetrics {
            cpu_usage,
            memory_usage,
            gpu_usage,
            thermal_info: thermal_info.clone(),
            power_consumption,
        })
    }

    fn estimate_power_consumption(&self, cpu_usage: f32, _memory_usage: f32, gpu_usage: Option<f32>) -> Option<f32> {
        // Very rough power estimation
        let base_power = 50.0; // Base system power
        let cpu_power = (cpu_usage / 100.0) * 65.0; // CPU under load
        let gpu_power = gpu_usage.map_or(0.0, |usage| (usage / 100.0) * 200.0); // GPU under load

        Some(base_power + cpu_power + gpu_power)
    }

    fn calculate_performance_tier(&self, profile: &HardwareProfile) -> PerformanceTier {
        let score = profile.calculate_score();

        // Performance tier thresholds based on the scoring algorithm
        if score >= 50.0 {
            PerformanceTier::Ultra
        } else if score >= 25.0 {
            PerformanceTier::High
        } else if score >= 10.0 {
            PerformanceTier::Medium
        } else {
            PerformanceTier::Low
        }
    }

    pub async fn monitor_system(&mut self, duration_secs: u64) -> Result<Vec<SystemMetrics>> {
        let mut metrics = Vec::new();
        let start_time = std::time::Instant::now();

        while start_time.elapsed().as_secs() < duration_secs {
            // Refresh system information
            self.system.refresh_cpu_all();
            self.system.refresh_memory();

            // Get thermal info
            let cpu_info = self.cpu_detector.detect_cpu_info()?;
            let gpu_info = self.gpu_detector.detect_gpus()?;
            let thermal_info = self.get_thermal_info(&cpu_info, &gpu_info)?;

            // Get current metrics
            let current_metrics = self.get_system_metrics(&thermal_info)?;
            metrics.push(current_metrics);

            // Wait 1 second before next measurement
            tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        }

        Ok(metrics)
    }

    pub fn benchmark_performance(&mut self) -> Result<f64> {
        tracing::info!("Starting performance benchmark...");

        let start = std::time::Instant::now();

        // Simple CPU benchmark - calculate primes
        let prime_count = self.calculate_primes_up_to(100000);

        let cpu_benchmark_time = start.elapsed().as_millis() as f64;

        // Memory bandwidth test
        let memory_benchmark = self.memory_bandwidth_test();

        // Combined score (lower is better for CPU time, higher is better for memory bandwidth)
        let performance_score = (1000.0 / cpu_benchmark_time) + (memory_benchmark / 1000.0);

        tracing::info!("Performance benchmark completed. Score: {:.2}, Primes: {}, Time: {}ms",
                      performance_score, prime_count, cpu_benchmark_time);

        Ok(performance_score)
    }

    fn calculate_primes_up_to(&self, limit: usize) -> usize {
        let mut is_prime = vec![true; limit + 1];
        is_prime[0] = false;
        if limit > 0 {
            is_prime[1] = false;
        }

        for i in 2..=((limit as f64).sqrt() as usize) {
            if is_prime[i] {
                for j in ((i * i)..=limit).step_by(i) {
                    is_prime[j] = false;
                }
            }
        }

        is_prime.iter().filter(|&&x| x).count()
    }

    fn memory_bandwidth_test(&self) -> f64 {
        let size = 10_000_000; // 10MB
        let data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();

        let start = std::time::Instant::now();

        // Simple memory copy operation
        let copied_data: Vec<u8> = data.iter().map(|&x| x.wrapping_add(1)).collect();

        let elapsed = start.elapsed().as_nanos() as f64;

        // Calculate bandwidth in MB/s
        let bandwidth_mbps = (size as f64 * 2.0) / (elapsed / 1_000_000_000.0) / (1024.0 * 1024.0);

        // Prevent optimization from removing the calculation
        std::hint::black_box(copied_data);

        bandwidth_mbps
    }
}

impl Default for HardwareProfiler {
    fn default() -> Self {
        Self::new()
    }
}