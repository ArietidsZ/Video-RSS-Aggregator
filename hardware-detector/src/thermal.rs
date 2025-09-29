use crate::{types::{ThermalInfo, ThermalState, CoolingCapability, HardwareProfile}, HardwareError, Result};
use sysinfo::{System, Components, CpuRefreshKind};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use serde::{Serialize, Deserialize};

pub struct ThermalMonitor {
    system: Arc<Mutex<System>>,
    components: Arc<Mutex<Components>>,
    thermal_history: Vec<ThermalReading>,
    config_adjustments: HashMap<String, f32>,
    last_update: Instant,
    throttle_threshold: f32,
    critical_threshold: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalReading {
    pub timestamp: u64,
    pub cpu_temp: f32,
    pub gpu_temp: Option<f32>,
    pub thermal_state: ThermalState,
    pub throttle_percentage: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicAdjustment {
    pub batch_size_multiplier: f32,
    pub frequency_multiplier: f32,
    pub memory_multiplier: f32,
    pub thread_multiplier: f32,
    pub quality_multiplier: f32,
    pub reason: String,
}

impl ThermalMonitor {
    pub fn new() -> Self {
        let system = System::new_all();
        let components = Components::new_with_refreshed_list();

        Self {
            system: Arc::new(Mutex::new(system)),
            components: Arc::new(Mutex::new(components)),
            thermal_history: Vec::with_capacity(60), // Store 60 readings (1 minute at 1Hz)
            config_adjustments: HashMap::new(),
            last_update: Instant::now(),
            throttle_threshold: 85.0,  // Start throttling at 85°C
            critical_threshold: 95.0,  // Critical at 95°C
        }
    }

    pub fn update_thermal_info(&mut self) -> Result<ThermalInfo> {
        let mut system = self.system.lock().map_err(|e|
            HardwareError::ThermalMonitoring(format!("Failed to lock system: {}", e)))?;
        let mut components = self.components.lock().map_err(|e|
            HardwareError::ThermalMonitoring(format!("Failed to lock components: {}", e)))?;

        components.refresh();
        system.refresh_cpu_specifics(CpuRefreshKind::everything());

        // Get CPU temperature from components
        let cpu_temp = self.get_cpu_temperature(&components, &system);

        // Get GPU temperature (platform-specific)
        let gpu_temp = self.get_gpu_temperature()?;

        // Determine thermal state
        let max_temp = cpu_temp.max(gpu_temp.unwrap_or(0.0));
        let thermal_state = self.classify_thermal_state(max_temp);

        // Detect cooling capability
        let cooling_capability = self.detect_cooling_capability(&system);

        // Calculate throttle percentage based on temperature
        let throttle_percentage = self.calculate_throttle_percentage(max_temp);

        // Release locks before mutating self
        drop(components);
        drop(system);

        // Store reading in history
        self.add_thermal_reading(cpu_temp, gpu_temp, thermal_state, throttle_percentage);

        // Update config adjustments based on thermal state
        self.update_config_adjustments(thermal_state);

        Ok(ThermalInfo {
            cpu_temperature: Some(cpu_temp),
            gpu_temperature: gpu_temp,
            thermal_state,
            cooling_capability,
        })
    }

    fn get_cpu_temperature(&self, components: &Components, system: &System) -> f32 {
        // Look for CPU temperature sensors
        for component in components.iter() {
            let label = component.label().to_lowercase();
            if label.contains("cpu") || label.contains("core") || label.contains("package") {
                return component.temperature();
            }
        }

        // If no CPU sensor found, estimate based on system load
        let cpu_usage = system.global_cpu_usage();
        self.estimate_temperature_from_load(cpu_usage)
    }

    fn get_gpu_temperature(&self) -> Result<Option<f32>> {
        #[cfg(target_os = "linux")]
        {
            // Try NVIDIA SMI
            if let Ok(output) = std::process::Command::new("nvidia-smi")
                .args(&["--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"])
                .output()
            {
                if let Ok(temp_str) = String::from_utf8(output.stdout) {
                    if let Ok(temp) = temp_str.trim().parse::<f32>() {
                        return Ok(Some(temp));
                    }
                }
            }

            // Try AMD ROCm
            if let Ok(output) = std::process::Command::new("rocm-smi")
                .arg("--showtemp")
                .output()
            {
                let output_str = String::from_utf8_lossy(&output.stdout);
                if let Some(temp) = self.parse_rocm_temperature(&output_str) {
                    return Ok(Some(temp));
                }
            }
        }

        #[cfg(target_os = "macos")]
        {
            // macOS doesn't expose GPU temperature easily
            // Estimate based on Metal Performance Shaders activity
            return Ok(None);
        }

        #[cfg(target_os = "windows")]
        {
            // Try WMI for GPU temperature
            if let Ok(temp) = self.get_windows_gpu_temp() {
                return Ok(Some(temp));
            }
        }

        #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
        Ok(None)
    }

    fn parse_rocm_temperature(&self, output: &str) -> Option<f32> {
        for line in output.lines() {
            if line.contains("Temperature") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                for part in parts {
                    if let Ok(temp) = part.trim_end_matches('C').parse::<f32>() {
                        return Some(temp);
                    }
                }
            }
        }
        None
    }

    #[cfg(target_os = "windows")]
    fn get_windows_gpu_temp(&self) -> Result<f32> {
        // This would use WMI or other Windows APIs
        // Placeholder implementation
        Ok(50.0)
    }

    fn classify_thermal_state(&self, temperature: f32) -> ThermalState {
        if temperature < 60.0 {
            ThermalState::Optimal
        } else if temperature < self.throttle_threshold {
            ThermalState::Warm
        } else if temperature < self.critical_threshold {
            ThermalState::Hot
        } else {
            ThermalState::Critical
        }
    }

    fn detect_cooling_capability(&self, _system: &System) -> CoolingCapability {
        // Platform-specific cooling detection
        #[cfg(target_os = "linux")]
        {
            if let Ok(output) = std::process::Command::new("sensors")
                .output()
            {
                let output_str = String::from_utf8_lossy(&output.stdout);
                if output_str.contains("water") || output_str.contains("liquid") {
                    return CoolingCapability::Liquid;
                }
                if output_str.contains("fan") && output_str.contains("rpm") {
                    return CoolingCapability::Active;
                }
            }
        }

        #[cfg(target_os = "macos")]
        {
            // Apple Silicon typically has excellent cooling
            if _system.cpus()[0].brand().contains("Apple") {
                return CoolingCapability::Liquid; // Treat as high-efficiency cooling
            }
        }

        // Default based on system characteristics
        let cpu_count = _system.cpus().len();
        if cpu_count > 16 {
            CoolingCapability::Active // High-core systems usually have active cooling
        } else {
            CoolingCapability::Passive
        }
    }

    fn calculate_throttle_percentage(&self, temperature: f32) -> f32 {
        if temperature < self.throttle_threshold {
            0.0
        } else if temperature >= self.critical_threshold {
            1.0 // 100% throttle
        } else {
            // Linear interpolation between throttle and critical thresholds
            (temperature - self.throttle_threshold) / (self.critical_threshold - self.throttle_threshold)
        }
    }

    fn add_thermal_reading(&mut self, cpu_temp: f32, gpu_temp: Option<f32>,
                           thermal_state: ThermalState, throttle_percentage: f32) {
        let reading = ThermalReading {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            cpu_temp,
            gpu_temp,
            thermal_state,
            throttle_percentage,
        };

        self.thermal_history.push(reading);

        // Keep only last 60 readings
        if self.thermal_history.len() > 60 {
            self.thermal_history.remove(0);
        }
    }

    fn update_config_adjustments(&mut self, thermal_state: ThermalState) {
        let multiplier = match thermal_state {
            ThermalState::Optimal => 1.0,
            ThermalState::Warm => 0.9,
            ThermalState::Hot => 0.7,
            ThermalState::Critical => 0.5,
        };

        self.config_adjustments.insert("performance".to_string(), multiplier);
        self.config_adjustments.insert("batch_size".to_string(), multiplier);
        self.config_adjustments.insert("threads".to_string(), multiplier);
    }

    pub fn get_dynamic_adjustments(&self, profile: &HardwareProfile) -> DynamicAdjustment {
        let current_state = self.get_current_thermal_state();
        let trend = self.analyze_thermal_trend();

        let base_multiplier: f32 = match current_state {
            ThermalState::Optimal => 1.0,
            ThermalState::Warm => 0.85,
            ThermalState::Hot => 0.6,
            ThermalState::Critical => 0.3,
        };

        // Adjust based on trend
        let trend_adjustment: f32 = match trend {
            ThermalTrend::Rising => -0.1,  // More aggressive throttling if rising
            ThermalTrend::Stable => 0.0,
            ThermalTrend::Falling => 0.05, // Less aggressive if falling
        };

        let final_multiplier = (base_multiplier + trend_adjustment).max(0.3).min(1.0);

        // Different components scale differently
        let batch_size_multiplier = final_multiplier;
        let frequency_multiplier = final_multiplier;
        let memory_multiplier = (final_multiplier + 0.1).min(1.0); // Memory less affected
        let thread_multiplier = final_multiplier;
        let quality_multiplier = (final_multiplier + 0.2).min(1.0); // Try to maintain quality

        let reason = format!("Thermal state: {:?}, Trend: {:?}, Base adjustment: {:.0}%",
            current_state, trend, (1.0 - final_multiplier) * 100.0);

        DynamicAdjustment {
            batch_size_multiplier,
            frequency_multiplier,
            memory_multiplier,
            thread_multiplier,
            quality_multiplier,
            reason,
        }
    }

    fn get_current_thermal_state(&self) -> ThermalState {
        self.thermal_history.last()
            .map(|r| r.thermal_state)
            .unwrap_or(ThermalState::Optimal)
    }

    fn analyze_thermal_trend(&self) -> ThermalTrend {
        if self.thermal_history.len() < 5 {
            return ThermalTrend::Stable;
        }

        let recent = &self.thermal_history[self.thermal_history.len() - 5..];
        let avg_recent = recent.iter().map(|r| r.cpu_temp).sum::<f32>() / 5.0;

        if self.thermal_history.len() < 10 {
            return ThermalTrend::Stable;
        }

        let older = &self.thermal_history[self.thermal_history.len() - 10..self.thermal_history.len() - 5];
        let avg_older = older.iter().map(|r| r.cpu_temp).sum::<f32>() / 5.0;

        if avg_recent > avg_older + 2.0 {
            ThermalTrend::Rising
        } else if avg_recent < avg_older - 2.0 {
            ThermalTrend::Falling
        } else {
            ThermalTrend::Stable
        }
    }

    fn estimate_temperature_from_load(&self, load_percentage: f32) -> f32 {
        // Simple estimation model
        let base_temp = 40.0; // Idle temperature
        let max_temp = 85.0;  // Full load temperature
        base_temp + (load_percentage / 100.0) * (max_temp - base_temp)
    }

    pub fn should_throttle(&self) -> bool {
        self.get_current_thermal_state() as u8 >= ThermalState::Hot as u8
    }

    pub fn get_throttle_factor(&self) -> f32 {
        self.thermal_history.last()
            .map(|r| 1.0 - r.throttle_percentage)
            .unwrap_or(1.0)
    }

    pub fn get_thermal_history(&self) -> &[ThermalReading] {
        &self.thermal_history
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum ThermalTrend {
    Rising,
    Stable,
    Falling,
}

impl Default for ThermalMonitor {
    fn default() -> Self {
        Self::new()
    }
}

// Dynamic performance adjustment system
pub struct PerformanceAdjuster {
    thermal_monitor: ThermalMonitor,
    last_adjustment: Instant,
    adjustment_interval: Duration,
    current_adjustments: DynamicAdjustment,
}

impl PerformanceAdjuster {
    pub fn new() -> Self {
        Self {
            thermal_monitor: ThermalMonitor::new(),
            last_adjustment: Instant::now(),
            adjustment_interval: Duration::from_secs(5), // Adjust every 5 seconds
            current_adjustments: DynamicAdjustment {
                batch_size_multiplier: 1.0,
                frequency_multiplier: 1.0,
                memory_multiplier: 1.0,
                thread_multiplier: 1.0,
                quality_multiplier: 1.0,
                reason: "Initial state".to_string(),
            },
        }
    }

    pub fn update(&mut self, profile: &HardwareProfile) -> Result<DynamicAdjustment> {
        // Update thermal info
        self.thermal_monitor.update_thermal_info()?;

        // Check if it's time for adjustment
        if self.last_adjustment.elapsed() >= self.adjustment_interval {
            self.current_adjustments = self.thermal_monitor.get_dynamic_adjustments(profile);
            self.last_adjustment = Instant::now();

            tracing::info!("Performance adjustment: {}", self.current_adjustments.reason);
        }

        Ok(self.current_adjustments.clone())
    }

    pub fn apply_to_batch_size(&self, base_batch_size: u32) -> u32 {
        let adjusted = (base_batch_size as f32 * self.current_adjustments.batch_size_multiplier) as u32;
        adjusted.max(1)
    }

    pub fn apply_to_thread_count(&self, base_threads: u32) -> u32 {
        let adjusted = (base_threads as f32 * self.current_adjustments.thread_multiplier) as u32;
        adjusted.max(1)
    }

    pub fn apply_to_memory_allocation(&self, base_memory_mb: u32) -> u32 {
        let adjusted = (base_memory_mb as f32 * self.current_adjustments.memory_multiplier) as u32;
        adjusted.max(256) // Minimum 256MB
    }

    pub fn get_quality_level(&self) -> QualityLevel {
        match self.current_adjustments.quality_multiplier {
            x if x >= 0.9 => QualityLevel::High,
            x if x >= 0.7 => QualityLevel::Medium,
            x if x >= 0.5 => QualityLevel::Low,
            _ => QualityLevel::Minimum,
        }
    }

    pub fn get_thermal_monitor(&self) -> &ThermalMonitor {
        &self.thermal_monitor
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QualityLevel {
    High,
    Medium,
    Low,
    Minimum,
}

impl Default for PerformanceAdjuster {
    fn default() -> Self {
        Self::new()
    }
}