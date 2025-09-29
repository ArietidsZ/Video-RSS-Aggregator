use crate::{types::CpuInfo, HardwareError, Result};
use sysinfo::System;

pub struct CpuDetector {
    system: System,
}

impl CpuDetector {
    pub fn new() -> Self {
        let mut system = System::new_all();
        system.refresh_all();
        Self { system }
    }

    pub fn detect_cpu_info(&mut self) -> Result<CpuInfo> {
        self.system.refresh_cpu_all();

        let cpus = self.system.cpus();
        if cpus.is_empty() {
            return Err(HardwareError::CpuDetection("No CPUs detected".to_string()));
        }

        let first_cpu = &cpus[0];
        let brand = first_cpu.brand().to_string();
        let cores_logical = cpus.len();
        let cores_physical = self.detect_physical_cores()?;
        let frequency_mhz = first_cpu.frequency();

        // Detect CPU features
        let features = self.detect_cpu_features();

        // Detect cache sizes
        let (cache_l1, cache_l2, cache_l3) = self.detect_cache_sizes()?;

        // Detect architecture
        let architecture = self.detect_architecture();

        // Get thermal information
        let thermal_max = self.get_thermal_max_temperature();

        Ok(CpuInfo {
            brand,
            cores_physical,
            cores_logical,
            frequency_mhz,
            cache_l1,
            cache_l2,
            cache_l3,
            architecture,
            features,
            thermal_max,
        })
    }

    fn detect_physical_cores(&self) -> Result<usize> {
        // Try to use hwloc for accurate physical core detection
        #[cfg(feature = "hwloc")]
        {
            use hwloc2::{Topology, TopologyObject};

            let topo = Topology::new().map_err(|e| {
                HardwareError::CpuDetection(format!("Failed to initialize hwloc: {}", e))
            })?;

            let mut physical_cores = 0;
            topo.objects_with_type(&hwloc2::ObjectType::Core)
                .for_each(|_| physical_cores += 1);

            if physical_cores > 0 {
                return Ok(physical_cores);
            }
        }

        // Fallback: estimate based on logical cores and typical hyperthreading
        let logical_cores = self.system.cpus().len();
        let estimated_physical = if self.has_hyperthreading() {
            logical_cores / 2
        } else {
            logical_cores
        };

        Ok(estimated_physical.max(1))
    }

    fn has_hyperthreading(&self) -> bool {
        // Simple heuristic: check if logical cores is significantly more than typical physical cores
        let logical_cores = self.system.cpus().len();
        let brand = self.system.cpus()[0].brand().to_lowercase();

        // Intel typically has hyperthreading, AMD Ryzen varies
        if brand.contains("intel") {
            logical_cores > 4 && logical_cores % 2 == 0
        } else if brand.contains("amd") {
            // Modern AMD Ryzen has SMT (similar to hyperthreading)
            brand.contains("ryzen") && logical_cores > 8
        } else {
            false
        }
    }

    fn detect_cpu_features(&self) -> Vec<String> {
        let mut features = Vec::new();

        // Platform-specific CPU feature detection
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx") {
                features.push("AVX".to_string());
            }
            if is_x86_feature_detected!("avx2") {
                features.push("AVX2".to_string());
            }
            if is_x86_feature_detected!("avx512f") {
                features.push("AVX512".to_string());
            }
            if is_x86_feature_detected!("sse4.2") {
                features.push("SSE4.2".to_string());
            }
            if is_x86_feature_detected!("fma") {
                features.push("FMA".to_string());
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                features.push("NEON".to_string());
            }
        }

        features
    }

    fn detect_cache_sizes(&self) -> Result<(u64, u64, u64)> {
        // Try platform-specific cache detection
        #[cfg(target_os = "linux")]
        {
            if let Ok((l1, l2, l3)) = self.detect_cache_linux() {
                return Ok((l1, l2, l3));
            }
        }

        #[cfg(target_os = "macos")]
        {
            if let Ok((l1, l2, l3)) = self.detect_cache_macos() {
                return Ok((l1, l2, l3));
            }
        }

        #[cfg(target_os = "windows")]
        {
            if let Ok((l1, l2, l3)) = self.detect_cache_windows() {
                return Ok((l1, l2, l3));
            }
        }

        // Fallback: estimate based on CPU brand and model
        let brand = self.system.cpus()[0].brand().to_lowercase();
        let (l1, l2, l3) = if brand.contains("intel") {
            (32 * 1024, 256 * 1024, 8 * 1024 * 1024) // Intel typical
        } else if brand.contains("amd") {
            (32 * 1024, 512 * 1024, 16 * 1024 * 1024) // AMD Ryzen typical
        } else if brand.contains("apple") {
            (128 * 1024, 4 * 1024 * 1024, 12 * 1024 * 1024) // Apple Silicon typical
        } else {
            (32 * 1024, 256 * 1024, 4 * 1024 * 1024) // Generic estimate
        };

        Ok((l1, l2, l3))
    }

    #[cfg(target_os = "linux")]
    fn detect_cache_linux(&self) -> Result<(u64, u64, u64)> {
        use std::fs;
        use std::path::Path;

        let mut l1_data = 0u64;
        let mut l2 = 0u64;
        let mut l3 = 0u64;

        // Check /sys/devices/system/cpu/cpu0/cache/
        let cache_path = Path::new("/sys/devices/system/cpu/cpu0/cache");
        if cache_path.exists() {
            for entry in fs::read_dir(cache_path).map_err(|e| {
                HardwareError::CpuDetection(format!("Failed to read cache directory: {}", e))
            })? {
                let entry = entry.map_err(|e| {
                    HardwareError::CpuDetection(format!("Failed to read cache entry: {}", e))
                })?;

                let index_path = entry.path();
                let level_file = index_path.join("level");
                let size_file = index_path.join("size");
                let type_file = index_path.join("type");

                if level_file.exists() && size_file.exists() && type_file.exists() {
                    let level = fs::read_to_string(&level_file).ok()
                        .and_then(|s| s.trim().parse::<u8>().ok());
                    let size_str = fs::read_to_string(&size_file).ok();
                    let cache_type = fs::read_to_string(&type_file).ok();

                    if let (Some(level), Some(size_str), Some(cache_type)) = (level, size_str, cache_type) {
                        let size = parse_cache_size(&size_str.trim());

                        match (level, cache_type.trim()) {
                            (1, "Data") => l1_data = size,
                            (2, _) => l2 = size,
                            (3, _) => l3 = size,
                            _ => {}
                        }
                    }
                }
            }
        }

        if l1_data == 0 && l2 == 0 && l3 == 0 {
            return Err(HardwareError::CpuDetection("Failed to detect cache sizes".to_string()));
        }

        Ok((l1_data, l2, l3))
    }

    #[cfg(target_os = "macos")]
    fn detect_cache_macos(&self) -> Result<(u64, u64, u64)> {
        use std::process::Command;

        let output = Command::new("sysctl")
            .args(&["-n", "hw.l1dcachesize", "hw.l2cachesize", "hw.l3cachesize"])
            .output()
            .map_err(|e| HardwareError::CpuDetection(format!("Failed to run sysctl: {}", e)))?;

        let output_str = String::from_utf8_lossy(&output.stdout);
        let lines: Vec<&str> = output_str.lines().collect();

        let l1 = lines.get(0).and_then(|s| s.parse::<u64>().ok()).unwrap_or(0);
        let l2 = lines.get(1).and_then(|s| s.parse::<u64>().ok()).unwrap_or(0);
        let l3 = lines.get(2).and_then(|s| s.parse::<u64>().ok()).unwrap_or(0);

        Ok((l1, l2, l3))
    }

    #[cfg(target_os = "windows")]
    fn detect_cache_windows(&self) -> Result<(u64, u64, u64)> {
        // Windows cache detection would use WMI or registry
        // For now, return typical values based on CPU brand
        let brand = self.system.cpus()[0].brand().to_lowercase();
        let (l1, l2, l3) = if brand.contains("intel") {
            (32 * 1024, 256 * 1024, 8 * 1024 * 1024)
        } else {
            (32 * 1024, 512 * 1024, 16 * 1024 * 1024)
        };

        Ok((l1, l2, l3))
    }

    fn detect_architecture(&self) -> String {
        std::env::consts::ARCH.to_string()
    }

    fn get_thermal_max_temperature(&self) -> Option<f32> {
        // Platform-specific thermal limits
        let brand = self.system.cpus()[0].brand().to_lowercase();

        if brand.contains("intel") {
            Some(100.0) // Intel typically 100°C
        } else if brand.contains("amd") {
            Some(95.0) // AMD typically 95°C
        } else if brand.contains("apple") {
            Some(100.0) // Apple Silicon
        } else {
            Some(85.0) // Conservative default
        }
    }
}

fn parse_cache_size(size_str: &str) -> u64 {
    let size_str = size_str.to_uppercase();

    if let Some(pos) = size_str.rfind(char::is_alphabetic) {
        let (number_part, _unit_part) = size_str.split_at(pos);
        let number: u64 = number_part.parse().unwrap_or(0);
        let unit = &size_str[pos..];

        match unit {
            "K" => number * 1024,
            "M" => number * 1024 * 1024,
            "G" => number * 1024 * 1024 * 1024,
            _ => number,
        }
    } else {
        size_str.parse().unwrap_or(0)
    }
}

impl Default for CpuDetector {
    fn default() -> Self {
        Self::new()
    }
}