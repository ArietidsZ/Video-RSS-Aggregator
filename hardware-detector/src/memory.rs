use crate::{types::MemoryInfo, HardwareError, Result};
use sysinfo::System;

pub struct MemoryDetector {
    system: System,
}

impl MemoryDetector {
    pub fn new() -> Self {
        let mut system = System::new_all();
        system.refresh_all();
        Self { system }
    }

    pub fn detect_memory_info(&mut self) -> Result<MemoryInfo> {
        self.system.refresh_memory();

        let total_bytes = self.system.total_memory() * 1024; // sysinfo returns KB
        let available_bytes = self.system.available_memory() * 1024;
        let swap_total = self.system.total_swap() * 1024;
        let swap_available = self.system.free_swap() * 1024;

        // Detect memory type and speed
        let (memory_type, speed_mhz, channels) = self.detect_memory_details()?;

        Ok(MemoryInfo {
            total_bytes,
            available_bytes,
            swap_total,
            swap_available,
            memory_type,
            speed_mhz,
            channels,
        })
    }

    fn detect_memory_details(&self) -> Result<(String, Option<u64>, Option<u32>)> {
        // Platform-specific memory detection
        #[cfg(target_os = "linux")]
        {
            if let Ok((mem_type, speed, channels)) = self.detect_memory_linux() {
                return Ok((mem_type, speed, channels));
            }
        }

        #[cfg(target_os = "macos")]
        {
            if let Ok((mem_type, speed, channels)) = self.detect_memory_macos() {
                return Ok((mem_type, speed, channels));
            }
        }

        #[cfg(target_os = "windows")]
        {
            if let Ok((mem_type, speed, channels)) = self.detect_memory_windows() {
                return Ok((mem_type, speed, channels));
            }
        }

        // Fallback: estimate based on total memory and current year
        let total_gb = self.system.total_memory() / (1024 * 1024); // Convert to GB
        let (memory_type, estimated_speed) = if total_gb >= 64 {
            ("DDR5".to_string(), Some(5600)) // High-end system
        } else if total_gb >= 32 {
            ("DDR4".to_string(), Some(3200)) // Mid-high end
        } else if total_gb >= 16 {
            ("DDR4".to_string(), Some(2666)) // Mid-range
        } else {
            ("DDR4".to_string(), Some(2400)) // Entry level
        };

        // Estimate channels based on memory size
        let estimated_channels = if total_gb >= 32 { Some(4) } else { Some(2) };

        Ok((memory_type, estimated_speed, estimated_channels))
    }

    #[cfg(target_os = "linux")]
    fn detect_memory_linux(&self) -> Result<(String, Option<u64>, Option<u32>)> {
        use std::fs;
        use std::path::Path;

        // Try to read from DMI tables
        let dmi_path = Path::new("/sys/firmware/dmi/tables/DMI");
        if dmi_path.exists() {
            // This requires root access, so we'll try alternative approaches
        }

        // Try reading from /proc/meminfo for additional details
        if let Ok(meminfo) = fs::read_to_string("/proc/meminfo") {
            // Parse meminfo for additional memory details
            // This doesn't give us speed/type directly, but we can infer some things
        }

        // Try lshw if available
        if let Ok(output) = std::process::Command::new("lshw")
            .args(&["-C", "memory", "-short"])
            .output()
        {
            let output_str = String::from_utf8_lossy(&output.stdout);
            return self.parse_lshw_memory(&output_str);
        }

        // Try dmidecode if available
        if let Ok(output) = std::process::Command::new("dmidecode")
            .args(&["-t", "memory"])
            .output()
        {
            let output_str = String::from_utf8_lossy(&output.stdout);
            return self.parse_dmidecode_memory(&output_str);
        }

        Err(HardwareError::MemoryDetection("Failed to detect memory details on Linux".to_string()))
    }

    #[cfg(target_os = "macos")]
    fn detect_memory_macos(&self) -> Result<(String, Option<u64>, Option<u32>)> {
        use std::process::Command;

        // Use system_profiler to get detailed memory information
        let output = Command::new("system_profiler")
            .args(&["SPMemoryDataType"])
            .output()
            .map_err(|e| HardwareError::MemoryDetection(format!("Failed to run system_profiler: {}", e)))?;

        let output_str = String::from_utf8_lossy(&output.stdout);
        self.parse_macos_memory(&output_str)
    }

    #[cfg(target_os = "windows")]
    fn detect_memory_windows(&self) -> Result<(String, Option<u64>, Option<u32>)> {
        // Use WMI to get memory information
        // For now, return estimates
        let total_gb = self.system.total_memory() / (1024 * 1024);

        let (memory_type, speed) = if total_gb >= 32 {
            ("DDR4".to_string(), Some(3200))
        } else {
            ("DDR4".to_string(), Some(2666))
        };

        Ok((memory_type, speed, Some(2)))
    }

    fn parse_lshw_memory(&self, output: &str) -> Result<(String, Option<u64>, Option<u32>)> {
        let mut memory_type = "Unknown".to_string();
        let mut speed_mhz = None;
        let mut channels = 0u32;

        for line in output.lines() {
            if line.contains("DIMM") {
                channels += 1;
            }

            if line.contains("DDR") {
                if line.contains("DDR5") {
                    memory_type = "DDR5".to_string();
                } else if line.contains("DDR4") {
                    memory_type = "DDR4".to_string();
                } else if line.contains("DDR3") {
                    memory_type = "DDR3".to_string();
                }
            }

            // Look for speed information
            if let Some(mhz_pos) = line.find("MHz") {
                let before_mhz = &line[..mhz_pos];
                if let Some(space_pos) = before_mhz.rfind(' ') {
                    if let Ok(speed) = before_mhz[space_pos + 1..].parse::<u64>() {
                        speed_mhz = Some(speed);
                    }
                }
            }
        }

        if channels == 0 {
            channels = 2; // Default assumption
        }

        Ok((memory_type, speed_mhz, Some(channels)))
    }

    fn parse_dmidecode_memory(&self, output: &str) -> Result<(String, Option<u64>, Option<u32>)> {
        let mut memory_type = "Unknown".to_string();
        let mut speed_mhz = None;
        let mut channels = 0u32;

        let mut in_memory_device = false;

        for line in output.lines() {
            let line = line.trim();

            if line.starts_with("Memory Device") {
                in_memory_device = true;
                continue;
            }

            if in_memory_device && line.is_empty() {
                in_memory_device = false;
                continue;
            }

            if in_memory_device {
                if line.starts_with("Size:") && !line.contains("No Module Installed") {
                    channels += 1;
                }

                if line.starts_with("Type:") {
                    let type_value = line.split(':').nth(1).unwrap_or("").trim();
                    if type_value.starts_with("DDR") {
                        memory_type = type_value.to_string();
                    }
                }

                if line.starts_with("Speed:") {
                    let speed_str = line.split(':').nth(1).unwrap_or("").trim();
                    if let Some(mhz_pos) = speed_str.find(" MT/s") {
                        if let Ok(speed) = speed_str[..mhz_pos].parse::<u64>() {
                            speed_mhz = Some(speed);
                        }
                    }
                }
            }
        }

        if channels == 0 {
            channels = 2; // Default assumption
        }

        Ok((memory_type, speed_mhz, Some(channels)))
    }

    fn parse_macos_memory(&self, output: &str) -> Result<(String, Option<u64>, Option<u32>)> {
        let mut memory_type = "Unknown".to_string();
        let mut speed_mhz = None;
        let mut channels = 0u32;

        for line in output.lines() {
            let line = line.trim();

            if line.contains("Type:") {
                let type_value = line.split(':').nth(1).unwrap_or("").trim();
                if type_value.contains("DDR") {
                    memory_type = type_value.to_string();
                }
            }

            if line.contains("Speed:") {
                let speed_str = line.split(':').nth(1).unwrap_or("").trim();
                if let Some(mhz_pos) = speed_str.find(" MHz") {
                    if let Ok(speed) = speed_str[..mhz_pos].parse::<u64>() {
                        speed_mhz = Some(speed);
                    }
                }
            }

            if line.contains("BANK") || line.contains("DIMM") {
                channels += 1;
            }
        }

        if channels == 0 {
            // Apple Silicon has unified memory
            if memory_type == "Unknown" {
                memory_type = "Unified".to_string();
                speed_mhz = Some(6400); // Apple Silicon typical
            }
            channels = 1; // Unified memory architecture
        }

        Ok((memory_type, speed_mhz, Some(channels)))
    }
}

impl Default for MemoryDetector {
    fn default() -> Self {
        Self::new()
    }
}