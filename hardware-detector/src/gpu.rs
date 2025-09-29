use crate::{types::{GpuInfo, GpuVendor}, HardwareError, Result};

pub struct GpuDetector;

impl GpuDetector {
    pub fn new() -> Self {
        Self
    }

    pub fn detect_gpus(&self) -> Result<Vec<GpuInfo>> {
        let mut gpus = Vec::new();

        // Try NVIDIA detection first
        if let Ok(nvidia_gpus) = self.detect_nvidia_gpus() {
            gpus.extend(nvidia_gpus);
        }

        // Try AMD detection
        if let Ok(amd_gpus) = self.detect_amd_gpus() {
            gpus.extend(amd_gpus);
        }

        // Try Intel detection
        if let Ok(intel_gpus) = self.detect_intel_gpus() {
            gpus.extend(intel_gpus);
        }

        // Try Apple Silicon detection
        if let Ok(apple_gpus) = self.detect_apple_gpus() {
            gpus.extend(apple_gpus);
        }

        if gpus.is_empty() {
            // Try generic detection as fallback
            if let Ok(generic_gpus) = self.detect_generic_gpus() {
                gpus.extend(generic_gpus);
            }
        }

        Ok(gpus)
    }

    fn detect_nvidia_gpus(&self) -> Result<Vec<GpuInfo>> {
        let mut gpus = Vec::new();

        // Try using nvml-wrapper
        match nvml_wrapper::Nvml::init() {
            Ok(nvml) => {
                let device_count = nvml.device_count()
                    .map_err(|e| HardwareError::GpuDetection(format!("Failed to get NVIDIA device count: {}", e)))?;

                for i in 0..device_count {
                    if let Ok(device) = nvml.device_by_index(i) {
                        let mut gpu_info = GpuInfo {
                            vendor: GpuVendor::Nvidia,
                            model: "Unknown NVIDIA GPU".to_string(),
                            memory_total: 0,
                            memory_available: 0,
                            compute_capability: None,
                            cuda_cores: None,
                            tensor_cores: None,
                            pcie_bandwidth: None,
                            power_limit: None,
                            temperature: None,
                            utilization: None,
                        };

                        // Get GPU name
                        if let Ok(name) = device.name() {
                            gpu_info.model = name;
                        }

                        // Get memory information
                        if let Ok(memory_info) = device.memory_info() {
                            gpu_info.memory_total = memory_info.total;
                            gpu_info.memory_available = memory_info.free;
                        }

                        // Get compute capability
                        if let Ok(compute_cap) = device.cuda_compute_capability() {
                            gpu_info.compute_capability = Some(format!("{}.{}", compute_cap.major, compute_cap.minor));
                        }

                        // Get power limit
                        if let Ok(power_limit) = device.power_management_limit_constraints() {
                            gpu_info.power_limit = Some(power_limit.max_limit / 1000); // Convert to watts
                        }

                        // Get current temperature
                        if let Ok(temp) = device.temperature(nvml_wrapper::enum_wrappers::device::TemperatureSensor::Gpu) {
                            gpu_info.temperature = Some(temp as f32);
                        }

                        // Get current utilization
                        if let Ok(util) = device.utilization_rates() {
                            gpu_info.utilization = Some(util.gpu as f32);
                        }

                        // Get PCIe bandwidth information (commented out due to API limitations)
                        // Modern nvml-wrapper doesn't expose PCIe link methods directly
                        // This would require direct NVML API calls or different approach

                        // Estimate CUDA cores based on GPU name
                        gpu_info.cuda_cores = self.estimate_nvidia_cuda_cores(&gpu_info.model);

                        // Estimate tensor cores for RTX cards
                        if gpu_info.model.contains("RTX") || gpu_info.model.contains("A100") || gpu_info.model.contains("A6000") {
                            gpu_info.tensor_cores = self.estimate_nvidia_tensor_cores(&gpu_info.model);
                        }

                        gpus.push(gpu_info);
                    }
                }
            }
            Err(_) => {
                // NVML not available, try alternative detection methods
                return self.detect_nvidia_fallback();
            }
        }

        Ok(gpus)
    }

    fn detect_nvidia_fallback(&self) -> Result<Vec<GpuInfo>> {
        // Try nvidia-smi command
        if let Ok(output) = std::process::Command::new("nvidia-smi")
            .args(&["--query-gpu=name,memory.total,memory.free,temperature.gpu,utilization.gpu", "--format=csv,noheader,nounits"])
            .output()
        {
            if output.status.success() {
                let output_str = String::from_utf8_lossy(&output.stdout);
                return self.parse_nvidia_smi_output(&output_str);
            }
        }

        Ok(Vec::new())
    }

    fn parse_nvidia_smi_output(&self, output: &str) -> Result<Vec<GpuInfo>> {
        let mut gpus = Vec::new();

        for line in output.lines() {
            let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
            if parts.len() >= 5 {
                let name = parts[0].to_string();
                let memory_total = parts[1].parse::<u64>().unwrap_or(0) * 1024 * 1024; // Convert MB to bytes
                let memory_free = parts[2].parse::<u64>().unwrap_or(0) * 1024 * 1024;
                let temperature = parts[3].parse::<f32>().ok();
                let utilization = parts[4].parse::<f32>().ok();

                let gpu_info = GpuInfo {
                    vendor: GpuVendor::Nvidia,
                    model: name.clone(),
                    memory_total,
                    memory_available: memory_free,
                    compute_capability: None,
                    cuda_cores: self.estimate_nvidia_cuda_cores(&name),
                    tensor_cores: if name.contains("RTX") { self.estimate_nvidia_tensor_cores(&name) } else { None },
                    pcie_bandwidth: None,
                    power_limit: None,
                    temperature,
                    utilization,
                };

                gpus.push(gpu_info);
            }
        }

        Ok(gpus)
    }

    fn estimate_nvidia_cuda_cores(&self, model: &str) -> Option<u32> {
        // Comprehensive estimates based on GPU models
        let model_lower = model.to_lowercase();

        // RTX 50 Series (Ada Lovelace Next-Gen)
        if model_lower.contains("5090") {
            Some(21760)
        } else if model_lower.contains("5080") {
            Some(15360)
        } else if model_lower.contains("5070") {
            Some(8960)

        // RTX 40 Series (Ada Lovelace)
        } else if model_lower.contains("4090") {
            Some(16384)
        } else if model_lower.contains("4080") && model_lower.contains("super") {
            Some(10240)
        } else if model_lower.contains("4080") {
            Some(9728)
        } else if model_lower.contains("4070") && model_lower.contains("ti") && model_lower.contains("super") {
            Some(8448)
        } else if model_lower.contains("4070") && model_lower.contains("ti") {
            Some(7680)
        } else if model_lower.contains("4070") && model_lower.contains("super") {
            Some(7168)
        } else if model_lower.contains("4070") {
            Some(5888)
        } else if model_lower.contains("4060") && model_lower.contains("ti") {
            Some(4352)
        } else if model_lower.contains("4060") {
            Some(3072)

        // RTX 30 Series (Ampere)
        } else if model_lower.contains("3090") && model_lower.contains("ti") {
            Some(10752)
        } else if model_lower.contains("3090") {
            Some(10496)
        } else if model_lower.contains("3080") && model_lower.contains("ti") {
            Some(10240)
        } else if model_lower.contains("3080") {
            Some(8704)
        } else if model_lower.contains("3070") && model_lower.contains("ti") {
            Some(6144)
        } else if model_lower.contains("3070") {
            Some(5888)
        } else if model_lower.contains("3060") && model_lower.contains("ti") {
            Some(4864)
        } else if model_lower.contains("3060") {
            Some(3584)

        // Data Center GPUs
        } else if model_lower.contains("h100") {
            Some(16896)
        } else if model_lower.contains("a100") && model_lower.contains("80gb") {
            Some(6912)
        } else if model_lower.contains("a100") {
            Some(6912)
        } else if model_lower.contains("a6000") {
            Some(10752)
        } else if model_lower.contains("a5000") {
            Some(8192)
        } else if model_lower.contains("a4000") {
            Some(6144)
        } else if model_lower.contains("a2000") {
            Some(3328)

        // Quadro Series
        } else if model_lower.contains("rtx 6000") {
            Some(4608)
        } else if model_lower.contains("rtx 5000") {
            Some(3072)
        } else if model_lower.contains("rtx 4000") {
            Some(2304)

        // GTX Series
        } else if model_lower.contains("1080") && model_lower.contains("ti") {
            Some(3584)
        } else if model_lower.contains("1080") {
            Some(2560)
        } else if model_lower.contains("1070") && model_lower.contains("ti") {
            Some(2432)
        } else if model_lower.contains("1070") {
            Some(1920)
        } else if model_lower.contains("1060") {
            Some(1280)
        } else {
            None
        }
    }

    fn estimate_nvidia_tensor_cores(&self, model: &str) -> Option<u32> {
        let model_lower = model.to_lowercase();

        // RTX 50 Series (4th Gen RT Cores)
        if model_lower.contains("5090") {
            Some(680)
        } else if model_lower.contains("5080") {
            Some(480)
        } else if model_lower.contains("5070") {
            Some(280)

        // RTX 40 Series (3rd Gen RT Cores)
        } else if model_lower.contains("4090") {
            Some(512)
        } else if model_lower.contains("4080") && model_lower.contains("super") {
            Some(320)
        } else if model_lower.contains("4080") {
            Some(304)
        } else if model_lower.contains("4070") && model_lower.contains("ti") && model_lower.contains("super") {
            Some(264)
        } else if model_lower.contains("4070") && model_lower.contains("ti") {
            Some(240)
        } else if model_lower.contains("4070") && model_lower.contains("super") {
            Some(224)
        } else if model_lower.contains("4070") {
            Some(184)
        } else if model_lower.contains("4060") && model_lower.contains("ti") {
            Some(136)
        } else if model_lower.contains("4060") {
            Some(96)

        // RTX 30 Series (2nd Gen RT Cores)
        } else if model_lower.contains("3090") && model_lower.contains("ti") {
            Some(336)
        } else if model_lower.contains("3090") {
            Some(328)
        } else if model_lower.contains("3080") && model_lower.contains("ti") {
            Some(320)
        } else if model_lower.contains("3080") {
            Some(272)
        } else if model_lower.contains("3070") && model_lower.contains("ti") {
            Some(192)
        } else if model_lower.contains("3070") {
            Some(184)
        } else if model_lower.contains("3060") && model_lower.contains("ti") {
            Some(152)
        } else if model_lower.contains("3060") {
            Some(112)

        // Data Center GPUs (Hopper/Ampere)
        } else if model_lower.contains("h100") {
            Some(1056) // 4th Gen Tensor Cores
        } else if model_lower.contains("a100") && model_lower.contains("80gb") {
            Some(432)
        } else if model_lower.contains("a100") {
            Some(432)
        } else if model_lower.contains("a6000") {
            Some(336)
        } else if model_lower.contains("a5000") {
            Some(256)
        } else if model_lower.contains("a4000") {
            Some(192)
        } else if model_lower.contains("a2000") {
            Some(104)

        // Quadro RTX Series
        } else if model_lower.contains("rtx 6000") {
            Some(144) // Turing tensor cores
        } else if model_lower.contains("rtx 5000") {
            Some(96)
        } else if model_lower.contains("rtx 4000") {
            Some(72)
        } else {
            None
        }
    }

    fn detect_amd_gpus(&self) -> Result<Vec<GpuInfo>> {
        let mut gpus = Vec::new();

        // Try ROCm detection first (most comprehensive)
        if let Ok(rocm_gpus) = self.detect_amd_rocm() {
            gpus.extend(rocm_gpus);
        }

        // If no ROCm GPUs found, try system command fallbacks
        if gpus.is_empty() {
            #[cfg(target_os = "linux")]
            {
                if let Ok(output) = std::process::Command::new("lspci")
                    .args(&["-nn"])
                    .output()
                {
                    let output_str = String::from_utf8_lossy(&output.stdout);
                    gpus.extend(self.parse_amd_lspci_output(&output_str)?);
                }
            }

            #[cfg(target_os = "windows")]
            {
                gpus.extend(self.detect_amd_windows()?);
            }

            #[cfg(target_os = "macos")]
            {
                gpus.extend(self.detect_amd_macos()?);
            }
        }

        Ok(gpus)
    }

    fn detect_amd_rocm(&self) -> Result<Vec<GpuInfo>> {
        let mut gpus = Vec::new();

        // Try rocm-smi command for detailed AMD GPU information
        if let Ok(output) = std::process::Command::new("rocm-smi")
            .args(&["--showid", "--showproductname", "--showmeminfo", "--showuse", "--showtemp"])
            .output()
        {
            if output.status.success() {
                let output_str = String::from_utf8_lossy(&output.stdout);
                gpus.extend(self.parse_rocm_smi_output(&output_str)?);
            }
        }

        // Try radeontop for additional metrics
        if gpus.is_empty() {
            if let Ok(output) = std::process::Command::new("radeontop")
                .args(&["-d", "-", "-l", "1"])
                .output()
            {
                if output.status.success() {
                    let output_str = String::from_utf8_lossy(&output.stdout);
                    gpus.extend(self.parse_radeontop_output(&output_str)?);
                }
            }
        }

        Ok(gpus)
    }

    fn parse_rocm_smi_output(&self, output: &str) -> Result<Vec<GpuInfo>> {
        let mut gpus = Vec::new();
        let mut current_gpu: Option<GpuInfo> = None;

        for line in output.lines() {
            let line = line.trim();

            // Parse GPU ID and model information
            if line.starts_with("GPU[") {
                if let Some(gpu) = current_gpu.take() {
                    gpus.push(gpu);
                }

                // Extract GPU model from the line
                if let Some(colon_pos) = line.find(':') {
                    let model = line[colon_pos + 1..].trim().to_string();
                    current_gpu = Some(GpuInfo {
                        vendor: GpuVendor::Amd,
                        model,
                        memory_total: 0,
                        memory_available: 0,
                        compute_capability: None,
                        cuda_cores: None,
                        tensor_cores: None,
                        pcie_bandwidth: None,
                        power_limit: None,
                        temperature: None,
                        utilization: None,
                    });
                }
            }

            if let Some(ref mut gpu) = current_gpu {
                // Parse memory information
                if line.contains("Memory") && line.contains("MB") {
                    if let Some(mb_start) = line.find(": ") {
                        let mem_str = &line[mb_start + 2..];
                        if let Some(mb_end) = mem_str.find(" MB") {
                            if let Ok(memory_mb) = mem_str[..mb_end].parse::<u64>() {
                                gpu.memory_total = memory_mb * 1024 * 1024;
                                gpu.memory_available = gpu.memory_total; // Approximate
                            }
                        }
                    }
                }

                // Parse temperature
                if line.contains("Temperature") && line.contains("°C") {
                    if let Some(temp_start) = line.find(": ") {
                        let temp_str = &line[temp_start + 2..];
                        if let Some(temp_end) = temp_str.find("°C") {
                            if let Ok(temp) = temp_str[..temp_end].parse::<f32>() {
                                gpu.temperature = Some(temp);
                            }
                        }
                    }
                }

                // Parse GPU utilization
                if line.contains("GPU use") && line.contains("%") {
                    if let Some(use_start) = line.find(": ") {
                        let use_str = &line[use_start + 2..];
                        if let Some(use_end) = use_str.find("%") {
                            if let Ok(utilization) = use_str[..use_end].parse::<f32>() {
                                gpu.utilization = Some(utilization);
                            }
                        }
                    }
                }
            }
        }

        if let Some(gpu) = current_gpu {
            gpus.push(gpu);
        }

        Ok(gpus)
    }

    fn parse_radeontop_output(&self, output: &str) -> Result<Vec<GpuInfo>> {
        let mut gpus = Vec::new();

        for line in output.lines() {
            if line.contains("gpu") && line.contains("%") {
                // Basic AMD GPU detected via radeontop
                let gpu_info = GpuInfo {
                    vendor: GpuVendor::Amd,
                    model: "AMD Radeon GPU".to_string(),
                    memory_total: 0,
                    memory_available: 0,
                    compute_capability: None,
                    cuda_cores: None,
                    tensor_cores: None,
                    pcie_bandwidth: None,
                    power_limit: None,
                    temperature: None,
                    utilization: None,
                };
                gpus.push(gpu_info);
                break;
            }
        }

        Ok(gpus)
    }

    #[cfg(target_os = "windows")]
    fn detect_amd_windows(&self) -> Result<Vec<GpuInfo>> {
        // Try WMI query for AMD GPUs
        if let Ok(output) = std::process::Command::new("wmic")
            .args(&["path", "win32_VideoController", "get", "Name,AdapterRAM,CurrentHorizontalResolution", "/format:csv"])
            .output()
        {
            let output_str = String::from_utf8_lossy(&output.stdout);
            return self.parse_amd_wmic_output(&output_str);
        }

        Ok(Vec::new())
    }

    #[cfg(target_os = "macos")]
    fn detect_amd_macos(&self) -> Result<Vec<GpuInfo>> {
        // AMD GPUs on macOS are rare, but check system_profiler
        if let Ok(output) = std::process::Command::new("system_profiler")
            .args(&["SPDisplaysDataType"])
            .output()
        {
            let output_str = String::from_utf8_lossy(&output.stdout);
            return self.parse_amd_macos_output(&output_str);
        }

        Ok(Vec::new())
    }

    #[cfg(target_os = "windows")]
    fn parse_amd_wmic_output(&self, output: &str) -> Result<Vec<GpuInfo>> {
        let mut gpus = Vec::new();

        for line in output.lines().skip(1) { // Skip header
            let fields: Vec<&str> = line.split(',').collect();
            if fields.len() >= 3 {
                let name = fields[1].trim();
                if name.to_lowercase().contains("amd") || name.to_lowercase().contains("radeon") {
                    let memory_bytes = fields[0].trim().parse::<u64>().unwrap_or(0);

                    let gpu_info = GpuInfo {
                        vendor: GpuVendor::Amd,
                        model: name.to_string(),
                        memory_total: memory_bytes,
                        memory_available: memory_bytes, // Approximate
                        compute_capability: None,
                        cuda_cores: None,
                        tensor_cores: None,
                        pcie_bandwidth: None,
                        power_limit: None,
                        temperature: None,
                        utilization: None,
                    };

                    gpus.push(gpu_info);
                }
            }
        }

        Ok(gpus)
    }

    #[cfg(target_os = "macos")]
    fn parse_amd_macos_output(&self, output: &str) -> Result<Vec<GpuInfo>> {
        let mut gpus = Vec::new();

        for line in output.lines() {
            let line = line.trim();
            if line.contains("Chipset Model:") {
                let model = line.split(':').nth(1).unwrap_or("").trim();
                if model.to_lowercase().contains("amd") || model.to_lowercase().contains("radeon") {
                    let gpu_info = GpuInfo {
                        vendor: GpuVendor::Amd,
                        model: model.to_string(),
                        memory_total: 0,
                        memory_available: 0,
                        compute_capability: None,
                        cuda_cores: None,
                        tensor_cores: None,
                        pcie_bandwidth: None,
                        power_limit: None,
                        temperature: None,
                        utilization: None,
                    };
                    gpus.push(gpu_info);
                }
            }
        }

        Ok(gpus)
    }

    fn parse_amd_lspci_output(&self, output: &str) -> Result<Vec<GpuInfo>> {
        let mut gpus = Vec::new();

        for line in output.lines() {
            if line.contains("VGA") || line.contains("3D") {
                if line.to_lowercase().contains("amd") || line.to_lowercase().contains("radeon") {
                    // Extract GPU name from lspci output
                    let parts: Vec<&str> = line.split(':').collect();
                    if parts.len() >= 3 {
                        let name = parts[2].trim().to_string();

                        let gpu_info = GpuInfo {
                            vendor: GpuVendor::Amd,
                            model: name,
                            memory_total: 0, // Would need ROCm tools to get this
                            memory_available: 0,
                            compute_capability: None,
                            cuda_cores: None,
                            tensor_cores: None,
                            pcie_bandwidth: None,
                            power_limit: None,
                            temperature: None,
                            utilization: None,
                        };

                        gpus.push(gpu_info);
                    }
                }
            }
        }

        Ok(gpus)
    }

    fn detect_intel_gpus(&self) -> Result<Vec<GpuInfo>> {
        let mut gpus = Vec::new();

        // Try Intel GPU tools first (most comprehensive)
        if let Ok(intel_gpus) = self.detect_intel_tools() {
            gpus.extend(intel_gpus);
        }

        // If no Intel GPUs found via tools, try system commands
        if gpus.is_empty() {
            #[cfg(target_os = "linux")]
            {
                if let Ok(output) = std::process::Command::new("lspci")
                    .args(&["-nn"])
                    .output()
                {
                    let output_str = String::from_utf8_lossy(&output.stdout);
                    gpus.extend(self.parse_intel_lspci_output(&output_str)?);
                }
            }

            #[cfg(target_os = "windows")]
            {
                gpus.extend(self.detect_intel_windows()?);
            }

            #[cfg(target_os = "macos")]
            {
                gpus.extend(self.detect_intel_macos()?);
            }
        }

        Ok(gpus)
    }

    fn detect_intel_tools(&self) -> Result<Vec<GpuInfo>> {
        let mut gpus = Vec::new();

        // Try intel_gpu_top for Linux
        #[cfg(target_os = "linux")]
        {
            if let Ok(output) = std::process::Command::new("intel_gpu_top")
                .args(&["-l", "-n", "1"])
                .output()
            {
                if output.status.success() {
                    let output_str = String::from_utf8_lossy(&output.stdout);
                    gpus.extend(self.parse_intel_gpu_top(&output_str)?);
                }
            }
        }

        // Try OpenVINO device enumeration if available
        if let Ok(ovino_gpus) = self.detect_intel_openvino() {
            for gpu in ovino_gpus {
                // Avoid duplicates by checking if we already have this GPU
                if !gpus.iter().any(|existing: &GpuInfo| existing.model == gpu.model) {
                    gpus.push(gpu);
                }
            }
        }

        Ok(gpus)
    }

    fn detect_intel_openvino(&self) -> Result<Vec<GpuInfo>> {
        // Try to detect Intel GPUs through OpenVINO tools
        // This is a simplified detection - full OpenVINO integration would require Python bindings
        if let Ok(output) = std::process::Command::new("python3")
            .args(&["-c", r#"
try:
    import openvino as ov
    core = ov.Core()
    devices = core.available_devices
    for device in devices:
        if 'GPU' in device:
            print(f'OpenVINO GPU: {device}')
            device_name = core.get_property(device, "FULL_DEVICE_NAME")
            print(f'Device Name: {device_name}')
except:
    pass
"#])
            .output()
        {
            if output.status.success() {
                let output_str = String::from_utf8_lossy(&output.stdout);
                return self.parse_openvino_output(&output_str);
            }
        }

        Ok(Vec::new())
    }

    fn parse_openvino_output(&self, output: &str) -> Result<Vec<GpuInfo>> {
        let mut gpus = Vec::new();
        let mut current_gpu_name: Option<String> = None;

        for line in output.lines() {
            if line.contains("OpenVINO GPU:") {
                // Found an OpenVINO GPU device
                current_gpu_name = Some("Intel GPU (OpenVINO)".to_string());
            } else if line.contains("Device Name:") && current_gpu_name.is_some() {
                let device_name = line.split(':').nth(1).unwrap_or("").trim();

                let gpu_info = GpuInfo {
                    vendor: GpuVendor::Intel,
                    model: device_name.to_string(),
                    memory_total: 0, // OpenVINO doesn't expose this easily
                    memory_available: 0,
                    compute_capability: Some("OpenVINO".to_string()),
                    cuda_cores: None,
                    tensor_cores: None,
                    pcie_bandwidth: None,
                    power_limit: None,
                    temperature: None,
                    utilization: None,
                };

                gpus.push(gpu_info);
                current_gpu_name = None;
            }
        }

        Ok(gpus)
    }

    #[cfg(target_os = "linux")]
    fn parse_intel_gpu_top(&self, output: &str) -> Result<Vec<GpuInfo>> {
        let mut gpus = Vec::new();

        for line in output.lines() {
            if line.contains("intel-gpu-top") || line.contains("GPU") {
                // Extract GPU information from intel_gpu_top output
                let gpu_info = GpuInfo {
                    vendor: GpuVendor::Intel,
                    model: "Intel GPU".to_string(),
                    memory_total: 0,
                    memory_available: 0,
                    compute_capability: None,
                    cuda_cores: None,
                    tensor_cores: None,
                    pcie_bandwidth: None,
                    power_limit: None,
                    temperature: None,
                    utilization: None,
                };

                // Parse utilization if available
                if line.contains("%") {
                    if let Some(percent_pos) = line.find("%") {
                        let before_percent = &line[..percent_pos];
                        if let Some(space_pos) = before_percent.rfind(' ') {
                            if let Ok(util) = before_percent[space_pos + 1..].parse::<f32>() {
                                // Update the last GPU's utilization
                                if let Some(last_gpu) = gpus.last_mut() {
                                    last_gpu.utilization = Some(util);
                                } else {
                                    let mut gpu = gpu_info;
                                    gpu.utilization = Some(util);
                                    gpus.push(gpu);
                                    continue;
                                }
                            }
                        }
                    }
                }

                if gpus.is_empty() {
                    gpus.push(gpu_info);
                }
                break;
            }
        }

        Ok(gpus)
    }

    #[cfg(target_os = "windows")]
    fn detect_intel_windows(&self) -> Result<Vec<GpuInfo>> {
        // Try WMI query for Intel GPUs
        if let Ok(output) = std::process::Command::new("wmic")
            .args(&["path", "win32_VideoController", "get", "Name,AdapterRAM", "/format:csv"])
            .output()
        {
            let output_str = String::from_utf8_lossy(&output.stdout);
            return self.parse_intel_wmic_output(&output_str);
        }

        Ok(Vec::new())
    }

    #[cfg(target_os = "macos")]
    fn detect_intel_macos(&self) -> Result<Vec<GpuInfo>> {
        // Intel GPUs on macOS (older Macs)
        if let Ok(output) = std::process::Command::new("system_profiler")
            .args(&["SPDisplaysDataType"])
            .output()
        {
            let output_str = String::from_utf8_lossy(&output.stdout);
            return self.parse_intel_macos_output(&output_str);
        }

        Ok(Vec::new())
    }

    #[cfg(target_os = "windows")]
    fn parse_intel_wmic_output(&self, output: &str) -> Result<Vec<GpuInfo>> {
        let mut gpus = Vec::new();

        for line in output.lines().skip(1) { // Skip header
            let fields: Vec<&str> = line.split(',').collect();
            if fields.len() >= 2 {
                let name = fields[1].trim();
                if name.to_lowercase().contains("intel") {
                    let memory_bytes = fields[0].trim().parse::<u64>().unwrap_or(0);

                    let gpu_info = GpuInfo {
                        vendor: GpuVendor::Intel,
                        model: name.to_string(),
                        memory_total: memory_bytes,
                        memory_available: memory_bytes, // Approximate
                        compute_capability: None,
                        cuda_cores: None,
                        tensor_cores: None,
                        pcie_bandwidth: None,
                        power_limit: None,
                        temperature: None,
                        utilization: None,
                    };

                    gpus.push(gpu_info);
                }
            }
        }

        Ok(gpus)
    }

    #[cfg(target_os = "macos")]
    fn parse_intel_macos_output(&self, output: &str) -> Result<Vec<GpuInfo>> {
        let mut gpus = Vec::new();

        for line in output.lines() {
            let line = line.trim();
            if line.contains("Chipset Model:") {
                let model = line.split(':').nth(1).unwrap_or("").trim();
                if model.to_lowercase().contains("intel") {
                    let gpu_info = GpuInfo {
                        vendor: GpuVendor::Intel,
                        model: model.to_string(),
                        memory_total: 0,
                        memory_available: 0,
                        compute_capability: None,
                        cuda_cores: None,
                        tensor_cores: None,
                        pcie_bandwidth: None,
                        power_limit: None,
                        temperature: None,
                        utilization: None,
                    };
                    gpus.push(gpu_info);
                }
            }
        }

        Ok(gpus)
    }

    fn parse_intel_lspci_output(&self, output: &str) -> Result<Vec<GpuInfo>> {
        let mut gpus = Vec::new();

        for line in output.lines() {
            if line.contains("VGA") || line.contains("3D") {
                if line.to_lowercase().contains("intel") {
                    let parts: Vec<&str> = line.split(':').collect();
                    if parts.len() >= 3 {
                        let name = parts[2].trim().to_string();

                        let gpu_info = GpuInfo {
                            vendor: GpuVendor::Intel,
                            model: name,
                            memory_total: 0,
                            memory_available: 0,
                            compute_capability: None,
                            cuda_cores: None,
                            tensor_cores: None,
                            pcie_bandwidth: None,
                            power_limit: None,
                            temperature: None,
                            utilization: None,
                        };

                        gpus.push(gpu_info);
                    }
                }
            }
        }

        Ok(gpus)
    }

    fn detect_apple_gpus(&self) -> Result<Vec<GpuInfo>> {
        #[cfg(target_os = "macos")]
        {
            if let Ok(output) = std::process::Command::new("system_profiler")
                .args(&["SPDisplaysDataType"])
                .output()
            {
                let output_str = String::from_utf8_lossy(&output.stdout);
                return self.parse_apple_gpu_output(&output_str);
            }
        }

        Ok(Vec::new())
    }

    fn parse_apple_gpu_output(&self, output: &str) -> Result<Vec<GpuInfo>> {
        let mut gpus = Vec::new();

        let mut current_gpu: Option<GpuInfo> = None;

        for line in output.lines() {
            let line = line.trim();

            if line.contains("Chipset Model:") {
                if let Some(gpu) = current_gpu.take() {
                    gpus.push(gpu);
                }

                let model = line.split(':').nth(1).unwrap_or("").trim().to_string();
                current_gpu = Some(GpuInfo {
                    vendor: if model.contains("Apple") { GpuVendor::Apple } else { GpuVendor::Unknown(model.clone()) },
                    model,
                    memory_total: 0,
                    memory_available: 0,
                    compute_capability: None,
                    cuda_cores: None,
                    tensor_cores: None,
                    pcie_bandwidth: None,
                    power_limit: None,
                    temperature: None,
                    utilization: None,
                });
            }

            if let Some(ref mut gpu) = current_gpu {
                if line.contains("VRAM (Total):") {
                    let vram_str = line.split(':').nth(1).unwrap_or("").trim();
                    if let Some(mb_pos) = vram_str.find(" MB") {
                        if let Ok(vram_mb) = vram_str[..mb_pos].parse::<u64>() {
                            gpu.memory_total = vram_mb * 1024 * 1024;
                            gpu.memory_available = gpu.memory_total; // Approximate
                        }
                    }
                }
            }
        }

        if let Some(gpu) = current_gpu {
            gpus.push(gpu);
        }

        Ok(gpus)
    }

    fn detect_generic_gpus(&self) -> Result<Vec<GpuInfo>> {
        // Fallback generic detection
        let gpu_info = GpuInfo {
            vendor: GpuVendor::Unknown("Generic".to_string()),
            model: "Generic GPU".to_string(),
            memory_total: 0,
            memory_available: 0,
            compute_capability: None,
            cuda_cores: None,
            tensor_cores: None,
            pcie_bandwidth: None,
            power_limit: None,
            temperature: None,
            utilization: None,
        };

        Ok(vec![gpu_info])
    }
}

impl Default for GpuDetector {
    fn default() -> Self {
        Self::new()
    }
}