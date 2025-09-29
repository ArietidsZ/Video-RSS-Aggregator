pub mod types;
pub mod cpu;
pub mod memory;
pub mod gpu;
pub mod profiler;
pub mod config;
pub mod thermal;

pub use types::*;
pub use profiler::HardwareProfiler;
pub use config::OptimalConfig;
pub use thermal::{ThermalMonitor, PerformanceAdjuster, DynamicAdjustment, QualityLevel};

use thiserror::Error;

#[derive(Error, Debug)]
pub enum HardwareError {
    #[error("Failed to detect CPU information: {0}")]
    CpuDetection(String),

    #[error("Failed to detect memory information: {0}")]
    MemoryDetection(String),

    #[error("Failed to detect GPU information: {0}")]
    GpuDetection(String),

    #[error("Failed to initialize hardware monitoring: {0}")]
    MonitoringInit(String),

    #[error("Configuration generation failed: {0}")]
    ConfigGeneration(String),

    #[error("System error: {0}")]
    System(String),

    #[error("Thermal monitoring error: {0}")]
    ThermalMonitoring(String),
}

pub type Result<T> = std::result::Result<T, HardwareError>;
