use crate::Result;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::time::Duration;
use sysinfo::{System, Process, Pid};
use tokio::sync::RwLock;
use tokio::time;
use tracing::{error, info, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceStatus {
    pub name: String,
    pub status: ServiceState,
    pub pid: Option<u32>,
    pub cpu_usage: f32,
    pub memory_mb: u64,
    pub uptime_seconds: u64,
    pub last_check: chrono::DateTime<chrono::Utc>,
    pub restart_count: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ServiceState {
    Running,
    Stopped,
    Starting,
    Failed,
    Unknown,
}

#[derive(Clone)]
pub struct MonitorConfig {
    pub check_interval_seconds: u64,
    pub max_restart_attempts: u32,
    pub restart_delay_seconds: u64,
    pub health_check_timeout_seconds: u64,
    pub services: Vec<ServiceConfig>,
}

#[derive(Clone)]
pub struct ServiceConfig {
    pub name: String,
    pub health_endpoint: String,
    pub start_command: String,
    pub stop_command: Option<String>,
    pub working_directory: Option<String>,
    pub env_vars: Vec<(String, String)>,
    pub expected_port: u16,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            check_interval_seconds: 30,
            max_restart_attempts: 3,
            restart_delay_seconds: 5,
            health_check_timeout_seconds: 10,
            services: vec![
                ServiceConfig {
                    name: "video_rss_server".to_string(),
                    health_endpoint: "http://localhost:8000/health".to_string(),
                    start_command: "cargo run --bin server".to_string(),
                    stop_command: None,
                    working_directory: None,
                    env_vars: vec![],
                    expected_port: 8000,
                },
            ],
        }
    }
}

pub struct ServiceMonitor {
    config: MonitorConfig,
    client: Client,
    system: Arc<RwLock<System>>,
    service_states: Arc<RwLock<Vec<ServiceStatus>>>,
}

impl ServiceMonitor {
    pub fn new(config: MonitorConfig) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(config.health_check_timeout_seconds))
            .build()
            .unwrap();

        let system = System::new_all();

        let service_states = config
            .services
            .iter()
            .map(|service| ServiceStatus {
                name: service.name.clone(),
                status: ServiceState::Unknown,
                pid: None,
                cpu_usage: 0.0,
                memory_mb: 0,
                uptime_seconds: 0,
                last_check: chrono::Utc::now(),
                restart_count: 0,
            })
            .collect();

        Self {
            config,
            client,
            system: Arc::new(RwLock::new(system)),
            service_states: Arc::new(RwLock::new(service_states)),
        }
    }

    pub async fn start_monitoring(&self) {
        info!("Starting service monitor...");

        let mut interval = time::interval(Duration::from_secs(self.config.check_interval_seconds));

        loop {
            interval.tick().await;
            self.check_all_services().await;
        }
    }

    async fn check_all_services(&self) {
        let services = self.config.services.clone();

        for (idx, service_config) in services.iter().enumerate() {
            let status = self.check_service(service_config).await;

            let mut states = self.service_states.write().await;
            if let Some(state) = states.get_mut(idx) {
                state.status = status.clone();
                state.last_check = chrono::Utc::now();

                // Update system metrics
                if let Some(pid) = state.pid {
                    let mut system = self.system.write().await;
                    system.refresh_processes();

                    if let Some(process) = system.processes().get(&Pid::from_u32(pid)) {
                        state.cpu_usage = process.cpu_usage();
                        state.memory_mb = process.memory() / 1024 / 1024;
                    }
                }

                // Handle service failures
                if status == ServiceState::Failed || status == ServiceState::Stopped {
                    warn!(
                        "Service '{}' is not running. Attempting restart...",
                        service_config.name
                    );

                    if state.restart_count < self.config.max_restart_attempts {
                        if self.restart_service(service_config).await.is_ok() {
                            state.restart_count += 1;
                            state.status = ServiceState::Starting;
                            info!(
                                "Service '{}' restart initiated (attempt {}/{})",
                                service_config.name, state.restart_count, self.config.max_restart_attempts
                            );
                        } else {
                            error!("Failed to restart service '{}'", service_config.name);
                            state.status = ServiceState::Failed;
                        }
                    } else {
                        error!(
                            "Service '{}' has exceeded maximum restart attempts",
                            service_config.name
                        );
                    }
                }
            }
        }
    }

    async fn check_service(&self, config: &ServiceConfig) -> ServiceState {
        // Try health check endpoint
        match self.client.get(&config.health_endpoint).send().await {
            Ok(response) if response.status().is_success() => {
                info!("Service '{}' health check passed", config.name);
                ServiceState::Running
            }
            Ok(response) => {
                warn!(
                    "Service '{}' health check failed with status: {}",
                    config.name,
                    response.status()
                );
                ServiceState::Failed
            }
            Err(e) => {
                warn!("Service '{}' health check error: {}", config.name, e);

                // Check if process is running on expected port
                if self.is_port_listening(config.expected_port).await {
                    ServiceState::Running
                } else {
                    ServiceState::Stopped
                }
            }
        }
    }

    async fn is_port_listening(&self, port: u16) -> bool {
        use std::net::{IpAddr, Ipv4Addr, SocketAddr, TcpStream};
        use std::time::Duration;

        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), port);

        match TcpStream::connect_timeout(&addr, Duration::from_secs(1)) {
            Ok(_) => true,
            Err(_) => false,
        }
    }

    async fn restart_service(&self, config: &ServiceConfig) -> Result<()> {
        // Stop the service if stop command is provided
        if let Some(stop_cmd) = &config.stop_command {
            info!("Stopping service '{}'...", config.name);
            self.execute_command(stop_cmd, config).await?;
            time::sleep(Duration::from_secs(self.config.restart_delay_seconds)).await;
        }

        // Start the service
        info!("Starting service '{}'...", config.name);
        self.execute_command(&config.start_command, config).await?;

        Ok(())
    }

    async fn execute_command(&self, command: &str, config: &ServiceConfig) -> Result<()> {
        let mut cmd = if cfg!(target_os = "windows") {
            let mut c = Command::new("cmd");
            c.args(&["/C", command]);
            c
        } else {
            let mut c = Command::new("sh");
            c.args(&["-c", command]);
            c
        };

        // Set working directory if specified
        if let Some(dir) = &config.working_directory {
            cmd.current_dir(dir);
        }

        // Set environment variables
        for (key, value) in &config.env_vars {
            cmd.env(key, value);
        }

        cmd.stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| crate::error::VideoRssError::Io(e))?;

        Ok(())
    }

    pub async fn get_status(&self) -> Vec<ServiceStatus> {
        self.service_states.read().await.clone()
    }

    pub async fn stop_service(&self, service_name: &str) -> Result<()> {
        let service_config = self
            .config
            .services
            .iter()
            .find(|s| s.name == service_name)
            .ok_or_else(|| {
                crate::error::VideoRssError::Config(format!("Service '{}' not found", service_name))
            })?;

        if let Some(stop_cmd) = &service_config.stop_command {
            self.execute_command(stop_cmd, service_config).await?;
        }

        Ok(())
    }

    pub async fn start_service(&self, service_name: &str) -> Result<()> {
        let service_config = self
            .config
            .services
            .iter()
            .find(|s| s.name == service_name)
            .ok_or_else(|| {
                crate::error::VideoRssError::Config(format!("Service '{}' not found", service_name))
            })?;

        self.execute_command(&service_config.start_command, service_config)
            .await
    }
}

// HTTP API for monitoring
pub mod api {
    use super::*;
    use axum::{
        extract::State,
        response::IntoResponse,
        routing::{get, post},
        Json, Router,
    };

    pub fn create_monitor_router(monitor: Arc<ServiceMonitor>) -> Router {
        Router::new()
            .route("/monitor/status", get(get_monitor_status))
            .route("/monitor/services/:name/start", post(start_service))
            .route("/monitor/services/:name/stop", post(stop_service))
            .route("/monitor/services/:name/restart", post(restart_service))
            .with_state(monitor)
    }

    async fn get_monitor_status(State(monitor): State<Arc<ServiceMonitor>>) -> impl IntoResponse {
        let status = monitor.get_status().await;
        Json(status)
    }

    async fn start_service(
        State(monitor): State<Arc<ServiceMonitor>>,
        axum::extract::Path(name): axum::extract::Path<String>,
    ) -> impl IntoResponse {
        match monitor.start_service(&name).await {
            Ok(_) => Json(serde_json::json!({
                "status": "success",
                "message": format!("Service '{}' start command executed", name)
            })),
            Err(e) => Json(serde_json::json!({
                "status": "error",
                "message": format!("Failed to start service '{}': {}", name, e)
            })),
        }
    }

    async fn stop_service(
        State(monitor): State<Arc<ServiceMonitor>>,
        axum::extract::Path(name): axum::extract::Path<String>,
    ) -> impl IntoResponse {
        match monitor.stop_service(&name).await {
            Ok(_) => Json(serde_json::json!({
                "status": "success",
                "message": format!("Service '{}' stop command executed", name)
            })),
            Err(e) => Json(serde_json::json!({
                "status": "error",
                "message": format!("Failed to stop service '{}': {}", name, e)
            })),
        }
    }

    async fn restart_service(
        State(monitor): State<Arc<ServiceMonitor>>,
        axum::extract::Path(name): axum::extract::Path<String>,
    ) -> impl IntoResponse {
        // Stop then start
        if let Err(e) = monitor.stop_service(&name).await {
            return Json(serde_json::json!({
                "status": "error",
                "message": format!("Failed to stop service '{}': {}", name, e)
            }));
        }

        tokio::time::sleep(Duration::from_secs(2)).await;

        match monitor.start_service(&name).await {
            Ok(_) => Json(serde_json::json!({
                "status": "success",
                "message": format!("Service '{}' restarted successfully", name)
            })),
            Err(e) => Json(serde_json::json!({
                "status": "error",
                "message": format!("Failed to restart service '{}': {}", name, e)
            })),
        }
    }
}