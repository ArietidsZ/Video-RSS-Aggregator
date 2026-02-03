use std::env;

use anyhow::{anyhow, Result};

#[derive(Clone, Debug)]
pub struct AppConfig {
    pub bind_address: String,
    pub database_url: String,
    pub api_key: Option<String>,
    pub storage_dir: String,
}

impl AppConfig {
    pub fn from_env() -> Result<Self> {
        let database_url =
            env::var("DATABASE_URL").map_err(|_| anyhow!("DATABASE_URL must be set"))?;
        let bind_address = env::var("BIND_ADDRESS").unwrap_or_else(|_| "0.0.0.0:8080".to_string());
        let api_key = env::var("API_KEY").ok();
        let storage_dir = env::var("VRA_STORAGE_DIR").unwrap_or_else(|_| ".data".to_string());

        Ok(Self {
            bind_address,
            database_url,
            api_key,
            storage_dir,
        })
    }
}
