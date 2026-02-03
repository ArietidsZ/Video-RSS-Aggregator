use std::net::SocketAddr;

use anyhow::Result;
use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;

use crate::accel::AccelConfig;
use crate::api::{self, AppState};
use crate::config::AppConfig;
use crate::pipeline::Pipeline;
use crate::storage::Database;

mod accel;
mod api;
mod auth;
mod config;
mod ffi;
mod media;
mod pipeline;
mod rss;
mod setup;
mod storage;
mod summarize;
mod transcribe;
mod verify;

#[derive(Parser)]
#[command(name = "video-rss-aggregator")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Serve {
        #[arg(long)]
        bind: Option<String>,
    },
    Verify,
    Setup,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();
    match cli.command {
        Command::Serve { bind } => serve(bind).await,
        Command::Verify => verify::run().await,
        Command::Setup => setup::run().await,
    }
}

async fn serve(bind: Option<String>) -> Result<()> {
    let mut config = AppConfig::from_env()?;
    if let Some(bind) = bind {
        config.bind_address = bind;
    }

    let db = Database::connect(&config.database_url).await?;
    db.migrate().await?;

    let accel = AccelConfig::from_env();
    let pipeline = Pipeline::new(&config, &accel, db).await?;
    let state = AppState::new(pipeline, config.api_key.clone());
    let app = api::router(state);

    let addr: SocketAddr = config.bind_address.parse()?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
