use std::env;
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Result};
use tokio::process::Command;

pub async fn run() -> Result<()> {
    ensure_macos_apple_silicon()?;
    ensure_tool("cmake").await?;
    ensure_tool("git").await?;
    ensure_xcode_cli().await?;

    let data_dir = PathBuf::from(".data");
    let vendor_dir = data_dir.join("vendor");
    let build_dir = data_dir.join("build");
    let backends_dir = data_dir.join("backends");

    fs::create_dir_all(&vendor_dir)?;
    fs::create_dir_all(&build_dir)?;
    fs::create_dir_all(&backends_dir)?;

    let whisper_repo = env::var("VRA_WHISPER_REPO")
        .unwrap_or_else(|_| "https://github.com/ggerganov/whisper.cpp.git".to_string());
    let llama_repo = env::var("VRA_LLAMA_REPO")
        .unwrap_or_else(|_| "https://github.com/ggerganov/llama.cpp.git".to_string());

    let whisper_dir = vendor_dir.join("whisper.cpp");
    let llama_dir = vendor_dir.join("llama.cpp");

    clone_repo(&whisper_repo, &whisper_dir).await?;
    clone_repo(&llama_repo, &llama_dir).await?;

    checkout_ref(&whisper_dir, "VRA_WHISPER_REF").await?;
    checkout_ref(&llama_dir, "VRA_LLAMA_REF").await?;

    let whisper_build = build_dir.join("whisper-mps");
    let llama_build = build_dir.join("llama-coreml");

    build_whisper_mps(&whisper_dir, &whisper_build).await?;
    build_llama_coreml(&llama_dir, &llama_build).await?;

    let whisper_lib_dir = find_library_dir(&whisper_build, "libwhisper")?;
    let llama_lib_dir = find_library_dir(&llama_build, "libllama")?;

    let whisper_include_dir = whisper_dir.join("include");
    let llama_include_dir = llama_dir.join("include");

    let backend_build_mps = build_dir.join("backend-mps");
    let backend_build_coreml = build_dir.join("backend-coreml");

    build_backend(
        "mps",
        &backend_build_mps,
        &whisper_include_dir,
        &whisper_lib_dir,
        &llama_include_dir,
        &llama_lib_dir,
    )
    .await?;

    build_backend(
        "coreml",
        &backend_build_coreml,
        &whisper_include_dir,
        &whisper_lib_dir,
        &llama_include_dir,
        &llama_lib_dir,
    )
    .await?;

    let mps_lib = find_file_by_name(&backend_build_mps, "libvra_mps_backend.dylib")?;
    let coreml_lib = find_file_by_name(&backend_build_coreml, "libvra_coreml_backend.dylib")?;

    let mps_target = backends_dir.join("libvra_mps_backend.dylib");
    let coreml_target = backends_dir.join("libvra_coreml_backend.dylib");

    fs::copy(&mps_lib, &mps_target)?;
    fs::copy(&coreml_lib, &coreml_target)?;

    println!("Backends ready in {}", backends_dir.display());
    println!("Transcribe: {}", mps_target.display());
    println!("Summarize: {}", coreml_target.display());
    println!("Run: cargo run --release -- serve --bind 0.0.0.0:8080");

    Ok(())
}

fn ensure_macos_apple_silicon() -> Result<()> {
    if !cfg!(target_os = "macos") {
        return Err(anyhow!("Setup is only supported on macOS"));
    }
    if env::consts::ARCH != "aarch64" {
        return Err(anyhow!("Apple silicon is required for this setup"));
    }
    Ok(())
}

async fn ensure_tool(name: &str) -> Result<()> {
    let status = Command::new(name)
        .arg("--version")
        .status()
        .await
        .map_err(|_| anyhow!("Missing tool: {}", name))?;

    if !status.success() {
        return Err(anyhow!("Failed to run {} --version", name));
    }

    Ok(())
}

async fn ensure_xcode_cli() -> Result<()> {
    let status = Command::new("xcode-select")
        .arg("-p")
        .status()
        .await
        .map_err(|_| anyhow!("xcode-select not found"))?;

    if !status.success() {
        return Err(anyhow!("Xcode Command Line Tools are required (run: xcode-select --install)"));
    }

    Ok(())
}

async fn clone_repo(repo: &str, destination: &Path) -> Result<()> {
    if destination.exists() {
        return Ok(());
    }

    let status = Command::new("git")
        .arg("clone")
        .arg("--depth")
        .arg("1")
        .arg(repo)
        .arg(destination)
        .status()
        .await?;

    if !status.success() {
        return Err(anyhow!("git clone failed for {}", repo));
    }

    Ok(())
}

async fn checkout_ref(repo_dir: &Path, env_key: &str) -> Result<()> {
    let Ok(reference) = env::var(env_key) else {
        return Ok(());
    };
    if reference.trim().is_empty() {
        return Ok(());
    }

    let status = Command::new("git")
        .arg("-C")
        .arg(repo_dir)
        .arg("checkout")
        .arg(reference)
        .status()
        .await?;

    if !status.success() {
        return Err(anyhow!("git checkout failed for {}", env_key));
    }

    Ok(())
}

async fn build_whisper_mps(source: &Path, build: &Path) -> Result<()> {
    let mut args = vec![
        "-DCMAKE_BUILD_TYPE=Release".to_string(),
        "-DWHISPER_METAL=ON".to_string(),
        "-DWHISPER_BUILD_EXAMPLES=OFF".to_string(),
        "-DWHISPER_BUILD_TESTS=OFF".to_string(),
        "-DWHISPER_BUILD_SERVER=OFF".to_string(),
    ];

    let extra = env::var("VRA_WHISPER_CMAKE_ARGS").ok();
    if let Some(extra) = extra.as_deref() {
        args.extend(extra.split_whitespace().map(|value| value.to_string()));
    }

    cmake_configure(source, build, &args).await?;
    cmake_build(build).await
}

async fn build_llama_coreml(source: &Path, build: &Path) -> Result<()> {
    let extra = env::var("VRA_LLAMA_CMAKE_ARGS").ok();

    let mut primary = vec![
        "-DCMAKE_BUILD_TYPE=Release".to_string(),
        "-DGGML_METAL=ON".to_string(),
        "-DGGML_COREML=ON".to_string(),
        "-DLLAMA_BUILD_EXAMPLES=OFF".to_string(),
        "-DLLAMA_BUILD_TESTS=OFF".to_string(),
        "-DLLAMA_BUILD_SERVER=OFF".to_string(),
    ];
    if let Some(extra) = extra.as_deref() {
        primary.extend(extra.split_whitespace().map(|value| value.to_string()));
    }

    if cmake_configure(source, build, &primary).await.is_ok() {
        return cmake_build(build).await;
    }

    if build.exists() {
        let _ = fs::remove_dir_all(build);
    }

    let mut fallback = vec![
        "-DCMAKE_BUILD_TYPE=Release".to_string(),
        "-DLLAMA_METAL=ON".to_string(),
        "-DLLAMA_COREML=ON".to_string(),
        "-DLLAMA_BUILD_EXAMPLES=OFF".to_string(),
        "-DLLAMA_BUILD_TESTS=OFF".to_string(),
        "-DLLAMA_BUILD_SERVER=OFF".to_string(),
    ];
    if let Some(extra) = extra.as_deref() {
        fallback.extend(extra.split_whitespace().map(|value| value.to_string()));
    }

    cmake_configure(source, build, &fallback).await?;
    cmake_build(build).await
}

async fn build_backend(
    kind: &str,
    build_dir: &Path,
    whisper_include: &Path,
    whisper_lib: &Path,
    llama_include: &Path,
    llama_lib: &Path,
) -> Result<()> {
    let mut args = vec![
        format!("-DVRA_BACKEND_KIND={}", kind),
        format!("-DWHISPER_CPP_INCLUDE_DIR={}", whisper_include.display()),
        format!("-DWHISPER_CPP_LIB_DIR={}", whisper_lib.display()),
        format!("-DLLAMA_CPP_INCLUDE_DIR={}", llama_include.display()),
        format!("-DLLAMA_CPP_LIB_DIR={}", llama_lib.display()),
    ];

    if let Ok(extra) = env::var("VRA_BACKEND_CMAKE_ARGS") {
        if !extra.trim().is_empty() {
            args.extend(extra.split_whitespace().map(|value| value.to_string()));
        }
    }

    cmake_configure(Path::new("backends"), build_dir, &args).await?;
    cmake_build(build_dir).await
}

async fn cmake_configure(source: &Path, build: &Path, args: &[String]) -> Result<()> {
    let status = Command::new("cmake")
        .arg("-S")
        .arg(source)
        .arg("-B")
        .arg(build)
        .args(args)
        .status()
        .await?;

    if !status.success() {
        return Err(anyhow!("cmake configure failed for {}", build.display()));
    }

    Ok(())
}

async fn cmake_build(build: &Path) -> Result<()> {
    let status = Command::new("cmake")
        .arg("--build")
        .arg(build)
        .arg("--config")
        .arg("Release")
        .status()
        .await?;

    if !status.success() {
        return Err(anyhow!("cmake build failed for {}", build.display()));
    }

    Ok(())
}

fn find_library_dir(root: &Path, prefix: &str) -> Result<PathBuf> {
    let file = find_file_by_prefix(root, prefix)?;
    file.parent()
        .map(|dir| dir.to_path_buf())
        .ok_or_else(|| anyhow!("Library dir not found for {}", prefix))
}

fn find_file_by_name(root: &Path, name: &str) -> Result<PathBuf> {
    if !root.exists() {
        return Err(anyhow!("Path not found: {}", root.display()));
    }

    let mut stack = vec![root.to_path_buf()];
    while let Some(path) = stack.pop() {
        for entry in fs::read_dir(&path)? {
            let entry = entry?;
            let file_type = entry.file_type()?;
            let entry_path = entry.path();
            if file_type.is_dir() {
                stack.push(entry_path);
            } else if file_type.is_file() {
                if let Some(file_name) = entry_path.file_name().and_then(OsStr::to_str) {
                    if file_name == name {
                        return Ok(entry_path);
                    }
                }
            }
        }
    }

    Err(anyhow!("File {} not found in {}", name, root.display()))
}

fn find_file_by_prefix(root: &Path, prefix: &str) -> Result<PathBuf> {
    if !root.exists() {
        return Err(anyhow!("Path not found: {}", root.display()));
    }

    let mut stack = vec![root.to_path_buf()];
    while let Some(path) = stack.pop() {
        for entry in fs::read_dir(&path)? {
            let entry = entry?;
            let file_type = entry.file_type()?;
            let entry_path = entry.path();
            if file_type.is_dir() {
                stack.push(entry_path);
            } else if file_type.is_file() {
                if let Some(file_name) = entry_path.file_name().and_then(OsStr::to_str) {
                    if file_name.starts_with(prefix)
                        && (file_name.ends_with(".dylib") || file_name.ends_with(".a"))
                    {
                        return Ok(entry_path);
                    }
                }
            }
        }
    }

    Err(anyhow!("Library {} not found in {}", prefix, root.display()))
}
