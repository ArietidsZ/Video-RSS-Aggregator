# Rust Setup Guide for Video RSS Aggregator

## Current Status

**Rust is NOT installed on this system.** This is why:
- ❌ VS Code shows NO errors (Rust Analyzer cannot analyze code)
- ❌ Rust Analyzer logs show `program not found` errors
- ❌ No `.cargo` or `.rustup` directories exist

## Root Cause Analysis

According to ERROR_ANALYSIS_REPORT.md, this codebase has **358 compilation errors** that you cannot see because Rust Analyzer requires the Rust toolchain to function.

## Step 1: Install Rust

### Option A: Using Rustup (Recommended)

1. Download and run the Rust installer:
   ```
   https://rustup.rs/
   ```

2. During installation:
   - Choose default installation
   - This will install both `rustc` and `cargo`
   - Automatically adds Rust to your PATH

3. After installation, restart your computer (or at minimum, restart VS Code)

### Option B: Using Package Manager

**Windows (via Chocolatey):**
```powershell
choco install rust
```

**Windows (via Scoop):**
```powershell
scoop install rust
```

## Step 2: Verify Installation

Open a NEW terminal/PowerShell window and run:

```powershell
cargo --version
rustc --version
```

Expected output:
```
cargo 1.XX.X
rustc 1.XX.X
```

## Step 3: Configure VS Code

The `.vscode/settings.json` file has been created with optimal settings for this multi-project workspace.

### Key Configuration:
- **linkedProjects**: Tells Rust Analyzer about all 12 Rust projects
- **checkOnSave**: Runs `cargo check` automatically
- **excludeDirs**: Ignores `target/` directories for better performance

## Step 4: Reload VS Code

After installing Rust:

1. Open Command Palette: `Ctrl+Shift+P`
2. Run: `Developer: Reload Window`
3. Wait for Rust Analyzer to index all projects (~2-5 minutes)

## Step 5: Verify Rust Analyzer is Working

Check the Rust Analyzer status:

1. Look at bottom-right status bar for "rust-analyzer" indicator
2. Open: `View` → `Output` → Select "Rust Analyzer Language Server"
3. You should see indexing progress, NOT "program not found" errors

## What to Expect After Setup

Once Rust is installed and Rust Analyzer loads, you will see:

### In VS Code Problems Panel (Ctrl+Shift+M):
- **~358 errors total** across all projects
- **79 SQLx errors** in `security/` (requires database setup)
- **70 errors** in `api-gateway/` (dependency/type issues)
- **50 errors** in `performance-monitor/` (trait bounds)
- **Minor errors** in other services

### Error Categories:
1. **SQLx Query Cache Missing** (22% of errors)
   - Fix: Setup PostgreSQL or use runtime queries

2. **Type Mismatches** (Governor rate limiter, etc.)
   - Fix: Update dependency APIs

3. **Missing Dependencies** (minhash-rs, proxy-agent, etc.)
   - Fix: Update Cargo.toml files

4. **Trait Bound Issues** (Argon2, Tower/Axum middleware)
   - Fix: Add error conversions and update middleware composition

## Next Steps After Installation

1. **Fix SQLx Issues** (Highest impact - 79 errors):
   ```bash
   cd security
   # Either setup database:
   export DATABASE_URL="postgresql://user:pass@localhost/db"
   cargo sqlx database create
   cargo sqlx migrate run
   cargo sqlx prepare

   # OR convert to runtime queries
   ```

2. **Update Dependencies**:
   ```bash
   # In each project directory:
   cargo update
   cargo check
   ```

3. **Review ERROR_ANALYSIS_REPORT.md** for detailed fix instructions

## Troubleshooting

### Rust Analyzer still shows "program not found"

1. Ensure Rust is in PATH:
   ```powershell
   $env:PATH -split ';' | Select-String cargo
   ```

2. Manually add Rust to VS Code settings if needed:
   ```json
   {
     "rust-analyzer.server.extraEnv": {
       "PATH": "C:\\Users\\zhong\\.cargo\\bin;${env:PATH}"
     }
   }
   ```

3. Reload VS Code window

### Rust Analyzer is slow/unresponsive

This is normal for projects with many errors. Performance will improve as errors are fixed.

Temporary solutions:
- Disable specific projects in `linkedProjects`
- Increase `rust-analyzer.cargo.buildScripts.overrideCommand` timeout
- Close unused Rust files

## Additional Resources

- Rust Installation: https://www.rust-lang.org/tools/install
- Rust Analyzer Docs: https://rust-analyzer.github.io/
- SQLx Setup: https://github.com/launchbadge/sqlx
- Project Error Report: ERROR_ANALYSIS_REPORT.md
