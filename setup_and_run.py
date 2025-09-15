#!/usr/bin/env python3
"""
Universal Setup and Run Script for Video RSS Aggregator
Works on Windows, macOS, and Linux
One-click demo with automatic dependency installation
"""

import os
import sys
import subprocess
import platform
import time
import socket
import signal
import atexit
import json
import urllib.request
import urllib.error
import tempfile
import shutil
from pathlib import Path

# Store process handles for cleanup
processes = []

def check_port(port):
    """Check if a port is available"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('localhost', port))
    sock.close()
    return result != 0

def kill_port(port):
    """Kill process using a specific port"""
    system = platform.system()
    try:
        if system == "Windows":
            # Windows command to find and kill process
            cmd = f'netstat -ano | findstr :{port}'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.stdout:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    parts = line.split()
                    if len(parts) > 4:
                        pid = parts[-1]
                        subprocess.run(f'taskkill /PID {pid} /F', shell=True, capture_output=True)
        else:
            # Unix/Linux/macOS command
            subprocess.run(f'lsof -ti:{port} | xargs kill -9', shell=True, capture_output=True)
    except:
        pass

def cleanup():
    """Clean up all processes on exit"""
    print("\n🛑 Stopping all services...")
    for process in processes:
        try:
            process.terminate()
            process.wait(timeout=5)
        except:
            try:
                process.kill()
            except:
                pass
    print("✅ All services stopped")

def check_command(command):
    """Check if a command exists"""
    try:
        subprocess.run([command, '--version'], capture_output=True, check=False)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def download_file(url, destination):
    """Download a file with progress indicator"""
    try:
        print(f"   Downloading from {url}...")
        with urllib.request.urlopen(url) as response:
            total_size = int(response.headers.get('Content-Length', 0))
            downloaded = 0
            block_size = 8192

            with open(destination, 'wb') as f:
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    downloaded += len(buffer)
                    f.write(buffer)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"   Progress: {percent:.1f}%", end='\r')
            print()
            return True
    except Exception as e:
        print(f"   Download failed: {e}")
        return False

def install_nodejs_windows():
    """Download and install Node.js on Windows"""
    print("\n📦 Installing Node.js for Windows...")

    # Node.js download URL
    node_url = "https://nodejs.org/dist/v20.11.0/node-v20.11.0-x64.msi"
    temp_file = os.path.join(tempfile.gettempdir(), "node_installer.msi")

    try:
        # Download Node.js installer
        if download_file(node_url, temp_file):
            print("   Installing Node.js (this may take a moment)...")
            # Run installer silently
            subprocess.run(['msiexec', '/i', temp_file, '/qn'], check=True)
            print("   ✅ Node.js installed successfully")
            print("   ⚠️  You may need to restart your terminal for changes to take effect")
            return True
    except Exception as e:
        print(f"   ❌ Failed to install Node.js: {e}")
        print("   Please install manually from: https://nodejs.org/")
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

    return False

def install_python_packages():
    """Install required Python packages"""
    packages = ['fastapi', 'uvicorn', 'aiohttp']

    print("📦 Checking Python packages...")

    # Check which packages are missing
    missing_packages = []
    for package in packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print(f"   Installing missing packages: {', '.join(missing_packages)}")

        # Upgrade pip first
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'],
                      capture_output=True, check=False)

        # Install missing packages
        for package in missing_packages:
            print(f"   Installing {package}...")
            result = subprocess.run([sys.executable, '-m', 'pip', 'install', package],
                                  capture_output=True, text=True, check=False)
            if result.returncode != 0:
                print(f"   ⚠️  Failed to install {package}: {result.stderr}")
                print(f"   Try manually: pip install {package}")
            else:
                print(f"   ✅ {package} installed")
    else:
        print("   ✅ All Python packages already installed")

    return True

def install_npm_packages():
    """Install npm packages for frontend"""
    # Check if npm is available
    if not check_command('npm'):
        print("   ⚠️  npm not available, skipping frontend packages")
        return False

    frontend_path = Path('frontend')

    if not frontend_path.exists():
        print("   ⚠️  Frontend directory not found, skipping frontend setup")
        return False

    # Check if node_modules exists
    node_modules = frontend_path / 'node_modules'
    package_json = frontend_path / 'package.json'

    if not package_json.exists():
        print("   ⚠️  package.json not found, skipping frontend setup")
        return False

    if node_modules.exists() and any(node_modules.iterdir()):
        print("   ✅ Frontend packages already installed")
        return True

    print("   Installing frontend packages (this may take a moment)...")

    # Try npm ci first (faster, uses package-lock.json)
    if (frontend_path / 'package-lock.json').exists():
        result = subprocess.run(['npm', 'ci'], cwd=str(frontend_path),
                              capture_output=True, text=True, check=False)
        if result.returncode == 0:
            print("   ✅ Frontend packages installed")
            return True

    # Fall back to npm install
    result = subprocess.run(['npm', 'install'], cwd=str(frontend_path),
                          capture_output=True, text=True, check=False)

    if result.returncode == 0:
        print("   ✅ Frontend packages installed")
        return True
    else:
        print(f"   ⚠️  Failed to install frontend packages")
        return False

def check_and_install_nodejs():
    """Check if Node.js is installed and attempt to install if missing"""
    if check_command('node') and check_command('npm'):
        return True

    system = platform.system()

    print("\n⚠️  Node.js is not installed")
    print("   Node.js is optional for the frontend interface")
    print("   The backend API will still work without it\n")

    if system == "Windows":
        response = input("Would you like to install Node.js automatically? (y/n, default=n): ").strip().lower()
        if response == 'y':
            if install_nodejs_windows():
                # Try to refresh PATH
                os.environ['PATH'] = os.environ['PATH'] + ';' + r'C:\Program Files\nodejs'
                if check_command('node'):
                    return True
            print("   Installation failed. You can install manually from: https://nodejs.org/")
        else:
            print("   Skipping Node.js installation")
            print("   You can install it later from: https://nodejs.org/")

    elif system == "Darwin":  # macOS
        response = input("Would you like to try installing Node.js? (y/n, default=n): ").strip().lower()
        if response == 'y':
            # Check if Homebrew is installed
            if check_command('brew'):
                print("   Using Homebrew to install Node.js...")
                result = subprocess.run(['brew', 'install', 'node'],
                                      capture_output=True, text=True, check=False)
                if result.returncode == 0:
                    print("   ✅ Node.js installed successfully")
                    return True
                else:
                    print("   Homebrew installation failed")

            print("   Please install Node.js from: https://nodejs.org/")
            print("   Or use Homebrew: brew install node")
        else:
            print("   Skipping Node.js installation")

    else:  # Linux
        print("   To install Node.js on Linux:")
        print("   Ubuntu/Debian: sudo apt-get install nodejs npm")
        print("   Fedora: sudo dnf install nodejs npm")
        print("   Or download from: https://nodejs.org/")
        print("\n   Continuing without Node.js...")

    return False

def load_config():
    """Load configuration from .env file"""
    config = {}
    env_file = Path('.env')

    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    config[key.strip()] = value.strip()

    return config

def save_config(config):
    """Save configuration to .env file"""
    env_file = Path('.env')

    # Read existing content
    lines = []
    if env_file.exists():
        with open(env_file, 'r') as f:
            lines = f.readlines()

    # Update or add configuration values
    updated = False
    new_lines = []

    for line in lines:
        updated_line = False
        for key, value in config.items():
            if line.strip().startswith(f"{key}="):
                new_lines.append(f"{key}={value}\n")
                updated_line = True
                break
        if not updated_line:
            new_lines.append(line)

    # Add any new keys that weren't in the file
    for key, value in config.items():
        if not any(line.strip().startswith(f"{key}=") for line in lines):
            new_lines.append(f"{key}={value}\n")

    # Write back to file
    with open(env_file, 'w') as f:
        f.writelines(new_lines)

def check_bilibili_config():
    """Check and configure Bilibili settings"""
    config = load_config()

    # Check if Bilibili configuration is needed
    needs_config = False
    if 'BILIBILI_SESSDATA' not in config or config.get('BILIBILI_SESSDATA') == 'your_sessdata_here':
        needs_config = True
    if 'BILIBILI_BILI_JCT' not in config or config.get('BILIBILI_BILI_JCT') == 'your_bili_jct_here':
        needs_config = True
    if 'BILIBILI_BUVID3' not in config or config.get('BILIBILI_BUVID3') == 'your_buvid3_here':
        needs_config = True

    if needs_config:
        print("\n" + "="*50)
        print("📺 Bilibili Configuration (Optional)")
        print("="*50)
        print("\nTo fetch real data from Bilibili, you can provide cookie values.")
        print("Skip this step to use demo data.\n")

        response = input("Configure Bilibili now? (y/n, default=n): ").strip().lower()

        if response == 'y':
            print("\nTo get these values:")
            print("1. Open https://www.bilibili.com in your browser")
            print("2. Log in to your account")
            print("3. Open Developer Tools (F12)")
            print("4. Go to Application/Storage → Cookies")
            print("5. Find these cookie values:\n")

            # Ask for SESSDATA
            sessdata = input("Enter SESSDATA cookie (or press Enter to skip): ").strip()
            if not sessdata:
                sessdata = "demo_mode"

            # Ask for bili_jct
            bili_jct = input("Enter bili_jct cookie (or press Enter to skip): ").strip()
            if not bili_jct:
                bili_jct = "demo_mode"

            # Ask for buvid3
            buvid3 = input("Enter buvid3 cookie (or press Enter to skip): ").strip()
            if not buvid3:
                buvid3 = "demo_mode"

            # Save configuration
            if sessdata != "demo_mode" or bili_jct != "demo_mode" or buvid3 != "demo_mode":
                save_choice = input("\nSave these settings? (y/n): ").strip().lower()
                if save_choice == 'y':
                    config['BILIBILI_SESSDATA'] = sessdata
                    config['BILIBILI_BILI_JCT'] = bili_jct
                    config['BILIBILI_BUVID3'] = buvid3
                    save_config(config)
                    print("✅ Configuration saved")

            # Set environment variables
            os.environ['BILIBILI_SESSDATA'] = sessdata
            os.environ['BILIBILI_BILI_JCT'] = bili_jct
            os.environ['BILIBILI_BUVID3'] = buvid3
        else:
            print("✅ Using demo mode")
            os.environ['BILIBILI_SESSDATA'] = 'demo_mode'
            os.environ['BILIBILI_BILI_JCT'] = 'demo_mode'
            os.environ['BILIBILI_BUVID3'] = 'demo_mode'
    else:
        print("✅ Bilibili configuration found")
        # Set environment variables from config
        os.environ['BILIBILI_SESSDATA'] = config.get('BILIBILI_SESSDATA', '')
        os.environ['BILIBILI_BILI_JCT'] = config.get('BILIBILI_BILI_JCT', '')
        os.environ['BILIBILI_BUVID3'] = config.get('BILIBILI_BUVID3', '')

def install_dependencies():
    """Install all dependencies automatically"""
    print("\n📦 Installing Dependencies...")

    # Install Python packages
    if not install_python_packages():
        print("   ⚠️  Python packages installation had issues")
        return False

    # Install frontend packages (optional)
    install_npm_packages()  # Don't fail if npm packages can't be installed

    print("✅ Dependencies ready")
    return True

def start_backend():
    """Start the backend server"""
    print("   Starting backend server (port 8000)...")

    env = os.environ.copy()
    env['PYTHONPATH'] = '.'

    # Try to suppress warnings
    env['PYTHONWARNINGS'] = 'ignore'

    process = subprocess.Popen(
        [sys.executable, '-m', 'uvicorn', 'simple_backend:app', '--host', '0.0.0.0', '--port', '8000'],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    processes.append(process)
    return process

def start_frontend():
    """Start the frontend server"""
    print("   Starting frontend server (port 3000)...")

    frontend_path = Path('frontend')
    if not frontend_path.exists():
        print("   ⚠️  Frontend directory not found")
        return None

    # Set environment to production to avoid development warnings
    env = os.environ.copy()
    env['CI'] = 'true'  # Suppress warnings

    # Windows needs special handling for npm
    if platform.system() == "Windows":
        process = subprocess.Popen(
            'npm start',
            cwd=str(frontend_path),
            shell=True,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    else:
        process = subprocess.Popen(
            ['npm', 'start'],
            cwd=str(frontend_path),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

    processes.append(process)
    return process

def check_health():
    """Check if services are healthy"""
    try:
        with urllib.request.urlopen('http://localhost:8000/health', timeout=5) as response:
            if response.status == 200:
                return True
    except (urllib.error.URLError, TimeoutError):
        pass
    return False

def open_browser():
    """Open the browser to the frontend"""
    import webbrowser
    webbrowser.open('http://localhost:3000')

def main():
    """Main setup and run function"""
    print("="*50)
    print("🚀 Video RSS Aggregator - Competition Demo")
    print("   Automatic Setup & Dependency Installation")
    print("="*50)
    print()

    # Register cleanup
    atexit.register(cleanup)
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
    if platform.system() != "Windows":
        signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))

    # Check Python version
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required")
        print(f"   Current version: {sys.version}")
        input("Press Enter to exit...")
        sys.exit(1)

    print("📋 Checking requirements...")

    # Check and install Node.js if needed
    nodejs_available = check_and_install_nodejs()

    if nodejs_available:
        print("✅ Node.js is available")
    else:
        print("✅ Continuing without Node.js (backend-only mode)")

    print("✅ Requirements checked")

    # Check and configure Bilibili settings
    check_bilibili_config()

    # Install dependencies
    if not install_dependencies():
        print("⚠️  Some dependencies failed to install")
        response = input("Continue anyway? (y/n): ").strip().lower()
        if response != 'y':
            sys.exit(1)

    print()

    # Check and free ports
    print("🔍 Checking ports...")
    if not check_port(8000):
        print("   Port 8000 in use, freeing it...")
        kill_port(8000)
        time.sleep(1)

    if not check_port(3000):
        print("   Port 3000 in use, freeing it...")
        kill_port(3000)
        time.sleep(1)

    print("✅ Ports ready")
    print()

    # Start services
    print("🌐 Starting services...")

    backend = start_backend()
    if not backend:
        print("❌ Failed to start backend")
        sys.exit(1)

    time.sleep(3)  # Wait for backend to start

    frontend = start_frontend()
    if frontend:
        time.sleep(5)  # Wait for frontend to start
    else:
        print("   ⚠️  Frontend not started (directory missing)")

    print("✅ Services started")
    print()

    # Check health
    if check_health():
        print("✅ Backend is healthy")
    else:
        print("⚠️  Backend health check failed (may still be starting)")

    print()
    print("="*50)
    print("📊 Demo Status:")
    print("="*50)
    print("  Backend API: http://localhost:8000")
    print("  API Docs:    http://localhost:8000/docs")
    print("  RSS Feed:    http://localhost:8000/rss/bilibili")

    if frontend:
        print("  Frontend:    http://localhost:3000")

    print()
    print("🎥 Demo Features:")
    print("  ✓ Real-time Chinese video platform data")
    print("  ✓ AI-powered content summaries")
    print("  ✓ RSS feed generation")
    print("  ✓ Legal compliance framework")
    print()
    print("🎉 DEMO IS READY!")

    if frontend:
        print("   Opening http://localhost:3000 in browser...")
        try:
            time.sleep(2)
            open_browser()
        except:
            print("   Please manually open: http://localhost:3000")
    else:
        print("   Please open: http://localhost:8000/docs")

    print()
    print("Press Ctrl+C to stop the demo")
    print()

    # Keep running
    try:
        while True:
            time.sleep(1)
            # Check if processes are still running
            if backend and backend.poll() is not None:
                print("⚠️  Backend stopped unexpectedly")
                break
            if frontend and frontend.poll() is not None:
                print("⚠️  Frontend stopped unexpectedly")
                # Frontend stopping is less critical
    except KeyboardInterrupt:
        print("\n📛 Stopping demo...")

    cleanup()

if __name__ == "__main__":
    main()