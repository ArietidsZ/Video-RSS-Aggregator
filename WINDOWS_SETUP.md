# Windows Setup Guide - Video RSS Aggregator

## 🚀 Quick Start for Fresh Windows Machine

### Prerequisites Installation (One-time setup)

1. **Install Python (3.8 or higher)**
   - Download from: https://www.python.org/downloads/
   - ⚠️ **IMPORTANT**: Check ✅ "Add Python to PATH" during installation
   - Verify: Open Command Prompt and run `python --version`

2. **Install Node.js (14 or higher)**
   - Download from: https://nodejs.org/
   - Choose "LTS" version (recommended)
   - Installer automatically adds to PATH
   - Verify: Open Command Prompt and run `node --version`

3. **Install Git (optional but recommended)**
   - Download from: https://git-scm.com/download/win
   - Default installation options are fine
   - Verify: Open Command Prompt and run `git --version`

### 📦 Download and Setup

#### Option 1: Using Git (Recommended)
```cmd
git clone https://github.com/ArietidsZ/video-rss-aggregator.git
cd video-rss-aggregator
```

#### Option 2: Download ZIP
1. Go to: https://github.com/ArietidsZ/video-rss-aggregator
2. Click green "Code" button → "Download ZIP"
3. Extract to desired location
4. Open Command Prompt in extracted folder

### 🎯 One-Click Demo Launch

Simply double-click or run in Command Prompt:
```cmd
demo.bat
```

This will:
- ✅ Check Python and Node.js installation
- ✅ Install all dependencies automatically
- ✅ Start backend server (port 8000)
- ✅ Start frontend server (port 3000)
- ✅ Open demo in your browser

### 🌐 Access Points

Once running, access:
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **RSS Feed**: http://localhost:8000/rss/bilibili

### 🛑 Stopping the Demo

Two options:
1. Press any key in the demo window
2. Run `stop_demo.bat`

### 🔧 Troubleshooting

#### "Python is not recognized"
- Python not installed or not in PATH
- Reinstall Python and check "Add to PATH"
- Or manually add: `C:\Users\[YourName]\AppData\Local\Programs\Python\Python3X` to PATH

#### "npm is not recognized"
- Node.js not installed or not in PATH
- Reinstall Node.js (it adds to PATH automatically)

#### Port already in use
- The demo script will try to free ports automatically
- If it fails, manually close applications using ports 3000 or 8000
- Check with: `netstat -ano | findstr :3000`

#### Dependencies installation fails
Manually install in Command Prompt:
```cmd
# Python dependencies
pip install fastapi uvicorn aiohttp

# Frontend dependencies
cd frontend
npm install
cd ..
```

#### Firewall/Antivirus blocking
- Windows Defender may prompt to allow Python/Node network access
- Click "Allow access" when prompted
- Add exceptions for python.exe and node.exe if needed

### 📝 Manual Start (Advanced)

If demo.bat doesn't work, manually start:

1. **Backend** (Command Prompt 1):
```cmd
set PYTHONPATH=.
python -m uvicorn simple_backend:app --host 0.0.0.0 --port 8000
```

2. **Frontend** (Command Prompt 2):
```cmd
cd frontend
npm start
```

### 💻 System Requirements

- **OS**: Windows 10/11 (64-bit recommended)
- **RAM**: 4GB minimum, 8GB recommended
- **Disk**: 500MB free space
- **Network**: Internet connection for initial setup
- **Browser**: Chrome, Firefox, or Edge (latest versions)

### 🔄 Updating the Project

```cmd
git pull origin main
demo.bat
```

Or download latest ZIP from GitHub and replace files.

### 🎥 Recording Demo on Windows

Best tools for screen recording:
1. **OBS Studio** (Free, Professional)
   - Download: https://obsproject.com/
2. **Windows Game Bar** (Built-in)
   - Press `Win + G` to open
3. **ShareX** (Free, Lightweight)
   - Download: https://getsharex.com/

### 📧 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Look at `backend.log` and `frontend.log` for errors
3. Create an issue at: https://github.com/ArietidsZ/video-rss-aggregator/issues

### 🎉 Success Checklist

- [ ] Python installed and in PATH
- [ ] Node.js installed and in PATH
- [ ] Repository downloaded/cloned
- [ ] demo.bat runs without errors
- [ ] Can access http://localhost:3000
- [ ] Videos load in the interface
- [ ] RSS feed generates at http://localhost:8000/rss/bilibili

You're ready for the competition demo! 🚀