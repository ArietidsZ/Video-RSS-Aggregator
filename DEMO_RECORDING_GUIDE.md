# Video RSS Aggregator - Demo Recording Guide

## 🎬 Competition Demo Recording Guidelines

### Pre-Recording Setup
1. **Clean Environment**
   ```bash
   ./stop_demo.sh  # Stop any existing services
   ./demo.sh       # Start fresh demo
   ```

2. **Browser Preparation**
   - Clear browser cache and cookies
   - Use incognito/private mode for clean demo
   - Have tabs ready: Frontend (localhost:3000), API Docs (localhost:8000/docs)

### Demo Flow (5-7 minutes recommended)

#### 1. Introduction (30 seconds)
- Show project name and purpose
- "Video RSS Aggregator - AI-powered Chinese video platform content aggregation"
- Briefly mention key features:
  - Real-time data extraction from Bilibili, Douyin, Kuaishou
  - AI-generated content summaries
  - RSS feed generation

#### 2. Live Data Extraction Demo (2 minutes)
- Navigate to http://localhost:3000
- Show the live platform selector
- Click "Fetch Videos" for Bilibili
- Demonstrate real-time data extraction
- Point out:
  - View counts, likes, upload dates
  - Chinese titles and descriptions
  - Legal compliance indicators

#### 3. RSS Feed Generation (2 minutes)
- Navigate to http://localhost:8000/rss/bilibili
- Show the generated RSS feed with AI summaries
- Highlight:
  - Content-rich summaries that replace video watching
  - Structured XML format
  - Fair use compliance metadata
- Copy RSS URL and show it works in an RSS reader (optional)

#### 4. API Documentation (1 minute)
- Navigate to http://localhost:8000/docs
- Show interactive API documentation
- Quick demo of API endpoints:
  - `/api/videos/bilibili` - Raw data endpoint
  - `/rss/{platform}` - RSS feed generation
  - Show the "Try it out" functionality

#### 5. Technical Highlights (1 minute)
- Show code structure briefly (VSCode or terminal)
- Highlight key features:
  - Async Python with FastAPI
  - React frontend
  - Legal streaming-based extraction
  - AI content analysis pipeline

#### 6. Closing (30 seconds)
- Summarize unique value proposition
- "Replaces video watching with intelligent summaries"
- Thank viewers

### Recording Tips

1. **Technical Setup**
   - Resolution: 1920x1080 or higher
   - Clear audio (use external mic if possible)
   - Steady screen recording (OBS Studio recommended)
   - Show mouse cursor for clarity

2. **Presentation Style**
   - Speak clearly and at moderate pace
   - Highlight cursor on important elements
   - Pause briefly on key information
   - Keep energy high but professional

3. **What to Emphasize**
   - Real-time data extraction (not cached)
   - AI summaries that provide actual video content
   - Legal compliance framework
   - One-click demo setup simplicity
   - Multi-platform support

4. **What to Avoid**
   - Don't show any error messages
   - Don't refresh too quickly (let data load)
   - Avoid showing empty states
   - Don't show development/debug modes

### Quick Checklist Before Recording

- [ ] Run `./demo.sh` successfully
- [ ] Backend health check passes
- [ ] Frontend loads without errors
- [ ] At least 5 videos show in the feed
- [ ] RSS feed generates with summaries
- [ ] API documentation page loads
- [ ] No credentials visible anywhere
- [ ] Browser console is hidden
- [ ] Screen is clean (close unnecessary apps)

### Sample Script Opening

"Hello! I'm excited to present the Video RSS Aggregator, an AI-powered solution that transforms how we consume content from Chinese video platforms. Our system legally extracts real-time data from Bilibili, Douyin, and Kuaishou, and generates intelligent RSS feeds with AI summaries that can actually replace watching the videos. Let me show you how it works..."

### Troubleshooting During Demo

If something goes wrong:
1. Stay calm and professional
2. Say "Let me refresh this real quick"
3. If persistent, move to next feature
4. Have backup screenshots ready

### Post-Recording
1. Review the recording for:
   - Audio clarity
   - Visible text/UI elements
   - Smooth transitions
   - No sensitive information
2. Trim any dead time
3. Add title/end cards if needed
4. Export in MP4 format (H.264 codec)

## File Naming
`video-rss-aggregator-demo-[date].mp4`

Good luck with your demo! 🎉