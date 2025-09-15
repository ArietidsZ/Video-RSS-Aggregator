#!/usr/bin/env python3
"""
RSS Server Monitor & Auto-Restart Script
Ensures the RTX 5090 RSS backend stays running 24/7
"""

import time
import subprocess
import requests
import logging
import os
import sys
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/rss_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RSSServerMonitor:
    def __init__(self):
        self.server_url = "http://111.186.3.124:9001"
        self.health_endpoint = f"{self.server_url}/stats"
        self.rss_endpoint = f"{self.server_url}/rss/bilibili"
        self.server_process = None
        self.restart_count = 0
        self.max_restarts = 10
        self.check_interval = 30  # seconds
        self.server_script = "C:/Users/zhong/Dropbox/Workspace/Hackathon/video-rss-aggregator/rtx5090_incremental_backend.py"

    def check_server_health(self):
        """Check if server is responding properly"""
        try:
            # Check stats endpoint
            response = requests.get(self.health_endpoint, timeout=10)
            if response.status_code == 200:
                stats = response.json()
                logger.info(f"✅ Server healthy - {stats.get('processed_videos', 0)} videos processed")
                return True
            else:
                logger.warning(f"⚠️ Stats endpoint returned {response.status_code}")
                return False

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Health check failed: {e}")
            return False

    def check_rss_feed(self):
        """Check if RSS feed is working"""
        try:
            response = requests.get(self.rss_endpoint, timeout=15)
            if response.status_code == 200 and "xml" in response.headers.get('content-type', ''):
                logger.info("✅ RSS feed responding correctly")
                return True
            else:
                logger.warning(f"⚠️ RSS feed issue - Status: {response.status_code}")
                return False

        except requests.exceptions.RequestException as e:
            logger.error(f"❌ RSS check failed: {e}")
            return False

    def find_server_process(self):
        """Find running server process"""
        try:
            result = subprocess.run(
                'ssh arietids_ds "netstat -ano | findstr :9001"',
                shell=True, capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if 'LISTENING' in line:
                        parts = line.split()
                        if len(parts) >= 5:
                            pid = parts[-1]
                            logger.info(f"📍 Found server process PID: {pid}")
                            return pid
            return None
        except Exception as e:
            logger.error(f"❌ Error finding process: {e}")
            return None

    def kill_server_process(self):
        """Kill existing server process"""
        pid = self.find_server_process()
        if pid:
            try:
                result = subprocess.run(
                    f'ssh arietids_ds "taskkill /F /PID {pid}"',
                    shell=True, capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    logger.info(f"🔪 Killed server process {pid}")
                    time.sleep(5)  # Wait for process to fully terminate
                    return True
                else:
                    logger.error(f"❌ Failed to kill process {pid}: {result.stderr}")
                    return False
            except Exception as e:
                logger.error(f"❌ Error killing process: {e}")
                return False
        return True

    def start_server(self):
        """Start the RSS server"""
        try:
            logger.info("🚀 Starting RSS server...")

            # Start server in background
            cmd = f'ssh arietids_ds "cd C:/Users/zhong/Dropbox/Workspace/Hackathon/video-rss-aggregator && set PYTHONIOENCODING=utf-8 && python rtx5090_incremental_backend.py"'

            process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            logger.info(f"📡 Server started with process ID: {process.pid}")

            # Wait for server to initialize
            logger.info("⏳ Waiting for server to initialize...")
            time.sleep(60)  # Give time for models to load

            # Check if server is responding
            for attempt in range(10):
                if self.check_server_health():
                    logger.info("✅ Server successfully started and responding!")
                    self.restart_count += 1
                    return True

                logger.info(f"⏳ Waiting for server... attempt {attempt + 1}/10")
                time.sleep(30)

            logger.error("❌ Server failed to start properly")
            return False

        except Exception as e:
            logger.error(f"❌ Error starting server: {e}")
            return False

    def restart_server(self):
        """Restart the server"""
        if self.restart_count >= self.max_restarts:
            logger.critical(f"🚨 Max restarts ({self.max_restarts}) reached. Manual intervention required!")
            return False

        logger.warning(f"🔄 Restarting server (attempt {self.restart_count + 1}/{self.max_restarts})")

        # Kill existing process
        self.kill_server_process()

        # Start new process
        return self.start_server()

    def monitor_loop(self):
        """Main monitoring loop"""
        logger.info("👁️ Starting RSS server monitoring...")
        logger.info(f"📊 Check interval: {self.check_interval} seconds")
        logger.info(f"🎯 Target URL: {self.server_url}")

        consecutive_failures = 0
        max_failures = 3

        while True:
            try:
                # Check server health
                health_ok = self.check_server_health()
                rss_ok = self.check_rss_feed()

                if health_ok and rss_ok:
                    consecutive_failures = 0
                    logger.info(f"✅ All systems operational - Next check in {self.check_interval}s")
                else:
                    consecutive_failures += 1
                    logger.warning(f"⚠️ Server issues detected (failures: {consecutive_failures}/{max_failures})")

                    if consecutive_failures >= max_failures:
                        logger.error("🚨 Server failure threshold reached - Restarting server!")
                        if self.restart_server():
                            consecutive_failures = 0
                        else:
                            logger.critical("❌ Server restart failed!")
                            break

                time.sleep(self.check_interval)

            except KeyboardInterrupt:
                logger.info("🛑 Monitor stopped by user")
                break
            except Exception as e:
                logger.error(f"❌ Monitor error: {e}")
                time.sleep(self.check_interval)

def main():
    monitor = RSSServerMonitor()

    # Initial check
    if not monitor.check_server_health():
        logger.warning("🔧 Server not responding - Starting initial server...")
        if not monitor.start_server():
            logger.critical("❌ Failed to start server initially")
            sys.exit(1)

    # Start monitoring
    monitor.monitor_loop()

if __name__ == "__main__":
    main()