#!/usr/bin/env python3
"""
RSS Server Status Checker
Quick health check and diagnostics for the RTX 5090 RSS backend
"""

import requests
import json
import subprocess
from datetime import datetime

def check_server_status():
    """Comprehensive server status check"""
    print(f"🔍 RSS Server Status Check - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    server_url = "http://111.186.3.124:9001"

    try:
        # Check stats endpoint
        print("📊 Checking server stats...")
        response = requests.get(f"{server_url}/stats", timeout=10)

        if response.status_code == 200:
            stats = response.json()
            print(f"✅ Server is ONLINE")
            print(f"   📹 Processed Videos: {stats.get('processed_videos', 0)}")
            print(f"   🎮 GPU: {stats.get('gpu_name', 'Unknown')}")
            print(f"   💾 GPU Memory Allocated: {stats.get('gpu_memory_allocated', 'Unknown')}")
            print(f"   💾 GPU Memory Reserved: {stats.get('gpu_memory_reserved', 'Unknown')}")
            print(f"   ⚡ Processing Rate: {stats.get('processing_rate', 'Unknown')}")

            # Check optimizations
            optimizations = stats.get('optimizations', [])
            if optimizations:
                print(f"   🚀 Optimizations: {', '.join(optimizations)}")

        else:
            print(f"❌ Stats endpoint returned: {response.status_code}")

    except Exception as e:
        print(f"❌ Server appears to be OFFLINE: {e}")

    print("\n" + "-" * 60)

    try:
        # Check RSS feed
        print("📡 Checking RSS feed...")
        response = requests.get(f"{server_url}/rss/bilibili", timeout=15)

        if response.status_code == 200:
            print(f"✅ RSS feed is WORKING")

            # Parse RSS content
            content = response.text
            if "AI-Powered Video Feed" in content:
                print("   📰 Feed title: AI-Powered Video Feed")

                # Count items
                item_count = content.count("<item>")
                print(f"   📋 Available items: {item_count}")

                # Check for recent updates
                if "lastBuildDate" in content:
                    print("   🕐 Feed has recent updates")

            else:
                print("   ⚠️ RSS format may be incorrect")

        else:
            print(f"❌ RSS feed returned: {response.status_code}")

    except Exception as e:
        print(f"❌ RSS feed check failed: {e}")

    print("\n" + "-" * 60)

    # Check process status
    print("🔍 Checking process status...")
    try:
        result = subprocess.run([
            "ssh", "arietids_ds",
            "netstat -ano | findstr :9001"
        ], capture_output=True, text=True, timeout=10)

        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if 'LISTENING' in line:
                    parts = line.split()
                    if len(parts) >= 5:
                        pid = parts[-1]
                        print(f"✅ Server process found - PID: {pid}")
                        print(f"   🌐 Listening on port 9001")
                        break
        else:
            print("❌ No process found listening on port 9001")

    except Exception as e:
        print(f"❌ Process check failed: {e}")

    print("\n" + "=" * 60)
    print("🚀 Status check complete!")

if __name__ == "__main__":
    check_server_status()