import asyncio
import nats
import sys
import os
import time

# Add generated protos to path
sys.path.append(os.path.join(os.getcwd(), "gen/python"))

from video_rss.events.v1 import events_pb2

async def main():
    nats_url = os.getenv("NATS_URL", "nats://localhost:4222")
    print(f"Connecting to NATS at {nats_url}...")
    
    try:
        nc = await nats.connect(nats_url)
        js = nc.jetstream()
    except Exception as e:
        print(f"Failed to connect: {e}")
        return

    # Subscribe to result
    async def result_handler(msg):
        event = events_pb2.SummarizationCompletedEvent()
        event.ParseFromString(msg.data)
        print(f"\nSUCCESS: Received summary for {event.video_id}")
        print(f"Summary: {event.summary_text}")
        print(f"Key Points: {event.key_points}")
        sys.exit(0)

    print("Subscribing to video.summarized...")
    await js.subscribe("video.summarized", cb=result_handler)
    
    # Publish trigger
    print("Publishing video.discovered...")
    event = events_pb2.VideoDiscoveredEvent()
    event.video_id = "test-video-123"
    # Use a sample audio URL (this is a short wav file)
    event.url = "https://www2.cs.uic.edu/~i101/SoundFiles/BabyElephantWalk60.wav" 
    event.title = "Test Audio"
    
    await js.publish("video.discovered", event.SerializeToString())
    print(f"Event published. Waiting for pipeline (timeout 30s)...")
    
    # Wait
    await asyncio.sleep(30)
    print("\nTimeout waiting for summary.")
    sys.exit(1)

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
