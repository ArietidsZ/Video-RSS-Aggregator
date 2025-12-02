import asyncio
import os
import sys
import nats
from nats.errors import ConnectionClosedError, TimeoutError, NoServersError

# Add generated protos to path
sys.path.append(os.path.join(os.getcwd(), "gen/python"))

from video_rss.events.v1 import events_pb2
from summarizer import Summarizer
# from vector_store import VectorStore # Temporarily disabled until implemented

async def main():
    nats_url = os.getenv("NATS_URL", "nats://localhost:4222")
    
    print(f"AI Service connecting to NATS at {nats_url}...")
    try:
        nc = await nats.connect(nats_url)
        js = nc.jetstream()
        print("Connected to NATS JetStream.")
    except Exception as e:
        print(f"Failed to connect to NATS: {e}")
        return

    # Create stream if not exists (idempotent)
    try:
        await js.add_stream(name="VIDEO_EVENTS", subjects=["video.discovered", "video.transcribed", "video.summarized"])
    except Exception as e:
        print(f"Stream creation warning (might exist): {e}")

    summarizer = Summarizer()
    # vector_store = VectorStore()

    async def message_handler(msg):
        try:
            event = events_pb2.TranscriptionCompletedEvent()
            event.ParseFromString(msg.data)
            print(f"Received transcription for video: {event.video_id}")
            
            # Summarize
            print(f"Summarizing text length: {len(event.transcription_text)}")
            summary = await summarizer.summarize(event.transcription_text)
            
            # Publish Summary Event
            s_event = events_pb2.SummarizationCompletedEvent()
            s_event.video_id = event.video_id
            s_event.summary_text = summary["summary"]
            s_event.key_points.extend(summary["key_points"])
            
            await js.publish("video.summarized", s_event.SerializeToString())
            print(f"Published video.summarized for {event.video_id}")

            # Store Embeddings (Future work)
            # await vector_store.store_embedding(event.video_id, summary["summary"])
            
            await msg.ack()
        except Exception as e:
            print(f"Error processing message: {e}")
            # await msg.nak()

    # Subscribe to video.transcribed
    print("Subscribing to video.transcribed...")
    await js.subscribe("video.transcribed", cb=message_handler, durable="ai_summarizer")

    # Keep running
    while True:
        await asyncio.sleep(1)

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
