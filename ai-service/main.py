import asyncio
import os
import nats
from nats.errors import ConnectionClosedError, TimeoutError, NoServersError
from video_rss.events.v1 import events_pb2
from transcriber import Transcriber
from summarizer import Summarizer
from vector_store import VectorStore

async def main():
    nats_url = os.getenv("NATS_URL", "nats://localhost:4222")
    
    print(f"Connecting to NATS at {nats_url}...")
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

    transcriber = Transcriber()
    summarizer = Summarizer()
    vector_store = VectorStore()

    async def message_handler(msg):
        subject = msg.subject
        data = msg.data
        
        try:
            event = events_pb2.VideoDiscoveredEvent()
            event.ParseFromString(data)
            print(f"Processing video: {event.title} ({event.video_id})")
            
            # 1. Transcribe
            transcription = await transcriber.transcribe(event.url)
            
            # Publish Transcription Event
            t_event = events_pb2.TranscriptionCompletedEvent(
                video_id=event.video_id,
                transcription_text=transcription["text"],
                language=transcription["language"]
            )
            await js.publish("video.transcribed", t_event.SerializeToString())
            print(f"Published video.transcribed for {event.video_id}")

            # 2. Summarize
            summary = await summarizer.summarize(transcription["text"])
            
            # Publish Summary Event
            s_event = events_pb2.SummarizationCompletedEvent(
                video_id=event.video_id,
                summary_text=summary["summary"],
                key_points=summary["key_points"]
            )
            await js.publish("video.summarized", s_event.SerializeToString())
            print(f"Published video.summarized for {event.video_id}")

            # 3. Store Embeddings
            await vector_store.store_embedding(event.video_id, summary["summary"])
            
            await msg.ack()
        except Exception as e:
            print(f"Error processing message: {e}")
            await msg.nak()

    # Subscribe to video.discovered
    print("Subscribing to video.discovered...")
    await js.subscribe("video.discovered", cb=message_handler, durable="ai_processor")

    # Keep running
    while True:
        await asyncio.sleep(1)

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
