import asyncio

class VectorStore:
    def __init__(self):
        print("Connecting to Qdrant (MOCK)...")
        # In real implementation:
        # self.client = QdrantClient(host="localhost", port=6333)
        pass

    async def store_embedding(self, video_id: str, text: str):
        print(f"Generating embedding for video {video_id}...")
        await asyncio.sleep(0.5) # Simulate embedding generation
        
        print(f"Storing embedding in Qdrant for {video_id}...")
        # In real implementation:
        # embedding = model.encode(text)
        # self.client.upsert(...)
        return True
