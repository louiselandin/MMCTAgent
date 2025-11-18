import asyncio
import os
from mmct.video_pipeline import IngestionPipeline, VideoAgent, Languages
from dotenv import load_dotenv

load_dotenv()

async def test_video_analysis():
    try:
        video_path = "ambulance_5_sek.mp4" 
        
        print("🎬 Starting video ingestion...")
        
        ingestion = IngestionPipeline(
            video_path=video_path,
            index_name="test-index",
            transcription_service=None,  # No audio transcription needed
            language=Languages.ENGLISH_INDIA, 
        )
        
        await ingestion.run()
        print("✅ Video ingestion completed!")
        
        print("🤖 Starting video analysis...")
        
        # Step 2: Analyze the video frames only
        video_agent = VideoAgent(
            query="Describe what you see happening in this video visually",
            index_name="test-index",
            use_critic_agent=True,
            stream=False
        )
        
        response = await video_agent()
        print("🎯 Visual Analysis Results:")
        print("=" * 50)
        print(response)
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure your .env file is configured correctly!")

if __name__ == "__main__":
    asyncio.run(test_video_analysis())