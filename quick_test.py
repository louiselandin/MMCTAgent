"""
Quick test script - skips transcription for faster testing
"""
import asyncio
from mmct.video_pipeline import IngestionPipeline, VideoAgent, Languages
from dotenv import load_dotenv

load_dotenv()

async def quick_test():
    try:
        video_path = "ambulance_5_sek.mp4" 
        
        print("🎬 Starting quick video ingestion (no transcription)...")
        
        # Run ingestion without transcription for faster testing
        ingestion = IngestionPipeline(
            video_path=video_path,
            index_name="test-index",
            transcription_service=None,  # Skip transcription (use None, not "none")
            language=Languages.ENGLISH_INDIA, 
        )
        
        await ingestion.run()
        print("✅ Video ingestion completed!")
        
        print("\n🤖 Starting video analysis...")
        
        # Analyze the video using only visual information
        video_agent = VideoAgent(
            query="What do you see in this video?",
            index_name="test-index",
            use_critic_agent=False,  # Disable critic for speed
            stream=False
        )
        
        response = await video_agent()
        print("\n🎯 Analysis Results:")
        print("=" * 50)
        print(response)
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(quick_test())
