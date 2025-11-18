# test_analysis_only.py
import asyncio
from mmct.video_pipeline import VideoAgent

async def test_analysis_only():
    try:
        print("🤖 Starting video analysis (skipping ingestion)...")
        
        # Use the index from your previous successful ingestion
        video_agent = VideoAgent(
            query="Describe what you see happening in this video visually",
            index_name="test-index",  # This index should already exist from before
            use_critic_agent=False,   # Disable critic to simplify
            stream=False
        )
        
        response = await video_agent()
        print("🎯 Analysis Results:")
        print("=" * 50)
        print(response)
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_analysis_only())