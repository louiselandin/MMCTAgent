"""
Enhanced Video Analysis Application with GPT-4o Vision Descriptions

This version adds GPT-4o Vision descriptions to each keyframe during ingestion,
so VideoAgent can provide accurate descriptions of what's actually happening in the video.
"""

import asyncio
import os
from mmct.video_pipeline import IngestionPipeline, VideoAgent, Languages
from dotenv import load_dotenv

load_dotenv()

async def analyze_video_with_vision():
    """
    Complete video analysis pipeline with vision descriptions:
    1. Ingest video and extract keyframes
    2. Generate GPT-4o Vision descriptions for each keyframe
    3. Store everything in search index
    4. Query the video using VideoAgent
    """
    try:
        video_path = "ambulance_5_sek.mp4"
        index_name = "test-index-vision"  # Use new index to avoid conflicts
        
        print("=" * 80)
        print("🎬 ENHANCED VIDEO ANALYSIS WITH GPT-4o VISION")
        print("=" * 80)
        print(f"📹 Video: {video_path}")
        print(f"📊 Index: {index_name}")
        print(f"🔍 Vision: Enabled (GPT-4o will describe each frame)")
        print("=" * 80)
        
        # Step 1: Ingest video with vision descriptions
        print("\n🔄 Step 1: Video Ingestion with Vision Descriptions")
        print("-" * 80)
        
        ingestion = IngestionPipeline(
            video_path=video_path,
            index_name=index_name,
            transcription_service=None,  # No audio transcription
            language=Languages.ENGLISH_INDIA,
        )
        
        print("⏳ Processing video...")
        print("   - Extracting keyframes")
        print("   - Generating CLIP embeddings")
        print("   - Describing frames with GPT-4o Vision")
        print("   - Storing in search index")
        print()
        
        await ingestion.run()
        
        print("\n✅ Ingestion completed!")
        print("=" * 80)
        
        # Step 2: Query the video
        print("\n🤖 Step 2: Querying Video with VideoAgent")
        print("-" * 80)
        
        query = "Describe what is happening in this video. What do you see? Who is involved?"
        
        print(f"❓ Query: {query}\n")
        print("⏳ Analyzing...")
        
        video_agent = VideoAgent(
            query=query,
            index_name=index_name,
            use_critic_agent=True,  # Enable critic for better answers
            stream=False,
            cache=False
        )
        
        response = await video_agent()
        
        # Display results
        print("\n" + "=" * 80)
        print("🎯 VIDEO ANALYSIS RESULTS")
        print("=" * 80)
        print(response)
        print("=" * 80)
        
        print("\n✅ Analysis complete!")
        print("\n💡 The response above should now describe:")
        print("   - People lying on the ground (injured)")
        print("   - Emergency responders in high-visibility clothing")
        print("   - Medical equipment (stretchers)")
        print("   - Warning signs")
        print("   - The accident/emergency scene")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 Troubleshooting:")
        print("   - Check your .env file has all required API keys")
        print("   - Ensure video file exists: ambulance_5_sek.mp4")
        print("   - Verify Azure OpenAI has GPT-4o with vision enabled")

async def quick_query_only():
    """
    Skip ingestion and just query an existing index.
    Use this if you've already run the full analysis once.
    """
    query = "What injuries can you see? Describe the people and their condition."
    index_name = "test-index-vision"
    
    print(f"\n🔍 Querying existing index: {index_name}")
    print(f"❓ Query: {query}\n")
    
    video_agent = VideoAgent(
        query=query,
        index_name=index_name,
        use_critic_agent=True,
        stream=False
    )
    
    response = await video_agent()
    
    print("\n" + "=" * 80)
    print("🎯 RESULTS")
    print("=" * 80)
    print(response)
    print("=" * 80)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--query-only":
        # Quick query mode
        asyncio.run(quick_query_only())
    else:
        # Full analysis with ingestion
        asyncio.run(analyze_video_with_vision())
