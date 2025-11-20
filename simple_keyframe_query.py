"""
Simple test - directly query keyframes without VideoAgent complexity
"""
import asyncio
from mmct.providers.factory import ProviderFactory
from dotenv import load_dotenv

load_dotenv()

async def test_keyframe_query():
    try:
        print("🔍 Testing keyframe search directly...")
        
        # Create search and LLM providers
        search_provider = ProviderFactory.create_search_provider()
        llm_provider = ProviderFactory.create_llm_provider()
        
        # Search for keyframes in the index we created
        index_name = "keyframes-test-index"
        query = "ambulance emergency vehicle"
        
        print(f"\n📊 Searching index: {index_name}")
        print(f"🔎 Query: {query}")
        
        # Search using vector similarity
        results = await search_provider.search(
            query=query,
            index_name=index_name,
            top=5
        )
        
        print(f"\n✅ Found {len(results)} keyframes!")
        
        # Display results
        for i, result in enumerate(results, 1):
            print(f"\n--- Keyframe {i} ---")
            print(f"  Video ID: {result.get('video_id', 'N/A')}")
            print(f"  Timestamp: {result.get('timestamp_seconds', 'N/A')}s")
            print(f"  Frame Number: {result.get('frame_number', 'N/A')}")
            print(f"  Score: {result.get('@search.score', 'N/A')}")
            if 'frame_path' in result:
                print(f"  Path: {result['frame_path']}")
        
        # Now use LLM to analyze the top keyframe
        if results:
            print("\n🤖 Asking LLM to analyze the keyframes...")
            
            # Create a simple prompt
            frames_info = "\n".join([
                f"Frame {i+1} at {r.get('timestamp_seconds', 'N/A')}s"
                for i, r in enumerate(results[:3])
            ])
            
            prompt = f"""Based on these keyframes from a video:
{frames_info}

Query: What do you see in this video?

Please provide a brief description based on the keyframes found."""

            messages = [{"role": "user", "content": prompt}]
            
            response = await llm_provider.generate(
                messages=messages,
                temperature=0.3,
                max_tokens=200
            )
            
            print("\n🎯 LLM Analysis:")
            print("=" * 50)
            print(response)
            print("=" * 50)
        
        # Cleanup
        await search_provider.close()
        await llm_provider.close()
        
        print("\n✅ Test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_keyframe_query())
