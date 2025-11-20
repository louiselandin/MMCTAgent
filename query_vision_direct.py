"""
Simple query script that directly uses keyframes with vision descriptions
This bypasses VideoAgent's complex workflow and directly shows what GPT-4o Vision saw
"""

import asyncio
import sys
from mmct.providers.factory import ProviderFactory
from dotenv import load_dotenv

load_dotenv()

async def query_video_directly(custom_query=None):
    """Query video using keyframe vision descriptions directly"""
    
    search_provider = ProviderFactory.create_search_provider()
    llm_provider = ProviderFactory.create_llm_provider()
    
    index_name = "keyframes-test-index-vision"
    
    try:
        print("=" * 80)
        print("🎬 VIDEO ANALYSIS WITH VISION DESCRIPTIONS")
        print("=" * 80)
        print(f"📊 Index: {index_name}")
        print()
        
        # Get all keyframes with vision descriptions
        print("🔍 Retrieving keyframes...")
        results = await search_provider.search(
            query="*",
            index_name=index_name,
            search_text="*",
            top=10
        )
        
        if not results:
            print("❌ No keyframes found!")
            return
        
        # Sort by timestamp
        results_sorted = sorted(results, key=lambda x: x.get('timestamp_seconds', 0))
        
        print(f"✅ Found {len(results_sorted)} keyframes with vision descriptions\n")
        
        # Build context from all vision descriptions
        vision_context = "Here are detailed visual descriptions of keyframes from a 5-second video:\n\n"
        
        for i, result in enumerate(results_sorted, 1):
            timestamp = result.get('timestamp_seconds', 0)
            vision_desc = result.get('vision_description', '')
            
            vision_context += f"**Frame {i} at {timestamp:.1f} seconds:**\n"
            vision_context += f"{vision_desc}\n\n"
        
        # Query the LLM with the vision descriptions
        if custom_query:
            query = custom_query
        else:
            query = "Based on these frame descriptions, describe what is happening in this video. What do you see? Who is involved? What kind of situation is this?"
        
        print(f"❓ Query: {query}\n")
        print("🤖 Analyzing with GPT-4o...\n")
        
        messages = [
            {
                "role": "system",
                "content": "You are a video analyst. You receive detailed descriptions of video keyframes and provide comprehensive analysis of what's happening in the video."
            },
            {
                "role": "user", 
                "content": f"{vision_context}\n\nQuestion: {query}"
            }
        ]
        
        result = await llm_provider.chat_completion(
            messages=messages,
            temperature=0.3,
            max_tokens=800
        )
        
        response = result.get('content', 'No response')
        
        print("=" * 80)
        print("🎯 VIDEO ANALYSIS RESULTS")
        print("=" * 80)
        print(response)
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await search_provider.close()
        await llm_provider.close()

if __name__ == "__main__":
    # Get custom query from command line if provided
    custom_query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None
    
    if custom_query:
        print(f"\n🔍 Custom query: {custom_query}\n")
    else:
        print("\n💡 Tip: You can provide a custom query as argument:")
        print('   python query_vision_direct.py "What injuries can you see?"\n')
    
    asyncio.run(query_video_directly(custom_query))
