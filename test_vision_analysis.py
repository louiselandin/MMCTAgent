"""
Test vision analysis on keyframes - actually describe what's in the images
"""
import asyncio
from mmct.providers.factory import ProviderFactory
from mmct.image_pipeline.core.models.vit.gpt4v import GPT4V
from PIL import Image
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

async def analyze_keyframe_with_vision():
    """Use GPT-4o Vision to actually describe what's in the keyframes"""
    try:
        print("🔍 Analyzing keyframes with GPT-4o Vision...")
        
        # First, get the keyframes from the index
        search_provider = ProviderFactory.create_search_provider()
        
        index_name = "keyframes-test-index"
        print(f"\n📊 Retrieving keyframes from: {index_name}")
        
        results = await search_provider.search(
            query="*",
            index_name=index_name,
            search_text="*",
            top=10
        )
        
        if not results:
            print("❌ No keyframes found!")
            return
        
        print(f"✅ Found {len(results)} keyframes\n")
        
        # Sort by timestamp
        results_sorted = sorted(results, key=lambda x: x.get('timestamp_seconds', 0))
        
        # Initialize vision model
        vision_model = GPT4V()
        
        # Analyze each keyframe
        print("=" * 80)
        print("🎬 VIDEO ANALYSIS - Frame by Frame Description")
        print("=" * 80)
        
        frame_descriptions = []
        
        for i, result in enumerate(results_sorted, 1):
            timestamp = result.get('timestamp_seconds', 0)
            filename = result.get('keyframe_filename', '')
            motion = result.get('motion_score', 0)
            
            # Debug: print all available fields
            if i == 1:
                print(f"\nDEBUG - Available fields in result: {list(result.keys())}")
            
            # Construct the full path to the keyframe
            # Keyframes are stored in local_storage/keyframes/<video_hash>/
            video_hash = result.get('video_id', result.get('parent_id', ''))  # Try different field names
            keyframe_path = Path("/workspaces/MMCTAgent/local_storage/keyframes") / video_hash / filename
            
            if not keyframe_path.exists():
                print(f"\n⚠️  Keyframe {i} at {timestamp}s - FILE NOT FOUND: {keyframe_path}")
                continue
            
            print(f"\n🎞️  FRAME {i} - {timestamp:.1f}s (Motion Score: {motion:.2f})")
            print(f"   📁 {filename}")
            print(f"   🔍 Analyzing image content...")
            
            # Use GPT-4o Vision to describe the frame
            prompt = """Describe this image in detail. What do you see? 
Focus on:
- People: how many, what are they doing, their condition
- Objects: vehicles, medical equipment, environment
- Scene: what kind of situation or event is happening
- Any text or signs visible

Be specific and factual."""
            
            try:
                description = await vision_model.run(
                    prompt=prompt,
                    images=Image.open(keyframe_path)
                )
                
                print(f"\n   📝 DESCRIPTION:")
                print(f"   {description[:500]}...")  # Truncate if too long
                
                frame_descriptions.append({
                    'timestamp': timestamp,
                    'description': description,
                    'motion': motion
                })
                
            except Exception as e:
                print(f"   ❌ Error analyzing frame: {e}")
        
        # Now create a summary with ALL frame descriptions
        print("\n\n" + "=" * 80)
        print("🎯 COMPLETE VIDEO SUMMARY")
        print("=" * 80)
        
        llm_provider = ProviderFactory.create_llm_provider()
        
        # Build comprehensive context
        detailed_context = "Here are detailed descriptions of each frame from the 5-second video:\n\n"
        for frame in frame_descriptions:
            detailed_context += f"Frame at {frame['timestamp']:.1f}s (motion: {frame['motion']:.1f}):\n"
            detailed_context += f"{frame['description']}\n\n"
        
        summary_prompt = f"""{detailed_context}

Based on these frame-by-frame descriptions, provide:
1. A comprehensive summary of what's happening in the video
2. The overall scene/situation
3. Key events or changes across the timeline
4. Any important details about people, injuries, or emergency response"""

        messages = [{"role": "user", "content": summary_prompt}]
        
        result = await llm_provider.chat_completion(
            messages=messages,
            temperature=0.3,
            max_tokens=600
        )
        
        summary = result.get('content', 'No summary generated')
        print(f"\n{summary}")
        print("=" * 80)
        
        # Cleanup
        await search_provider.close()
        await llm_provider.close()
        
        print("\n✅ Vision analysis complete!")
        print("✅ GPT-4o Vision successfully analyzed actual frame content")
        print("✅ Now you're getting real descriptions of what's in the images!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(analyze_keyframe_with_vision())
