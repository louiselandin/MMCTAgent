"""
Quick vision test with hardcoded path
"""
import asyncio
from mmct.image_pipeline.core.models.vit.gpt4v import GPT4V
from PIL import Image
from pathlib import Path

async def test_one_frame():
    """Test vision on just one keyframe"""
    
    video_hash = "e604e1b921682ee7e764ed9227ff1a4d24303eb2d3b86ecce960bde04421f4ba"
    
    # Test frame at 2.0s (highest motion)
    frame_file = f"{video_hash}_60.jpg"
    frame_path = Path(f"/workspaces/MMCTAgent/local_storage/keyframes/{video_hash}/{frame_file}")
    
    print(f"🔍 Analyzing frame: {frame_path}")
    print(f"📁 File exists: {frame_path.exists()}\n")
    
    if not frame_path.exists():
        print("❌ File not found!")
        return
    
    vision_model = GPT4V()
    
    prompt = """Describe this image in detail. What do you see? 
Focus on:
- People: how many, what are they doing, their condition
- Objects: vehicles, medical equipment, environment  
- Scene: what kind of situation or event is happening
- Any injuries or medical situations

Be specific and factual."""
    
    print("🤖 Sending to GPT-4o Vision...\n")
    
    description = await vision_model.run(
        prompt=prompt,
        images=Image.open(frame_path)
    )
    
    print("=" * 80)
    print("📝 FRAME DESCRIPTION:")
    print("=" * 80)
    print(description)
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(test_one_frame())
