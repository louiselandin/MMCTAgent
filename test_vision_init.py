"""
Direct test of vision description generation
"""
import asyncio
from pathlib import Path
from mmct.video_pipeline.core.ingestion.key_frames_extractor.keyframe_processor import KeyframeProcessor
from mmct.video_pipeline.core.ingestion.key_frames_extractor.keyframe_extractor import KeyframeExtractionConfig, FrameMetadata
from dotenv import load_dotenv

load_dotenv()

async def test_vision_init():
    """Test if vision descriptions can be enabled"""
    
    print("Testing KeyframeProcessor initialization...")
    
    config = KeyframeExtractionConfig(
        motion_threshold=1.5,
        sample_fps=2,
        index_name="test",
        search_endpoint="test"
    )
    
    processor = KeyframeProcessor(
        keyframe_config=config,
        enable_vision_descriptions=True
    )
    
    print(f"✅ Processor created")
    print(f"   Vision enabled: {processor.enable_vision_descriptions}")
    print(f"   Vision model: {processor.vision_model}")
    
    # Create fake keyframe metadata
    fake_metadata = [
        FrameMetadata(frame_number=0, timestamp_seconds=0.0, motion_score=0.0),
        FrameMetadata(frame_number=45, timestamp_seconds=1.5, motion_score=11.8),
    ]
    
    video_hash = "e604e1b921682ee7e764ed9227ff1a4d24303eb2d3b86ecce960bde04421f4ba"
    
    print(f"\n🔍 Testing vision description generation...")
    descriptions = await processor._generate_vision_descriptions(fake_metadata, video_hash)
    
    print(f"\n📊 Results:")
    print(f"   Descriptions generated: {len(descriptions)}")
    for frame_num, desc in descriptions.items():
        print(f"   Frame {frame_num}: {desc[:100]}...")

if __name__ == "__main__":
    asyncio.run(test_vision_init())
