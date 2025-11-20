"""
Verify that vision descriptions were stored in the keyframes index
"""
import asyncio
from mmct.providers.factory import ProviderFactory
from dotenv import load_dotenv

load_dotenv()

async def check_vision_descriptions():
    """Check if vision descriptions are in the keyframes index"""
    search_provider = ProviderFactory.create_search_provider()
    
    index_name = "keyframes-test-index-vision"
    
    try:
        print(f"🔍 Checking index: {index_name}")
        print()
        
        # Get all keyframes
        results = await search_provider.search(
            query="*",
            index_name=index_name,
            search_text="*",
            top=10
        )
        
        if not results:
            print("❌ No keyframes found in index!")
            print("⚠️  The ingestion may still be running. Wait a few minutes and try again.")
            return
        
        print(f"✅ Found {len(results)} keyframes\n")
        print("=" * 80)
        print("CHECKING VISION DESCRIPTIONS")
        print("=" * 80)
        
        # Sort by timestamp
        results_sorted = sorted(results, key=lambda x: x.get('timestamp_seconds', 0))
        
        descriptions_found = 0
        
        for i, result in enumerate(results_sorted, 1):
            timestamp = result.get('timestamp_seconds', 0)
            vision_desc = result.get('vision_description', '')
            
            print(f"\n🎞️  Frame {i} at {timestamp:.1f}s:")
            
            if vision_desc:
                descriptions_found += 1
                # Show first 200 chars
                preview = vision_desc[:200] + "..." if len(vision_desc) > 200 else vision_desc
                print(f"   ✅ HAS VISION DESCRIPTION:")
                print(f"   {preview}")
            else:
                print(f"   ❌ NO VISION DESCRIPTION")
        
        print("\n" + "=" * 80)
        print(f"SUMMARY: {descriptions_found}/{len(results)} frames have vision descriptions")
        print("=" * 80)
        
        if descriptions_found == len(results):
            print("\n🎉 SUCCESS! All keyframes have vision descriptions!")
            print("✅ The VideoAgent should now be able to describe what's actually in the video!")
        elif descriptions_found > 0:
            print(f"\n⚠️  Only {descriptions_found} out of {len(results)} frames have descriptions")
            print("   The ingestion may have been interrupted. Try running again.")
        else:
            print("\n❌ NO vision descriptions found!")
            print("   The vision description feature may not have been enabled during ingestion.")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if "not found" in str(e).lower():
            print("\n💡 The index doesn't exist yet. The ingestion is probably still running.")
            print("   Wait a few minutes and run this script again.")
    finally:
        await search_provider.close()

if __name__ == "__main__":
    asyncio.run(check_vision_descriptions())
