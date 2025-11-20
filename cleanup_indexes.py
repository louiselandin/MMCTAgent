"""
Clean up old indexes before running the vision-enhanced analysis
"""
import asyncio
from mmct.providers.factory import ProviderFactory
from dotenv import load_dotenv

load_dotenv()

async def cleanup_indexes():
    """Delete old test indexes to start fresh"""
    search_provider = ProviderFactory.create_search_provider()
    
    indexes_to_delete = [
        "keyframes-test-index-vision",
        "test-index-vision"
    ]
    
    for index_name in indexes_to_delete:
        try:
            if await search_provider.index_exists(index_name):
                print(f"🗑️  Deleting index: {index_name}")
                await search_provider.delete_index(index_name)
                print(f"✅ Deleted: {index_name}")
            else:
                print(f"ℹ️  Index doesn't exist: {index_name}")
        except Exception as e:
            print(f"⚠️  Error deleting {index_name}: {e}")
    
    await search_provider.close()
    print("\n✅ Cleanup complete!")

if __name__ == "__main__":
    asyncio.run(cleanup_indexes())
