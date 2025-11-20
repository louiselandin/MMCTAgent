"""
Script to recreate the test-index with the correct schema.
This fixes the missing 'hash_video_id' field issue.
"""
import asyncio
from mmct.providers.factory import ProviderFactory
from loguru import logger

async def recreate_index():
    """Delete and recreate the test-index."""
    index_name = "test-index"
    
    # Create search provider
    search_provider = ProviderFactory.create_search_provider()
    
    # Check if index exists
    exists = await search_provider.index_exists(index_name)
    
    if exists:
        logger.info(f"Index '{index_name}' exists. Deleting...")
        await search_provider.delete_index(index_name)
        logger.info(f"Index '{index_name}' deleted successfully")
    else:
        logger.info(f"Index '{index_name}' does not exist")
    
    # Create the index with chapter schema
    logger.info(f"Creating index '{index_name}' with chapter schema...")
    await search_provider.create_index(index_name, "chapter")
    logger.info(f"Index '{index_name}' created successfully with correct schema")
    
    # Verify it was created
    exists = await search_provider.index_exists(index_name)
    if exists:
        logger.success(f"✅ Index '{index_name}' is ready to use!")
    else:
        logger.error(f"❌ Failed to create index '{index_name}'")
    
    # Close the provider
    await search_provider.close()

if __name__ == "__main__":
    asyncio.run(recreate_index())
