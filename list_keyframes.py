"""
List all documents in keyframes index to verify data was stored
"""
import asyncio
from mmct.providers.factory import ProviderFactory
from dotenv import load_dotenv

load_dotenv()

async def list_keyframes():
    try:
        print("📋 Listing all keyframes in the index...")
        
        # Create search provider
        search_provider = ProviderFactory.create_search_provider()
        
        index_name = "keyframes-test-index"
        
        # Get all documents (search with wildcard)
        results = await search_provider.search(
            query="*",  # Wildcard to get all documents
            index_name=index_name,
            search_text="*",
            top=20
        )
        
        print(f"\n✅ Found {len(results)} keyframes in the index!")
        
        if results:
            print("\n" + "=" * 60)
            for i, result in enumerate(results, 1):
                print(f"\n--- Keyframe {i} ---")
                for key, value in result.items():
                    # Skip large embedding vectors
                    if key == 'embeddings' or key.startswith('@'):
                        continue
                    print(f"  {key}: {value}")
            print("\n" + "=" * 60)
        else:
            print("\n⚠️  No keyframes found in the index!")
            print("This means the ingestion didn't complete successfully.")
        
        # Cleanup
        await search_provider.close()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(list_keyframes())
