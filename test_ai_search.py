from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
import os
from dotenv import load_dotenv

# Load the .env file explicitly
load_dotenv()


try:
    search_client = SearchClient(
        endpoint="https://search-lab-louise.search.windows.net",
        index_name="video-test-index", 
        credential=AzureKeyCredential(os.getenv("SEARCH_API_KEY"))
    )
    print("✅ Search client connected successfully!")
except Exception as e:
    print(f"❌ Search connection failed: {e}")

# Check what's being read
api_key = os.getenv("SEARCH_API_KEY")
print(f"API Key type: {type(api_key)}")
print(f"API Key value: {repr(api_key)}")  # This will show if it's None or empty
print(f"API Key length: {len(api_key) if api_key else 'None'}")

# Check if .env is loaded properly
print(f"Search endpoint: {os.getenv('SEARCH_ENDPOINT')}")
print(f"Search provider: {os.getenv('SEARCH_PROVIDER')}")


try:
    api_key = os.getenv("SEARCH_API_KEY")
    
    if not api_key:
        raise ValueError("SEARCH_API_KEY not found in environment")
    
    if not isinstance(api_key, str):
        raise ValueError(f"API key is not a string, it's {type(api_key)}")
    
    search_client = SearchClient(
        endpoint="https://search-lab-louise.search.windows.net",
        index_name="video-test-index", 
        credential=AzureKeyCredential(api_key.strip())  # Strip any whitespace
    )
    print("✅ Search client connected successfully!")
    
except Exception as e:
    print(f"❌ Search connection failed: {e}")