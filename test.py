from mmct.video_pipeline import VideoAgent
import nest_asyncio
nest_asyncio.apply()

try:
    from mmct.config.settings import MMCTConfig
    config = MMCTConfig()
    print(f"LLM Provider: {config.llm.provider}")
    print(f"LLM Endpoint: {config.llm.endpoint}")
    print(f"LLM Deployment: {config.llm.deployment_name}")
    print(f"Embedding Provider: {config.embedding.provider}")
    print(f"Embedding Endpoint: {config.embedding.endpoint}")
    print(f"Embedding Deployment: {config.embedding.deployment_name}")
    print("✅ Configuration loaded successfully")
except Exception as e:
    print(f"❌ Configuration failed: {e}")
    import traceback
    traceback.print_exc()

# Create VideoAgent instance
video_agent = VideoAgent(
    query="user-query", #"input-query",
    index_name="relevant-index-name", #"your-index-name",
    video_id=None,  # Optional: specify video ID
    url=None,  # Optional: URL to filter out the documents
    use_critic_agent=True,  # Enable critic agent
    stream=True,  # Stream response
    cache=False  # Optional: enable caching
)

# Run the agent
#response = await video_agent()
print("VideoAgent executed successfully!")

# Display the response
#print(response)