"""
This is a retreive documents tool which provide the summary with the transcript of video related to the query.
"""

# Importing Libraries
import os
from typing_extensions import Annotated
from mmct.video_pipeline.utils.helper import get_media_folder
from azure.search.documents.models import VectorizedQuery, VectorFilterMode
from mmct.config.settings import MMCTConfig
from mmct.providers.factory import provider_factory

config = MMCTConfig()
# Initialize providers
llm_provider = provider_factory.create_llm_provider(config.llm.provider, config.llm.model_dump())

# Try to create embedding provider, fallback to LLM config if embedding config is incomplete
try:
    embedding_provider = provider_factory.create_embedding_provider(
        config.embedding.provider, config.embedding.model_dump()
    )
except Exception as e:
    # Use LLM config for embedding provider as fallback
    llm_config = config.llm.model_dump()
    # Add embedding-specific deployment name if available
    if hasattr(config.llm, "embedding_deployment_name") and config.llm.embedding_deployment_name:
        llm_config["deployment_name"] = config.llm.embedding_deployment_name
    elif hasattr(config.llm, "deployment_name") and config.llm.deployment_name:
        llm_config["deployment_name"] = config.llm.deployment_name

    embedding_provider = provider_factory.create_embedding_provider(config.llm.provider, llm_config)

# Try to create search provider, but handle missing configuration gracefully
try:
    search_provider = provider_factory.create_search_provider(
        config.search.provider, config.search.model_dump()
    )
except Exception as e:
    search_provider = None


async def get_context(
    query: Annotated[str, "query for which documents needs to fetch"],
    index_name: Annotated[str, "vector index name"],
    video_id: Annotated[str, "video id if provided in the instruction"] = None,
    youtube_url: Annotated[str, "youtube url if provided in the instruction"] = None,
) -> str:
    """
    retreive related documents based on the query from the vector database.
    """

    # embedding the query
    query_embds = await embedding_provider.embedding(query)
    if youtube_url:
        filter_query = f"youtube_url eq '{youtube_url}'"
    elif video_id:
        filter_query = f"hash_video_id eq '{video_id}'"
    else:
        filter_query = None  # no filter

    search_results = await search_provider.search(
        query=query,
        index_name=index_name,
        search_text=None,
        vector_queries=[VectorizedQuery(vector=query_embds, fields="embeddings")],
        vector_filter_mode=VectorFilterMode.PRE_FILTER,
        top=3,
        filter=filter_query,
        select=[
            "detailed_summary",
            "topic_of_video",
            "action_taken",
            "text_from_scene",
            "chapter_transcript",
            "hash_video_id",
            "youtube_url",
        ],
    )

    search_results = [dict(result) for result in search_results]
    return search_results


if __name__ == "__main__":
    import asyncio

    video_id = "6242f2b4e99f13c3dec2f9b4cc2b4f2d1e4970d06f1173443875c8ce62580fc2"
    query = "what are the key elements in the machine learning?"
    index_name = "education-video-index-v2"
    results = asyncio.run(get_context(video_id, query, index_name))
