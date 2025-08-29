from mcp_server.tools.schemas.kb_tool_schemas import SearchRequest, get_filter_string
from mmct.providers.factory import provider_factory
from mmct.config.settings import MMCTConfig
from typing import Annotated, List, Dict
from mcp_server.server import mcp
from loguru import logger

try:
    config = MMCTConfig()
    logger.info("Successfully retrieved the MMCT config")
except Exception as e:
    logger.exception(f"Exception occurred while fetching the MMCT config: {e}")

try:
    logger.info("Instantiating the embedding and search providers")
    search_provider = provider_factory.create_search_provider(
        config.search.provider, config.search.model_dump()
    )
    embed_provider = provider_factory.create_embedding_provider(
        provider_name=config.embedding.provider, config=config.embedding.model_dump()
    )
    logger.info("Successfully instantiated the search and embedding providers")
except Exception as e:
    logger.exception(f"Exception occurred while instantiating providers: {e}")


@mcp.tool(
    description="Search the video-chapter knowledge base with optional modes and filters",
)
async def kb_tool(
    request: Annotated[
        SearchRequest, "Search parameters including query, mode, k, and optional filters"
    ],
) -> List[Dict]:
    """
    Perform search over video chapter knowledge base, using chosen mode and filters,
    returning top-k results with optional semantic answer.
    """
    embedding = None

    if (request.query and request.query in ["*"]) and (
        request.query_type and request.query_type in ["vector", "semantic"]
    ):
        raise Exception("Invalid input segment. For * queries, query type must be `full`")

    if request.query_type in ("vector", "semantic"):
        embedding = await embed_provider.embedding(text=request.query)

    results = await search_provider.search(
        index_name=request.index_name,
        query=request.query,
        query_type=request.query_type,
        top=request.k,
        filter=await get_filter_string(request.filters.model_dump()) if request.filters else None,
        embedding=embedding,
    )

    return results