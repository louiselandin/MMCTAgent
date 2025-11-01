"""
Semantic Chunking Module

This module provides semantic clustering functionality for video transcripts.
"""

from mmct.video_pipeline.core.ingestion.semantic_chunking.semantic_chunker import SemanticChunker
from mmct.video_pipeline.core.ingestion.semantic_chunking.process_transcript import (
    semantic_chunking,
    add_empty_intervals,
    merge_short_clusters,
    TranscriptSegment,
)

__all__ = [
    "SemanticChunker",
    "semantic_chunking",
    "add_empty_intervals",
    "merge_short_clusters",
    "TranscriptSegment",
]
