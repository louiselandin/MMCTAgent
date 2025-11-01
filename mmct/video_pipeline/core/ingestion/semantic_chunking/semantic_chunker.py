"""
Semantic Chunker Module

This module handles semantic clustering of video transcripts.
It processes transcripts and groups semantically similar segments together.
"""

import warnings
import asyncio
from typing import List
from loguru import logger
from mmct.video_pipeline.core.ingestion.semantic_chunking.process_transcript import (
    semantic_chunking,
    add_empty_intervals,
    merge_short_clusters
)


class SemanticChunker:
    """
    Handles semantic clustering of video transcripts.

    This class is responsible ONLY for semantic chunking operations:
    - Parsing transcripts
    - Performing semantic clustering
    - Adding empty intervals for gaps
    - Merging short clusters
    """

    def __init__(self, transcript: str):
        """
        Initialize SemanticChunker.

        Args:
            transcript (str): Raw SRT transcript text to be processed
        """
        self.transcript = transcript
        self.chunked_segments = []

    async def _process_and_chunk_transcript(self, srt_text: str) -> List:
        """
        Process the transcript through semantic chunking pipeline.

        Args:
            srt_text (str): Raw SRT transcript text

        Returns:
            List[TranscriptSegment]: Processed and chunked transcript segments
        """
        # Perform semantic chunking - returns List[TranscriptSegment]
        chunked_segments = await semantic_chunking(srt_text=srt_text)
        logger.info(f"Semantic chunking complete: {len(chunked_segments)} chunks created")

        # Add empty intervals for gaps in the transcript
        chunked_segments = await add_empty_intervals(chunked_segments)
        logger.info(f"After adding empty intervals: {len(chunked_segments)} segments")

        # Merge short duration clusters to improve chapter quality and reduce processing time
        chunked_segments = await merge_short_clusters(chunked_segments, min_duration_seconds=30)
        logger.info(f"Clusters after merging: {len(chunked_segments)}")

        return chunked_segments

    async def process(self) -> List:
        """
        Process the transcript through the complete semantic chunking pipeline.

        Returns:
            List[TranscriptSegment]: Semantically chunked transcript segments, or empty list on error
        """
        logger.info(f"Starting semantic chunking for transcript: {self.transcript[:500]}...")

        # Process through semantic chunking pipeline
        chunked_segments = await self._process_and_chunk_transcript(srt_text=self.transcript)

        # Validate chunks
        if not chunked_segments:
            warnings.warn("Formatted Transcript is Empty.", RuntimeWarning)
            logger.error("No clusters generated from transcript!")
            return []

        # Store the chunked segments
        self.chunked_segments = chunked_segments
        logger.info(f"Semantic chunking completed: {len(chunked_segments)} final segments")

        return self.chunked_segments


if __name__ == "__main__":
    # Example usage
    sample_transcript = """1
00:00:00,000 --> 00:00:05,000
This is a sample transcript.

2
00:00:05,000 --> 00:00:10,000
It contains multiple segments."""

    chunker = SemanticChunker(transcript=sample_transcript)
    asyncio.run(chunker.process())
