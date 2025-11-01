"""
Chapter Creator Module

This module handles batch creation of chapters from semantically chunked transcript segments.
It manages parallel processing, rate limiting, and retry logic for chapter generation.
"""

import asyncio
from typing import List, Dict, Tuple
from loguru import logger
from mmct.video_pipeline.core.ingestion.models import ChapterCreationResponse


class ChapterCreator:
    """
    Handles batch creation of chapters from transcript segments.
    Manages concurrency, rate limiting, and error handling.
    """

    def __init__(self, chapter_generator, max_concurrent_requests: int = 3):
        """
        Initialize ChapterCreator.

        Args:
            chapter_generator: Instance of ChapterGenerator class
            max_concurrent_requests: Maximum number of concurrent API requests
        """
        self.chapter_generator = chapter_generator
        self.max_concurrent_requests = max_concurrent_requests

    async def create_chapters_from_segments(
        self,
        chunked_segments: List,
        video_id: str,
        subject_variety: Dict[str, str],
        categories: str = ""
    ) -> Tuple[List[ChapterCreationResponse], List[str]]:
        """
        Create chapters from chunked transcript segments with parallel processing.

        Args:
            chunked_segments: List of TranscriptSegment objects from semantic chunking
            video_id: Unique video identifier
            subject_variety: Dict with 'subject' and 'variety_of_subject' keys
            categories: Category and subcategory information (optional)

        Returns:
            Tuple of (chapter_responses, chapter_transcripts)
        """
        if not chunked_segments:
            logger.warning("No chunked segments available for chapter creation")
            return [], []

        # Create semaphore to limit concurrent Azure OpenAI requests
        semaphore = asyncio.Semaphore(self.max_concurrent_requests)

        async def create_single_chapter(idx: int, segment) -> Tuple:
            """
            Create a single chapter with retry logic and rate limiting.

            Args:
                idx: Index of the segment
                segment: TranscriptSegment object

            Returns:
                Tuple of (idx, chapter_response, seg_text) or None on failure
            """
            async with semaphore:
                attempts = 0
                max_attempts = 3
                delay = 1

                # Convert TranscriptSegment to timestamp format
                seg_text = self._format_segment_to_timestamp(segment)

                while attempts < max_attempts:
                    try:
                        # Get ChapterCreationResponse instance
                        chapter_response = await self.chapter_generator.create_chapter(
                            transcript=seg_text,
                            video_id=video_id,
                            categories=categories,
                            subject_variety=subject_variety
                        )

                        logger.info(f"Chapter {idx}: transcript segment: {seg_text}")
                        logger.info(f"Chapter {idx}: raw chapter: {chapter_response}")

                        if chapter_response is not None:
                            return idx, chapter_response, seg_text
                        else:
                            logger.warning(
                                f"Chapter {idx}: No response received, "
                                f"attempting retry {attempts + 1}/{max_attempts}"
                            )
                            attempts += 1
                            if attempts < max_attempts:
                                await asyncio.sleep(delay)
                                delay *= 2
                            continue

                    except Exception as e:
                        # Check if it's a rate limiting error
                        if "rate limit" in str(e).lower() or "too many requests" in str(e).lower():
                            logger.warning(
                                f"Chapter {idx}: Rate limit hit, waiting longer before retry..."
                            )
                            await asyncio.sleep(delay * 2)
                        else:
                            logger.error(f"Chapter {idx}: Error on attempt {attempts + 1}: {e}")

                        attempts += 1
                        if attempts < max_attempts:
                            await asyncio.sleep(delay)
                            delay *= 2
                        else:
                            logger.error(f"Chapter {idx}: Failed after {max_attempts} attempts")
                            raise

                return None

        # Create tasks for all chapters
        logger.info(
            f"Creating {len(chunked_segments)} chapters with "
            f"max {self.max_concurrent_requests} concurrent requests..."
        )
        tasks = [
            create_single_chapter(idx, segment)
            for idx, segment in enumerate(chunked_segments)
        ]

        # Execute all chapter creation tasks with controlled concurrency
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results in order
        successful_chapters = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Chapter creation failed with exception: {result}")
                continue
            elif result is not None:
                successful_chapters.append(result)

        # Sort by chapter index to maintain order
        successful_chapters.sort(key=lambda x: x[0])

        # Extract chapter responses and transcripts
        chapter_responses = []
        chapter_transcripts = []
        for _, chapter_response, seg in successful_chapters:
            chapter_responses.append(chapter_response)
            chapter_transcripts.append(seg)

        logger.info(
            f"Chapter Generation Completed! "
            f"Successfully created {len(chapter_responses)} chapters in parallel."
        )

        return chapter_responses, chapter_transcripts

    @staticmethod
    def _format_segment_to_timestamp(segment) -> str:
        """
        Convert TranscriptSegment to timestamp format string.

        Args:
            segment: TranscriptSegment object with start_time, end_time, and sentence

        Returns:
            Formatted string: "HH:MM:SS,mmm --> HH:MM:SS,mmm text"
        """
        start_time = segment.start_time
        end_time = segment.end_time

        seg_text = (
            f"{int(start_time // 3600):02d}:"
            f"{int((start_time % 3600) // 60):02d}:"
            f"{int(start_time % 60):02d},"
            f"{int((start_time % 1) * 1000):03d} --> "
            f"{int(end_time // 3600):02d}:"
            f"{int((end_time % 3600) // 60):02d}:"
            f"{int(end_time % 60):02d},"
            f"{int((end_time % 1) * 1000):03d} "
            f"{segment.sentence}"
        )

        return seg_text
