"""
KeyframeProcessor: High-level orchestrator for keyframe processing pipeline.

Combines keyframe extraction, embedding generation, vision description, and search index storage
into a single, clean interface.
"""

import os
from pathlib import Path
from PIL import Image
from loguru import logger

from mmct.video_pipeline.core.ingestion.key_frames_extractor.keyframe_extractor import (
    KeyframeExtractor,
    KeyframeExtractionConfig,
)
from mmct.video_pipeline.core.ingestion.key_frames_extractor.clip_embeddings import (
    CLIPEmbeddingsGenerator,
)
from mmct.video_pipeline.core.ingestion.key_frames_extractor.keyframe_search_index import (
    KeyframeSearchIndex,
)
from mmct.image_pipeline.core.models.vit.gpt4v import GPT4V
from mmct.config.settings import ImageEmbeddingConfig


class KeyframeProcessor:
    """
    Orchestrates the complete keyframe processing pipeline:
    1. Extract keyframes from video
    2. Generate CLIP embeddings for keyframes
    3. Store embeddings to search index
    """

    def __init__(
        self,
        keyframe_config: KeyframeExtractionConfig,
        enable_vision_descriptions: bool = True,
    ):
        """
        Initialize the keyframe processor.

        Args:
            keyframe_config: Configuration for keyframe extraction (including index_name and search_endpoint)
            enable_vision_descriptions: Whether to generate GPT-4o Vision descriptions for keyframes
        """
        self.keyframe_config = keyframe_config
        self.keyframe_search_index = None
        self.enable_vision_descriptions = enable_vision_descriptions
        
        if enable_vision_descriptions:
            try:
                self.vision_model = GPT4V()
                logger.info("✅ GPT-4o Vision model initialized for keyframe descriptions")
            except Exception as e:
                logger.error(f"❌ Failed to initialize GPT-4o Vision: {e}")
                self.enable_vision_descriptions = False
                self.vision_model = None
        else:
            self.vision_model = None
            logger.info("Vision descriptions disabled")

    async def _initialize_search_index(self):
        """
        Initialize the keyframe search index client if not already initialized.
        Creates a client connection to the existing index (index creation happens in ingestion_pipeline).
        """
        if self.keyframe_search_index is None:
            if not self.keyframe_config.index_name or not self.keyframe_config.search_endpoint:
                raise ValueError(
                    "index_name and search_endpoint must be provided in keyframe_config"
                )

            # Create keyframe search index client instance
            keyframe_index_name = f"keyframes-{self.keyframe_config.index_name}"
            self.keyframe_search_index = KeyframeSearchIndex(
                search_endpoint=self.keyframe_config.search_endpoint,
                index_name=keyframe_index_name,
            )
            logger.info(f"Keyframe search index client initialized: {keyframe_index_name}")

    async def _generate_vision_descriptions(self, keyframe_metadata: list, video_hash_id: str) -> dict:
        """
        Generate GPT-4o Vision descriptions for keyframes.

        Args:
            keyframe_metadata: List of keyframe metadata
            video_hash_id: Hash ID for the video

        Returns:
            Dict mapping frame_number to vision description
        """
        logger.info(f"🔍 Vision descriptions enabled: {self.enable_vision_descriptions}")
        logger.info(f"🤖 Vision model initialized: {self.vision_model is not None}")
        
        if not self.enable_vision_descriptions or not self.vision_model:
            logger.warning("⚠️  Vision descriptions skipped - not enabled or model not initialized")
            return {}

        vision_descriptions = {}
        # Keyframes are saved to local_storage/keyframes by default
        keyframe_dir = Path("local_storage/keyframes") / video_hash_id

        logger.info(f"🎬 Generating vision descriptions for {len(keyframe_metadata)} keyframes...")
        logger.info(f"📁 Looking for keyframes in: {keyframe_dir}")

        vision_prompt = """Describe this video frame in detail. Focus on:
- People: count, actions, condition, injuries
- Objects: vehicles, medical equipment, signs
- Scene: type of situation, environment
- Any emergency or medical details

Be specific and factual."""

        for frame_meta in keyframe_metadata:
            frame_number = frame_meta.frame_number  # Access as attribute, not dict
            frame_filename = f"{video_hash_id}_{frame_number}.jpg"
            frame_path = keyframe_dir / frame_filename

            if not frame_path.exists():
                logger.warning(f"Keyframe file not found: {frame_path}")
                continue

            try:
                # Use GPT-4o Vision to describe the frame
                description = await self.vision_model.run(
                    prompt=vision_prompt,
                    images=Image.open(frame_path)
                )
                vision_descriptions[frame_number] = description
                logger.info(f"Generated vision description for frame {frame_number}")
            except Exception as e:
                logger.error(f"Failed to generate vision description for frame {frame_number}: {e}")
                vision_descriptions[frame_number] = ""

        return vision_descriptions

    async def process_keyframes(
        self,
        video_path: str,
        video_hash_id: str,
        parent_id: str,
        parent_duration: float,
        video_duration: float,
    ) -> None:
        """
        Process keyframes for a video part: extract, generate embeddings, and store.

        Args:
            video_path: Path to the video file
            video_hash_id: Hash ID for this video part
            parent_id: Hash ID of the parent/original video
            parent_duration: Duration of the parent video in seconds
            video_duration: Duration of this video part in seconds
        """
        try:
            # Initialize search index if not already done
            await self._initialize_search_index()

            # Step 1: Extract keyframes
            logger.info(f"Extracting keyframes for video {video_hash_id}...")
            keyframe_extractor = KeyframeExtractor(self.keyframe_config)
            keyframe_metadata = await keyframe_extractor.extract_keyframes(
                video_path=video_path, video_id=video_hash_id
            )
            logger.info(f"Successfully extracted {len(keyframe_metadata)} keyframes")

            # Step 2: Generate embeddings
            logger.info(f"Generating embeddings for {len(keyframe_metadata)} keyframes...")
            embedding_config = ImageEmbeddingConfig()
            embeddings_generator = CLIPEmbeddingsGenerator(embedding_config)

            try:
                frame_embeddings = await embeddings_generator.process_frames(
                    keyframe_metadata, video_hash_id
                )
                logger.info(f"Successfully generated {len(frame_embeddings)} frame embeddings")
            finally:
                # Clean up embeddings generator resources
                await embeddings_generator.cleanup()

            # Step 2.5: Generate vision descriptions (if enabled)
            vision_descriptions = await self._generate_vision_descriptions(keyframe_metadata, video_hash_id)

            # Step 3: Store embeddings to search index
            logger.info(f"Storing {len(frame_embeddings)} frame embeddings to search index...")
            success = await self.keyframe_search_index.upload_frame_embeddings(
                frame_embeddings=frame_embeddings,
                video_id=video_hash_id,
                video_path=video_path,
                parent_id=parent_id,
                parent_duration=parent_duration,
                video_duration=video_duration,
                vision_descriptions=vision_descriptions,  # Pass vision descriptions
            )

            if success:
                logger.info("Successfully stored frame embeddings to search index")
            else:
                logger.error("Failed to store frame embeddings to search index")

        except Exception as e:
            logger.exception(f"Exception occurred during keyframe processing: {e}")
            raise

    async def close(self):
        """Close the keyframe search index connection."""
        if self.keyframe_search_index:
            await self.keyframe_search_index.close()
            logger.info("Keyframe search index closed successfully")
