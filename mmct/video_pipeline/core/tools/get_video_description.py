"""
This is a get_video_description tool which provide the summary with the transcript of video.
"""
# Importing Libraries
import os
from typing_extensions import Annotated
from mmct.video_pipeline.utils.helper import get_media_folder
from loguru import logger

async def get_video_description(video_id:Annotated[str,'video id'])->str:
    """
    `get_video_description` tool which retrieve the high-level description of the video, including summary, transcript, and action_taken.

    Args:
    video_id (str): Unique identifier for the video.
    """
    logger.info("Utilizing the get video description tool")
    base_dir = await get_media_folder()
    summary_path = os.path.join(base_dir, f"{video_id}.json")
    if not os.path.exists(summary_path):
        logger.error(f"File path do not exists: {summary_path}")
    with open(summary_path, 'r', encoding="utf-8") as file:
        content = file.read()
        logger.info(f"Video description for {video_id} retrieved successfully")
        return content