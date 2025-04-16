"""
This is a get_summary_n_transcript tool which provide the summary with the transcript of video.
"""
# Importing Libraries
import os
from typing_extensions import Annotated

async def get_summary_n_transcript(video_id:Annotated[str,'video id'])->str:
    base_dir = os.getenv('BLOB_DOWNLOAD_DIR')
    summary_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),"utilities",base_dir, f"{video_id}.json")
    with open(summary_path, 'r', encoding="utf-8") as file:
        return file.read()