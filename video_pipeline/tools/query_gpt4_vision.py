"""
This tool allows to do the query on selected frames around a given timestamp
"""

# Importing Libaries
import os
from typing_extensions import Annotated
import time
import asyncio
from mmct.video_pipeline.utilities.helper import load_data_from_txt
from mmct.llm_client import LLMClient

service_provider = os.getenv("LLM_PROVIDER", "azure")
openai_client = LLMClient(service_provider=service_provider)
openai_client = openai_client.get_client()

async def query_GPT4_Vision(
    query: Annotated[str, 'query'], 
    timestamp: Annotated[str, 'timestamp in format %H:%M:%S'], 
    video_id : Annotated[str,"video id"]
) -> str:
    try:
        base_dir = os.getenv('BLOB_DOWNLOAD_DIR')
        frames_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),"utilities",base_dir, f"frames_{video_id}.txt")
        timestamps_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),"utilities",base_dir, f"timestamps_{video_id}.txt")

        # Convert timestamp to milliseconds
        h, m, s = map(int, timestamp.split(':'))
        timestamp_ms = (h * 3600 + m * 60 + s) * 1000

        frames = await load_data_from_txt(frames_path)
        frame_timestamps = await load_data_from_txt(timestamps_path)
        frame_timestamps = [float(timestamp) for timestamp in frame_timestamps]

        # Find the nearest frame index
        nearest_frame_index = min(
            range(len(frame_timestamps)), 
            key=lambda i: abs(frame_timestamps[i] - timestamp_ms)
        )

        # Select frames
        start_index = max(0, nearest_frame_index - 4)
        end_index = min(nearest_frame_index + 5, len(frames) - 1)
        selected_frames = frames[start_index:end_index + 1]

        # Preparing the payload with prompts and frames.
        content = []
        for i in selected_frames:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{i}",
                    "detail":"high",
                    
                }
            })

        content.append(
            {
                "type":"text",
                "text" :f"{query}"
            }
        )

        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "You are `gpt-4`, the OpenAI model that can describe still frames provided by the user "
                                "from a short video clip in extreme detail. The user has attached frames sampled at 1 fps for you"
                                "to analyze. For every user query, you must carefully examine the frames for the relevant information "
                                "corresponding to the user query and respond accordingly. You are not allowed to hallucinate. Understand the frames carefully, If you have doubt then do not provide answer."
                                "Provide answers only available in the frames."
                                "You are given frames around a timestamp, may be frames are not relevant to the query, you need to be careful."
                            )
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": content
                }
            ],
            "temperature": 0,
            "max_tokens":1000,
            "top_p":0.1
        }

        

        retry_intervals = [10,15]
        for attempt, wait_time in enumerate(retry_intervals,start=1):
            try:
                response = await asyncio.to_thread(openai_client.chat.completions.create,
                        model=os.getenv("GPT4V_DEPLOYMENT"),
                        temperature=payload["temperature"],
                        messages=payload['messages'],
                        top_p=payload['top_p'],
                        max_tokens=payload["max_tokens"]
                    )
                return response.choices[0].message.content
            except Exception as e:
                error_message = str(e)
                if attempt<len(retry_intervals):
                    print(f"Attempt {attempt} failed: {error_message}. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else: 
                    print(f"Final attempt failed: {error_message}")
                    return error_message
    except Exception as e:
        raise Exception(e)
