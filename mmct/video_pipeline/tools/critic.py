"""
This tool can do the visual criticism on the provided logs containing reasoning and query.
"""

# Importing Libaries
import os
import json
import asyncio
from typing_extensions import Annotated
from mmct.video_pipeline.prompts_and_description import SYSTEM_PROMPT_CRITIC_TOOL
from mmct.llm_client import LLMClient
from mmct.video_pipeline.utilities.helper import load_data_from_txt, encode_image_to_base64, stack_images_horizontally

service_provider = os.getenv("LLM_PROVIDER", "azure")
openai_client = LLMClient(service_provider=service_provider)
openai_client = openai_client.get_client()


async def critic_tool(timestamps_predicted: Annotated[str, "A pipe-separated list of relavant timestamps (timestamp count must be less than 10 always) in the format %H:%M:%S only. Example: '00:00:00|00:01:30'. Invalid entries like '00:00:27,920|00:00:32,680' or '00:00:00|END' are not allowed."]
, logs:Annotated[str, "complete retreival and reasoning logs with tool usage (which tool used) and output"], video_id: Annotated[str,"video id"]):
    try:
        base_dir = os.getenv('BLOB_DOWNLOAD_DIR')
        os.makedirs(base_dir,exist_ok=True)
        frames_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),"utilities",base_dir, f"frames_{video_id}.txt")
        timestamps_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),"utilities",base_dir, f"timestamps_{video_id}.txt")

        frame_timestamps = await load_data_from_txt(timestamps_path)
        frame_timestamps = [float(timestamp) for timestamp in frame_timestamps]

        timestamps = timestamps_predicted.split("|")
        if not timestamps or timestamps[0] == "END":
            return "Invalid timestamps"

        frames = await load_data_from_txt(frames_path)
        total_frames, base64Frames = 10, []
        frames_per_timestamp = total_frames // len(timestamps)
        remainder_frames = total_frames % len(timestamps)
        assignment_str, assignment_idx = "", 0

        for index, timestamp in enumerate(timestamps):
            h, m, s = map(int, timestamp.split(':'))
            timestamp_ms = (h * 3600 + m * 60 + s) * 1000
            nearest_frame_index = min(range(len(frame_timestamps)), key=lambda i: abs(frame_timestamps[i] - timestamp_ms))
            num_frames = frames_per_timestamp + (remainder_frames if index == len(timestamps) - 1 else 0)
            start_index, end_index = max(0, nearest_frame_index - 5), min(nearest_frame_index + 4, len(frames) - 1)
            possible_frames = frames[start_index:end_index + 1]
            
            selected_frames = possible_frames if len(possible_frames) < num_frames else []
            if len(possible_frames) >= num_frames:
                stack_size = len(possible_frames) // num_frames
                remainder = len(possible_frames) % num_frames
                selected_frames = [await encode_image_to_base64(await stack_images_horizontally(possible_frames[i*stack_size:(((i+1)*stack_size) + (remainder if i == num_frames - 1 else 0))])) for i in range(num_frames)]

            base64Frames.extend(selected_frames)
            assigned_frame_numbers = [str(i + 1) for i in range(assignment_idx, assignment_idx + len(selected_frames))]
            assignment_idx += len(selected_frames)
            assignment_str += f"Image(s) {', '.join(assigned_frame_numbers)} are for timestamp {timestamp}; "
        
        assignment_str = assignment_str.rstrip('; ') + ". Note that each image may contain multiple stacked frames."
        

        content = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{i}"}} for i in base64Frames]
        content.append({"type": "text", "text": f"These are the logs: {logs}"})
        content.append({"type": "text", "text": assignment_str})
        
        payload = {
            "messages": [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT_CRITIC_TOOL}]},
                          {"role": "user", "content": content}],
            "temperature": 0, "top_p": 0.1
        }
        
        retry_intervals = [10, 15]
        for attempt, wait_time in enumerate(retry_intervals, start=1):
            try:
                response = await asyncio.to_thread(openai_client.chat.completions.create,
                    model= os.getenv("GPT4V_DEPLOYMENT"), temperature=payload["temperature"],
                    messages=payload['messages'], top_p=payload['top_p']
                )
                break
            except Exception as e:
                if attempt < len(retry_intervals):
                    print(f"Attempt {attempt} failed: {e}. Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    return f"Final attempt failed: {e}"

        response_content = json.loads(response.choices[0].message.content)
        return json.dumps({'Critic Feedback': response_content.get('Feedback', '')}), response_content.get('Verdict', 'YES')
    except Exception as e:
        raise Exception(e)