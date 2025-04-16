import asyncio
from PIL import Image
import base64
import os
from io import BytesIO
from azure.storage.blob import BlobServiceClient,ContentSettings
from azure.identity import get_bearer_token_provider, DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(),override=True)

async def load_data_from_txt(filepath:str)->list:
    try:
        """
        function to load data from txt file
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = [line.strip() for line in f.readlines()]
        return data
    except Exception as e:
        raise Exception(f"Error loading file: {e}")
    
# Function to decode base64 frames to image
async def decode_base64_to_image(base64_str):
    try:
        """Decode a base64 string to a PIL image."""
        image_data = base64.b64decode(base64_str)
        return Image.open(BytesIO(image_data))
    except Exception as e:
        raise Exception(f"Error decoding the base64 image to image: {e}")
    
async def encode_image_to_base64(image):
    try:
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')
    except Exception as e:
        raise Exception(f"Error encoding image to base64: {e}")
    
async def stack_images_horizontally(frames):
    try:
        images = await asyncio.gather(*[decode_base64_to_image(img) for img in frames])
        total_width = sum(image.width for image in images)
        max_height = max(image.height for image in images)
        new_img = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        for img in images:
            new_img.paste(img, (x_offset, 0))
            x_offset += img.width
        return new_img
    except Exception as e:
        raise Exception(f"Error while stacking the images horizontally: {e}")
    
async def load_required_files(session_id):
    try:
        base_dir = os.getenv('BLOB_DOWNLOAD_DIR')
        os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)),base_dir),exist_ok=True)
        def save_content(container_name,blob_name,list_flag=True):
            blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
            local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),base_dir,blob_name)
            # Download the blob's content
            blob_data = blob_client.download_blob().read()

            if list_flag==True:
                # Decode the byte data to string and split into lines
                content = blob_data.decode("utf-8").splitlines()
                with open(local_path,"w", encoding="utf-8") as f:
                    f.write("\n".join(content))
            else:
                content = blob_data.decode("utf-8")
                with open(local_path,"w", encoding="utf-8") as f:
                    f.write(content)
            

        transcript_blob_name = f"transcript_{session_id}.srt"
        timestamps_blob_name = f"timestamps_{session_id}.txt"
        frames_blob_name = f"frames_{session_id}.txt"
        summary_blob_name = f"{session_id}.json"
        
        blob_service_client = BlobServiceClient(os.getenv('BLOB_ACCOUNT_URL'),DefaultAzureCredential())
        save_content(container_name=os.getenv("TRANSCRIPT_CONTAINER_NAME"),blob_name=transcript_blob_name,list_flag=False)
        save_content(container_name=os.getenv("TIMESTAMPS_CONTAINER_NAME"),blob_name=timestamps_blob_name,list_flag=True)
        save_content(container_name=os.getenv("FRAMES_CONTAINER_NAME"),blob_name=frames_blob_name,list_flag=True)
        save_content(container_name=os.getenv("SUMMARY_CONTAINER_NAME"),blob_name=summary_blob_name,list_flag=False)
    except Exception as e:
        raise Exception(e)
    
