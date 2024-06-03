from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
from PIL import Image
import requests
import openai
import os
import io
import base64

class GPT4V:
    def __init__(self, api_key=None, api_base=None, api_type=None, api_version=None, model=None, device = None):
        self.api_key = api_key or os.getenv("GPT4V_OPENAI_API_KEY")
        self.api_base = api_base or os.getenv("GPT4V_OPENAI_API_BASE")
        self.api_type = api_type or os.getenv("GPT4V_OPENAI_API_TYPE")
        self.api_version = api_version or os.getenv("GPT4V_OPENAI_API_VERSION")
        self.model = model or os.getenv("GPT4V_OPENAI_API_MODEL")

    def convert_image(image):
        image_bytes = io.BytesIO()
        image.save(image_bytes, format=format)
        base64_image = base64.b64encode(image_bytes.getvalue()).decode("utf-8")
        return base64_image
    
    def get_concat_h_resize(self, im1, im2):
        if im2.height > im1.height:
            ratio = im2.height/im1.height
            im1 = im1.resize((int(ratio*im1.width), int(ratio*im1.height)))
        else:
            ratio = im1.height/im2.height
            im2 = im2.resize((int(ratio*im2.width), int(ratio*im2.height)))
        
        padding = int(0.04*max(im1.width, im2.width))
        new_width = im1.width + padding + im2.width
        dst = Image.new('RGB', (new_width, max(im1.height, im2.height)))
        if im2.height>im1.height:
            pad_h = (im2.height-im1.height)//2
            dst.paste(im1, (0, pad_h))
            dst.paste(im2, (im1.width + padding, 0))
        else:
            pad_h = (im1.height-im2.height)//2
            dst.paste(im1, (0, 0))
            dst.paste(im2, (im1.width + padding, pad_h))
        return dst
    
    def __call__(self, images, prompt, selected_image=None):
        if isinstance(images, list):
            if selected_image.lower() == "left":
                image = images[0]
            elif selected_image.lower() == "right":
                image = images[1]
            else:
                image = self.get_concat_h_resize(*images)
        else:
            image = images
        
        response = openai.OpenAI().chat.completions.create(
          model=self.model,
          messages=[
            {
              "role": "user",
              "content": [
                {
                  "type": "text",
                  "text": f"""{prompt}""",
                },
                {
                  "type": "image_url",
                  "image_url": {
                    "url": f"data:image/jpeg;base64,{self.convert_image(image)}",
                  },
                },
              ],
            }
          ],
          max_tokens=500,
          api_key=self.api_key,
          api_base=self.api_base,
          api_type=self.api_type,
          api_version=self.api_version
        )
        generated_text = response.choices[0].message.content
        return generated_text.strip()

    def get_name(self):
        return """
                Vision Expert: vit\n
               """
    def get_desc(self):
        return """
                You can query information about the given image/images using simple natural language,
                This returns responses in simple language.
                input: 
                    {"query": "What is the number of objects in the image"}
                    or 
                    {"query": "What is the number of objects in the image", "selected_image": "1"}

                    The input can contain two values "query" and "selected_image". "selected_image" is optional but "query" is necessary for all queries.
                    "query" is to define the question that the Vision expert would answer about the image.
                    "selected_image" is used only when there are multiple images given in the problem setting. There are three valid options for "selected_image" i.e., "1", "2", "all". By default all is used, and for scenarios where there is only one image "selected_image" do not change the selection of image.

                response:
                    The output is simple text answering the query given.
               """
    def get_fn_schema(self):
        return """
               query: str
               selected_image: Optional[str] = "all" \n \t possible values: ["1","2",...(any number)...,"all"]
               """
    def __str__(self):
        return f"""
                {self.get_name()}
                {self.get_desc()}
                """