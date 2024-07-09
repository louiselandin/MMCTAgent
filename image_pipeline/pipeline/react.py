from llama_index.agent import ReActAgent
from llama_index.agent.react.output_parser import ReActOutputParser
from llama_index.llms import OpenAILike, AzureOpenAI, OpenAI
import os

class React:
    def __init__(self, tools=None, credentials=None, model_name="gpt-35-turbo", engine_name=None, verbose=True, except_thought=False):
        if tools is None:
            raise Exception("Tools cannot be empty")
        llm_type = os.environ.get("OPENAI_API_TYPE", "like")
        self.llms = []
        if credentials is not None:
            for [api_key, api_base, api_type, api_version], model_n, engine_n in zip(credentials, model_name, engine_name):
                if api_type == "azure":
                    use_azure_ad = False
                    if api_key is None:
                        use_azure_ad = True
                    llm = AzureOpenAI(model=model_n, engine= engine_n or "gpt-35-turbo" ,max_retries=100, verbose=True, temperature=0, azure_endpoint=api_base, api_key=api_key, use_azure_ad=use_azure_ad, api_version=api_version, api_type=api_type)
                elif api_type == "openai":
                    llm = OpenAI(model=model_n, max_retries=100, verbose=True, temperature=0)
                else:
                    llm = OpenAILike(model=model_n, context_window=4096,is_chat_model=True,is_function_calling_model=True, verbose=True, temperature=0)
                self.llms.append(llm)
        else:
            if llm_type== "azure":
                use_azure_ad = False
                if os.getenv("OPENAI_API_KEY") is None:
                    use_azure_ad = True
                llm = AzureOpenAI(model=model_name[0], engine= engine_name[0] or "gpt-35-turbo" ,use_azure_ad=use_azure_ad, max_retries=100, verbose=True, temperature=0)
            elif llm_type == "openai":
                llm = OpenAI(model=model_name[0], max_retries=100, verbose=True, temperature=0)
            else:
                llm = OpenAILike(model=model_name[0], context_window=4096,is_chat_model=True,is_function_calling_model=True, verbose=True, temperature=0)
            self.llms.append(llm)
        self.except_thought = except_thought
        self.react_agents = []
        for i in range(len(self.llms)):
            self.react_agents.append(ReActAgent.from_tools(tools, llm=self.llms[i], verbose=verbose, except_thought=self.except_thought, max_iterations=10))
        # self.react_agent = ReActAgent.from_tools(tools, llm=llm, verbose=verbose, except_thought=self.except_thought, max_iterations=10)
        # self.react_agent = self.react_agents[0]
        
    def chat(self, text, idx=0):
        return self.react_agents[idx].chat(text)

    async def achat(self, text, idx=0):
        response = await self.react_agents[idx].achat(text)
        return response
    
    def fetch_last_step(self, idx=0):
        self.react_agents[idx].fetch_last_step()
    
    def remove_last_step(self, idx=0):
        self.react_agents[idx].remove_last_step()

    def update_last_step(self, text, idx=0):
        self.react_agents[idx].update_last_step(text)
    
    def reset(self, idx=0):
        self.react_agents[idx].reset()

if __name__ == "__main__":
    import os
    import openai
    from tools import BLIPTool
    from llama_index.tools import FunctionTool
    from PIL import Image
    import requests
    import time
    openai.api_key = os.environ["OPENAI_API_KEY"]
    openai.api_base = os.environ["OPENAI_API_BASE"]
    models = openai.Model.list()
    model = models["data"][0]["id"]
    blip = BLIPTool()
    import math
    def sqrt(x:int , y:int ):
        return math.sqrt(x*y)
        
    add = FunctionTool.from_defaults(sqrt, "sqrt", "It computes sqrt of two number x y multipled together")
    imgs_url = ["https://i.pinimg.com/originals/09/f2/21/09f2219f0a4458931939d95b1d9ba1fa.jpg", "https://s-media-cache-ak0.pinimg.com/originals/93/9b/37/939b37b923454a4b2302a9069b0b4dae.jpg"]
    images = []
    for url in imgs_url:
        image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        images.append(image)
    blip.set_images(images)

    r = React([blip, add], model, True)

    response = r.chat("""Sure, I can help you with that! Here's a step-by-step plan to solve the vision language task:

    First, we need to query the image(s) to get information about the brand label on the bottles. We can do this by asking an expert system, such as Vit, to identify the text on the label.
    Query: What text can you read on the label of the bottles?

    Expected response: The expert system will respond with the text on the label, such as "Brand A" or "Brand B".

    Next, we need to query the image(s) to get information about the shape and size of the bottles. We can do this by asking the expert system to identify the object detection features of the bottles.
    Query: What are the shapes and sizes of the bottles?

    Expected response: The expert system will respond with information about the shapes and sizes of the bottles, such as "The bottles are cylindrical with a round base and have a height of approximately 20 cm."

    Now, we need to compare the brand label and shape/size information of each bottle to determine if they are all from the same brand. We can do this by using a logical algorithm to compare the information gathered in steps 1 and 2.
    Query: Are all the bottles from the same brand?

    Expected response: The expert system will respond with a yes or no answer, depending on whether all the bottles have the same brand label and shape/size.

    If the answer is yes, then we can conclude that all the bottles are from the same brand. If the answer is no, then we need to further investigate to determine the reason for the difference in brand labels or shapes/sizes.
    Query: What could be the reason for the difference in brand labels or shapes/sizes?

    Expected response: The expert system will respond with possible reasons, such as "The bottles may be from different production batches" or "The bottles may be counterfeit".

    Finally, we need to generate a natural language response to answer the original question. We can do this by combining the information gathered in steps 1-4

    follow each step one by one and utilize the vit function and formulate the query that is given in individual steps. Always use tools and never skip using tools for any query.

    Execute First step and then wait for observation""")
    print(response)    
    response = r.chat("now execute the second step and wait for observation")
    response = r.chat("now execute the third steps and answer the original question")
    print(response)
    