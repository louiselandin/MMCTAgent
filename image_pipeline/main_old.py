import os
import openai
from tool import BLIPTool
from data.nlvr.nlvr2.dataloader import NLVR2Dataset
from PIL import Image
import requests
import time
from llama_index.llms import OpenAILike
from pipeline import React

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


def main():
    openai.api_key = os.environ["OPENAI_API_KEY"]
    openai.api_base = os.environ["OPENAI_API_BASE"]
    if os.environ.get("OPENAI_API_TYPE", None):
        openai.api_type = os.environ["OPENAI_API_TYPE"]
        openai.api_version = os.environ["OPENAI_API_VERSION"]
        model = "gpt-4"
        deployment_name = "gpt-4"
    else:
        models = openai.Model.list()
        model = models["data"][0]["id"]
        deployment_name = None
    
    
    dataset = NLVR2Dataset("~/MMCT/image_pipeline/data/nlvr/nlvr2/data/test2.json", "~/nlvr", "test2")
    blip = BLIPTool()
    count = 0
    for data in dataset:
        images = [data["img1"], data["img2"]]
        blip.set_images(images)
        question = data["query_text"]
        identifier = data["identifier"]
        print("\n--------------------\n")
        print(f"Question: {question} Identifier: {identifier}") 
        prompt = f"""your task is to solve a given question, this is a vision language task where the question requires to understand the given image/images(if specified in the question). 
            To solve the question you have to generate a plan of actions in which you can use a tool if required, it is called vit which you can incorporate in your output using queries <q> your query within parenthesis </q>,
            This enables you to ask questions about input image/images to an vision expert, this will return rich response containing information from the image/images for your query."""
        img1_desc = blip.call("Describe the image in detail", "left")
        img2_desc = blip.call("Describe the image in detail", "right")
        user_prompt = f"""solve the vision language task for answering the question based on two input images( first image is called left and second image is called right) 
        left image description:
        {img1_desc}
        right image description:
        {img2_desc} 
        With this information you have to solve question: \"{question}\". 
            For which generate a step by step plan which contains details of what is required to be queried about the images and the logic of utilizing these informations to solve the question. All the queries have better results when query are segregated for left and right images in different steps. And there is a limit of one Query defined in one step. So split the step into two seperates steps when required. 
            The Format of the plan should be
            Step 1: detail on what the ideation is.
            Query: the exact query from the image if required
            
            Step 2: detail on what the ideation is.
            Query: the exact query from the image if required
            ... soon ...
            Step n: detail on what the ideation is
            Query: the exact query from the image if required 
            
            num_steps: n where is n is the number of proposed steps. giving this metrics is very important"""
 
        print("left img desc", img1_desc)
        print("right img desc", img2_desc)
        plan_of_action_flag = 0
        num_steps = -1
        while 6>plan_of_action_flag>=0 :
            chat_completion = openai.ChatCompletion.create(
                                model=model,
                                deployment_id=deployment_name,
                                messages=[{"role":"system", "content":prompt}, {"role":"user", "content":user_prompt}]
                            )
            
            response = chat_completion["choices"][0]['message']['content']
            print(response)
            idx = response.find("num_steps: ")
            if idx>=0:
                try:
                    num_steps = int(response[idx+len("num_steps: "):].split(" ")[0])
                    plan_of_action_flag = -1
                except:
                    pass
        if num_steps==-1:
            print("!!!!!!!!!!! Failed in Planning !!!!!!!!!!!!!")
            continue
        
        r = React([blip], model,deployment_name, verbose=True, except_thought=True) # deployment_name
        cot_flag = 0
        for i in range(num_steps):
            try:
                if i==0:
                    response = r.chat(response+f"""\n Your a good bot and your task is to utilize the vit function and help to run the specified query only do not use your knowledge to answer queries but only to interpret it. Always use tools and never skip using tools for any query. do not produce multiple thought and action in one reply if there is a query in the current step.
        
        Now your first task is to Execute Step {i+1} and then wait for Observation you never generate Observation in any circumstance. If your next word is Observation just add eos token.
        """)
                else:
                    response = r.chat(f"""now execute the Step {i+1} and wait for Observation you never generate Observation in any circumstance. If your next word is Observation just add eos token.
        """)
            except Exception as err:
                cot_flag = 1
                print(err)
                print(r.react_agent.chat_history)
                print("!!!!!!!!!!! Failed in COT !!!!!!!!!!!!!")
        response = r.chat(f"Finally with the above chain of thoughts and observation generate only an answer in the following format 'Answer: True' or 'Answer: False', which answers the original question.")
        pred = 0
        pred += int(str(response).lower().find("true")>=0)
        pred -= int(str(response).lower().find("false")>=0)
        if pred==0:
            pred+= int(str(response).lower().find("yes")>=0)
            pred-= int(str(response).lower().find("no")>=0)

        if pred==0:
            print("Analyze")
        pred_label = int(pred>0)
        if data["label"] == pred_label:
            count += 1
        print("\n--------------------\n")
        time.sleep(3)
    print(100.0 * (count/len(dataset)))
    
if __name__ == "__main__":
    main()