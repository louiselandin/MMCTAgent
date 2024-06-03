import os
import openai
from data.mm_vet.dataloader import MMVETDataset
from PIL import Image
import requests
import time
from llama_index.llms import OpenAILike
from pipeline import React
from env import ToolReactEnv
from dotenv import load_dotenv, find_dotenv
import json
import queue
import threading
import concurrent.futures
from tqdm import tqdm

load_dotenv(find_dotenv())

from io import StringIO 
import sys

pbar = None

class ParallelCapturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio
        sys.stdout = self._stdout

def process_sample(dataset, env, sample_id, idx=0):
    # print(dataset, idx)
    # print(f"Sample id: {sample_id}")
    # return "asdf"
    data = dataset[sample_id]
    
    with ParallelCapturing() as output:
        global results
        if data["identifier"] in results.keys():
            return "asdf"
        image = data["img1"]
        env.reset(image, idx=idx)
        question = data["query_text"]
        identifier = data["identifier"]
        print("\n--------------------\n")
        print(f"Question: {question} Identifier: {identifier}") 
        cot_flag = 0
        response = env.step(f"""your task is to solve a given question, this is a vision language task where the question requires to understand the given image. To do so you can use the multiple tools to analyze the image, Answer the question: {question} in few words.""", idx=idx)
        # response = env.step(f"""{question}""")
        
        results[identifier] = str(response)
        print("\n--------------------\n")
    time.sleep(3)
    
    # print("Captured output", output)
    return output
    
def worker(dataset, env, worker_id):
    global samples_queue, log_list
    while not samples_queue.empty():
        
        sample_id = samples_queue.get(block=False)
        if sample_id is None:
            return
        out = process_sample(dataset, env, sample_id, worker_id)
        # try:
        #     out = process_sample(dataset, env, sample_id, worker_id)
        #     log_list.append(out)
        # except Exception as e:
        #     print(e)
        
        pbar.update(int(100*(len(dataset) - samples_queue.qsize() )/ len(dataset)))
    return 0

def main():
    global results, pbar
    creds = None
    models = None
    deployment_names = None
    if int(os.environ.get("OPENAI_API_NUM", 0))>0:
        creds = []
        models = []
        deployment_names = []
        for i in range(int(os.environ["OPENAI_API_NUM"])):
            cred = [ os.environ[f"OPENAI_API_KEY_{i+1}"],
                        os.environ[f"OPENAI_API_BASE_{i+1}"],
                        os.environ[f"OPENAI_API_TYPE_{i+1}"],
                        os.environ[f"OPENAI_API_VERSION_{i+1}"]]
            creds.append(cred)
            models.append(os.environ[f"OPENAI_API_MODEL_{i+1}"])
            deployment_names.append(os.environ[f"OPENAI_API_DEPLOYMENT_{i+1}"])
    else:
        openai.api_key = os.environ["OPENAI_API_KEY"]
        openai.api_base = os.environ["OPENAI_API_BASE"]
        if os.environ.get("OPENAI_API_TYPE", None):
            openai.api_type = os.environ["OPENAI_API_TYPE"]
            openai.api_version = os.environ["OPENAI_API_VERSION"]
            models = ["gpt-4"]
            deployment_names = ["gpt-4-deployment"]
        else:
            models = openai.Model.list()
            models = [models["data"][0]["id"]]
            deployment_names = None
    
    
    dataset = MMVETDataset("/home/aiscuser/COT_HRL_VQA/data/mm_vet/mm-vet.json", "/home/aiscuser/COT_HRL_VQA/data/mm_vet/images")
    env = ToolReactEnv(credentials=creds, num_llms=int(os.environ.get("OPENAI_API_NUM", 1)), model_name=models,engine_name=deployment_names, verbose=True, except_thought=True)
    samples = 0
    
    workers_num = int(os.environ.get("OPENAI_API_NUM", 1))
    for i in range(len(dataset)):
        samples_queue.put(i)
    
    pbar = tqdm(total=100)
    threads = [threading.Thread(target=worker, args=(dataset, env, worker_id)) for worker_id in range(workers_num)]
    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()
    pbar.close()
    
if __name__ == "__main__":
    resume = True
    results = {}
    log_list = []
    if resume:
        with open("results_new_manual1.json", 'r') as f:
            results = json.load(f)
    samples_queue = queue.Queue()
    # worker_queue = queue.Queue()
    
    #try:
    main()
    #except Exception as e:
    #    print(e)
    #    print(results)
    #    for log in log_list:
    #        for l in log:
    #            print(l)
    
    try:
        with open("results_new_sing.json", 'w') as f:
            json.dump(results, f, indent=4)
    except:
        print(results)
    try:
        with open("logs/log_best_sing.logs", 'w') as f:
            for log in log_list:
                for l in log:
                    print(l, file=f)
    except:
        for log in log_list:
                for l in log:
                    print(l)