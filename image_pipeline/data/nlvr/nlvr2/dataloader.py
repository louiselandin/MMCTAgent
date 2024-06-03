import os
import json
import random
import torch
import glob

from torch.utils.data import Dataset, DataLoader
import json
from PIL import Image

class NLVR2Dataset(Dataset):
    """NVLR2 single item only dataset."""

    def __init__(self, json_file, root_data_dir, prefix, shuffle =True, return_paths=False):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.items = []
        with open(json_file, mode="r", encoding="utf-8") as reader:
            for line in reader:
                data = json.loads(line)
                path = os.path.join(prefix, str(data["directory"])) if "directory" in data else prefix
                path = os.path.join(path, "-".join(data["identifier"].split("-")[:-1]))
                query_text = data["sentence"]
                
                self.items.append({
                    "image_path": path + "-img0.png",
                    "image2_path": path + "-img1.png",
                    "query_text": query_text,
                    "label": 1 if data["label"] == "True" else 0,
                    "identifier": data["identifier"], 
                })
        if shuffle:
            random.shuffle(self.items)
        self.return_paths = return_paths
        self.root_data_dir = root_data_dir
        

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        element = self.items[idx]
        img1_path = os.path.join(self.root_data_dir, element["image_path"])
        img2_path = os.path.join(self.root_data_dir, element["image2_path"])
        if not self.return_paths:
            img1 = Image.open(img1_path).convert("RGB")
            img2 = Image.open(img2_path).convert("RGB")
        else:
            img1 = img1_path
            img2 = img2_path
        sample = {"img1": img1, "img2": img2, "query_text": element["query_text"], "identifier": element["identifier"], "label": element["label"]}

        return sample

    def fetch_item(self, idx):
        elements = list(filter(lambda it: it['identifier'] == idx, self.items))
        try:
            element = elements[0]
        except:
            print(idx)
            raise Exception("This is not a valid identifier")
        img1_path = os.path.join(self.root_data_dir, element["image_path"])
        img2_path = os.path.join(self.root_data_dir, element["image2_path"])
        if not self.return_paths:
            img1 = Image.open(img1_path).convert("RGB")
            img2 = Image.open(img2_path).convert("RGB")
        else:
            img1 = img1_path
            img2 = img2_path
        sample = {"img1": img1, "img2": img2, "query_text": element["query_text"], "identifier": element["identifier"], "label": element["label"]}

        return sample



    


if __name__ == "__main__":
    nlvr_single = NLVR2Dataset("~/COT_HRL_VQA/data/nlvr/nlvr2/data/test2.json", "~/nlvr", "test2")
    # for single in nlvr_single:
    #     print(single)
    # dataloader = DataLoader(nlvr_single, batch_size=1,
    #                     shuffle=True, num_workers=1)
    # print(next(iter(dataloader)))