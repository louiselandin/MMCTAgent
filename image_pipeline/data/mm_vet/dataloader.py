import os
import json
import random
import torch
import glob

from torch.utils.data import Dataset, DataLoader
import json
from PIL import Image

class MMVETDataset(Dataset):
    """NVLR2 single item only dataset."""

    def __init__(self, json_file, root_data_dir, shuffle =True, return_paths=False, images_only=False):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.images_only = images_only
        self.items = []
        with open(json_file, 'r') as f:
            data = json.load(f)
        data_num = len(data)
        for i in range(len(data)):
            id = f"v1_{i}"
            imagename = data[id]['imagename']
            img_path = os.path.join(root_data_dir, imagename)
            query = data[id]['question']
            self.items.append({
                "identifier": id,
                "image_path": img_path,
                "query_text": query.strip(),
                "answer": data[id]["answer"],
                "capability": data[id]["capability"]
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
        if self.images_only:
            img1 = Image.open(img1_path).convert("RGB")
            return img1, int(element["identifier"][3:])
        if not self.return_paths:
            img1 = Image.open(img1_path).convert("RGB")
        else:
            img1 = img1_path
        sample = {"img1": img1, "query_text": element["query_text"], "identifier": element["identifier"], "answer": element["answer"], "capability": element["capability"]}

        return sample

    def fetch_item(self, idx):
        elements = list(filter(lambda it: it['identifier'] == idx, self.items))
        try:
            element = elements[0]
        except:
            print(idx)
            raise Exception("This is not a valid identifier")
        img1_path = os.path.join(self.root_data_dir, element["image_path"])
        if not self.return_paths:
            img1 = Image.open(img1_path).convert("RGB")
        else:
            img1 = img1_path
        sample = {"img1": img1, "query_text": element["query_text"], "identifier": element["identifier"], "answer": element["answer"], "capability": element["capability"]}

        return sample



    


if __name__ == "__main__":
    nlvr_single = MMVETDataset("~/COT_HRL_VQA/data/mm_vet/mm-vet.json", "~/COT_HRL_VQA/data/mm_vet/images")
    # for single in nlvr_single:
    #     print(single)
    # dataloader = DataLoader(nlvr_single, batch_size=1,
    #                     shuffle=True, num_workers=1)
    # print(next(iter(dataloader)))