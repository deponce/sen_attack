from torch.utils.data import Dataset
import torchvision.datasets as datasets
from torchvision import transforms
import glob
import os
import pickle
from PIL import Image


class random_sampler(datasets.ImageNet):
    """docstring for loader"""
    def __init__(self, root, t_split="val", transform=None):
        self.root = root
        UID_CLASSID_PKL_PATH = os.path.join(self.root, "uid_to_classID.pkl")
        if os.path.exists(UID_CLASSID_PKL_PATH):
            self.uid_to_classid_path = UID_CLASSID_PKL_PATH
            file = open(UID_CLASSID_PKL_PATH,'rb')
            self.uid_to_classid = pickle.load(file)
            file.close()
        super().__init__(self.root, split="val")
        self.transform = transform
        self.root = os.path.join(self.root, t_split)
        self.img_list = glob.glob(self.root + '/*/*/*.JPEG')
        # '/home/h2amer/AhmedH.Salamah/ilsvrc2012/train/shard-4/426/n07584110_4206.JPEG'

        self.samples_count = {}
        for key, values in self.wnid_to_idx.items():
            self.samples_count[key] = 0

    def __getitem__(self, index):
        img_path = self.img_list[index]
        parse = img_path.split("/")
        file_name = parse[-1]
        GT = file_name.split("_")[0]
        target = self.uid_to_classid[GT]
        # target = self.wnid_to_idx[GT]
        self.samples_count[GT] += 1
        img_tensor = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img_tensor)
        return img_tensor, target

    def __len__(self):
        return len(self.img_list)

def load_dict(root):
    UID_CLASSID_PKL_PATH = os.path.join(root, "uid_to_classID.pkl")
    samples_count = {}
    if os.path.exists(UID_CLASSID_PKL_PATH):
        uid_to_classid_path = UID_CLASSID_PKL_PATH
        file = open(UID_CLASSID_PKL_PATH,'rb')
        uid_to_classid = pickle.load(file)
        file.close()
    for key, values in uid_to_classid.items():
        samples_count[key] = 0
    # print(samples_count)
    return samples_count