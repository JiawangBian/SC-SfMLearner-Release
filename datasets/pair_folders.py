import torch.utils.data as data
import numpy as np
from imageio import imread
from path import Path
import random
import os


def load_as_float(path):
    return imread(path).astype(np.float32)


class PairFolder(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
        root/scene_1/0000000_0.jpg
        root/scene_1/0000001_1.jpg
        ..
        root/scene_1/cam.txt
        .
        transform functions must take in a list a images and a numpy array (usually intrinsics matrix)
    """

    def __init__(self, root, seed=None, train=True, transform=None):
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)
        scene_list_path = self.root/'train.txt' if train else self.root/'val.txt'
        self.scenes = [self.root/folder[:-1] for folder in open(scene_list_path)]
        self.transform = transform
        self.crawl_folders()

    def crawl_folders(self,):
        pair_set = []
        for scene in self.scenes:
            # intrinsics = np.genfromtxt(scene/'cam.txt').astype(np.float32).reshape((3, 3))
            
            imgs = sorted(scene.files('*.jpg'))
            intrinsics = sorted(scene.files('*.txt'))

            for i in range(0, len(imgs)-1, 2):
                intrinsic = np.genfromtxt(intrinsics[int(i/2)]).astype(np.float32).reshape((3, 3))
                sample = {'intrinsics': intrinsic, 'tgt': imgs[i], 'ref_imgs': [imgs[i+1]]}
                pair_set.append(sample)
        random.shuffle(pair_set)
        self.samples = pair_set

    def __getitem__(self, index):
        sample = self.samples[index]
        tgt_img = load_as_float(sample['tgt'])
        ref_imgs = [load_as_float(ref_img) for ref_img in sample['ref_imgs']]
        if self.transform is not None:
            imgs, intrinsics = self.transform([tgt_img] + ref_imgs, np.copy(sample['intrinsics']))
            tgt_img = imgs[0]
            ref_imgs = imgs[1:]
        else:
            intrinsics = np.copy(sample['intrinsics'])
        return tgt_img, ref_imgs, intrinsics, np.linalg.inv(intrinsics)

    def __len__(self):
        return len(self.samples)
