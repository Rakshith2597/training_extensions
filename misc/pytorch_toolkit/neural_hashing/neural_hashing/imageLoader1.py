import numpy as np
import torch.utils.data as data
import os
import PIL.Image as Image
from loader_mnist import dataset


def custom_dataset(fpath):
    return dataset(fpath)


def find_classes(dir):
    classes = [d for d in os.listdir(
        dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


class Dataloder_img(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        classes, class_to_idx = find_classes(root)
        #samples = oversample(root)
        samples = custom_dataset(root)
       # print(len(samples))
        self.root = root

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        # print('index:',(index))
        path1, path2, target, target1, target2 = self.samples[index]

        a1 = np.load(path1)
        a2 = np.load(path2)
        # print(a1)
        # print(a1.dtype)
       # print(a2.dtype)
        #sample1 = torch.from_numpy(np.load(path1))
        #sample2 = torch.from_numpy(np.load(path2))
       # print(sample1.dtype)
       # print(sample2.dtype)
        #sample1 = Image.open(sample1).convert("RGB")

        sample1 = Image.fromarray(a1)
        # print(sample1)
        sample2 = Image.fromarray(a2)
        if self.transform is not None:
            sample1 = self.transform(sample1)
            sample2 = self.transform(sample2)

        if self.target_transform is not None:
            target = self.target_transform(target)
            target1 = self.target_transform(target1)
            target2 = self.target_transform(target2)
            # c cluster_centers = self.target_transform(cluster_centers)

        return sample1, sample2, target, target1, target2  # , cluster_centers

    def __len__(self):
        return len(self.samples)
