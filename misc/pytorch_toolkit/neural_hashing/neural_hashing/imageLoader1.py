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
        samples = custom_dataset(root)
        self.root = root

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        path1, path2, target, target1, target2 = self.samples[index]

        a1 = np.load(path1)
        a2 = np.load(path2)

        sample1 = Image.fromarray(a1)
        sample2 = Image.fromarray(a2)
        if self.transform is not None:
            sample1 = self.transform(sample1)
            sample2 = self.transform(sample2)

        if self.target_transform is not None:
            target = self.target_transform(target)
            target1 = self.target_transform(target1)
            target2 = self.target_transform(target2)

        return sample1, sample2, target, target1, target2

    def __len__(self):
        return len(self.samples)
