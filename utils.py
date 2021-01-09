import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.datasets as dset
import torchvision.utils as vutils
import glob
import PIL
import cv2

def to_rgb(image):
    rgb_image = PIL.Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image

class ImageDataset(Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.transform = transform
        self.style = sorted(glob.glob(os.path.join(root, "trainA/" + "*.*")))
        self.content = sorted(glob.glob(os.path.join(root, "trainB/" + "*.*")))

    def __getitem__(self, index):
        img_style = PIL.Image.open(self.style[index % len(self.style)])
        img_content = PIL.Image.open(self.content[index % len(self.content)])
        img_style, img_content = to_rgb(img_style), to_rgb(img_content)
        img_style, img_content = self.transform(img_style), self.transform(img_content)

        return {"style": img_style, "content": img_content}

    def __len__(self):
        return max(len(self.style), len(self.content))

def compute_stat(x, eps=1e-8):
    assert len(x.size()) == 4, "Invalid Tensor Shape: {}".format(x.size())
    N, C, H, W = x.size()
    x = x.view(N, C, -1)
    mean = torch.mean(x, dim=2).view(N, C, 1, 1)
    std = torch.std(x, dim=2).view(N, C, 1, 1) + eps
    return mean, std

def AdaIN(style, content):
    assert style.size()[:2] == content.size()[:2], "Dimension Mismatch between style and content, {} vs {}".format(style.size(), content.size())
    size = style.size()
    mean_s, std_s = compute_stat(style)
    mean_c, std_c = compute_stat(content)
    scaled_c = (content - mean_c.expand(size)) / std_c.expand(size)
    return std_s.expand(size) * scaled_c + mean_s.expand(size)
