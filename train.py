import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import cv2
import os
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.datasets as dset
import torchvision.utils as vutils
import torchvision.models as models
import glob
import PIL
import copy
import argparse
import time
from collections import defaultdict

# Align working directory
os.chdir(os.path.dirname(os.path.realpath(__file__)))

from model import *
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg = models.vgg19(pretrained=True).features.eval()

parser = argparse.ArgumentParser(description=" ")
parser.add_argument("--data_directory", required=True, help="Directory of dataset. Style and content should each be stored at /trainA and /trainB.")
parser.add_argument("--output_directory", required=False, default="./outputs" help="Directory of outputs: model checkpoints and generated images(if chosen to)")
parser.add_argument("--generate_image", required=False, default=False, help="Whether to generate images in certain epochs")
parser.add_argument("--batch_size", required=False, default=8, help="Batch size of training")
parser.add_argument("--epochs", required=False, default=20, help='Number of epochs to train on')
parser.add_argument("--learning_rate", required=False, default=0.0002, help="Learning rate to update parameters")

args = parser.parse_args()

# Hyperparameters
root = args.data_directory
batch_size = args.batch_size
n_epochs = args.epochs
lr = args.learning_rate
image_size = 224
nc = 3
out_dir = args.output_directory
generate_images = args.generate_image

# Create Image dataset, returning style and content as dictionary
dataset = ImageDataset(root=root,
                           transform = transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                           ]))

loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

encoder = build_encoder(base_model=vgg)
decoder = build_decoder()
model = Net(encoder, decoder)
model.to(device)
optimizer = optim.Adam(model.decoder.parameters(), lr=lr, betas=(0.5, 0.999))
Tensor = torch.cuda.FloatTensor if device == 'cuda' else torch.FloatTensor

output = defaultdict(list)

if __name__ == "__main__":

    for epoch in range(n_epochs):
        t0 = time.time()
        s, c = 0, 0
        for i, batch in enumerate(loader):
            style = batch['style']
            content = batch['content']
            if torch.cuda.is_available():
                style, content = style.cuda(), content.cuda()

            optimizer.zero_grad()
            loss_c, loss_s = model(content, style)
            loss = 0.5 * (loss_c + loss_s)
            loss.backward()
            optimizer.step()

            s += loss_s.item()
            c += loss_c.item()

        t1 = time.time()
        output['epoch'].append(epoch+1)
        output['Loss_C'].append(c)
        output['Loss_S'].append(s)

        print("Epoch: {}".format(epoch+1))
        print("Time Taken: {:.1f}m".format((t1-t0)/60))
        print("Content Loss: {:.4f}".format(c))
        print("Style Loss: {:.4f}".format(s))
        print('-'*50)

        if epoch+1 % 5 == 0:
            torch.save(model.state_dict(), out_dir+"/model_epoch_{}".format(epoch+1))

            if generate_images:

                alphas = np.arange(start=0.2, stop=1.2, step=0.2)

                j = np.random.randint(0, 8)
                s = next(iter(loader))['style'][j].unsqueeze(0).cuda()
                c = next(iter(loader))['content'][j].unsqueeze(0).cuda()
                outputs = [model(c, s, output_image=True, alpha=a) for a in alphas]

                s_img = s.squeeze(0).detach().cpu().permute(1,2,0).numpy()
                c_img = c.squeeze(0).detach().cpu().permute(1,2,0).numpy()
                imgs = [s_img, c_img] + [output.squeeze(0).detach().cpu().permute(1,2,0).numpy() for output in outputs]

                fig, ax = plt.subplots(1, 7, figsize=(13,10), dpi=100)
                for k in range(7):
                    ax[k].imshow(imgs[k])
                    ax.axis("off")

                fig.savefig(out_dir+"/images_epoch_{}".format(epoch+1))
