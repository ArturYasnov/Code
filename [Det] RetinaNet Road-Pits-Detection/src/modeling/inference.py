import os
import re
import cv2
import time
import numpy as np
import pandas as pd


import torch
import torch.optim as optim
import torchvision.transforms as T
from torchvision.utils import make_grid
from torch.utils.data import DataLoader, Dataset

from retinanet import model
from retinanet.dataloader import collater, Resizer, Augmenter, Normalizer, UnNormalizer

import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt



device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
torch.cuda.empty_cache()

retinanet = model.resnet18(num_classes = 2, pretrained = False)

# Loading Pre-trained model - if you load pre-trained model, comment above line.
retinanet = torch.load(DIR_MODEL_SAVE+"/retina.pth")
retinanet.to(device);
retinanet.eval()

unnormalize = UnNormalizer()

it = iter(test_data_loader)
data = next(it)

images = data['img'].cuda().float()

print(images.shape)

outputs = retinanet(images)

print(images[0])

boxes = outputs[2].detach().cpu().numpy().astype(np.int32)
sample = images[0].permute(1,2,0).cpu().numpy()



for box in boxes:
    cv2.rectangle(sample,
                  (box[0], box[1]),
                  (box[2], box[3]),
                  (0, 0, 120), 1)


matplotlib.image.imsave(DIR_INPUT+"/sample.png", sample)

