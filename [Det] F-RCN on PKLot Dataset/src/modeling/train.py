import pandas as pd
import numpy as np
import cv2
import os
import re

from PIL import Image

import glob
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler

from IPython.display import display
from matplotlib import pyplot as plt
from os import path

DIR_INPUT = '/home/arti/DL/PyCharmPj/[Det] F-RCN on PKLot Dataset/data'
DIR_MODEL_SAVE = '/home/arti/DL/PyCharmPj/[Det] F-RCN on PKLot Dataset/models'
DIR_TRAIN = '{DIR_INPUT}/train/'
lable_path_train = '{DIR_INPUT}/train/_annotations.txt'
num_classes = 2




from transforms import get_train_transform, get_valid_transform
from dataset import PKLotDataset, dataframe_from_yolov4_format, expand_bbox, get_data_labeling

df = get_data_labeling(lable_path_train)

image_ids = df['image_id'].unique()
valid_df = df[df['image_id'].isin(image_ids[-1500:])]
train_df = df[df['image_id'].isin(image_ids[:-1500])]

train_dataset = PKLotDataset(train_df, DIR_TRAIN, get_train_transform())
valid_dataset = PKLotDataset(valid_df, DIR_TRAIN, get_valid_transform())






# load a model; pre-trained on COCO
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

print("Model loaded")

num_classes = 2  # 1 class (car) + background

# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)



# split the dataset in train and test set
indices = torch.randperm(len(train_dataset)).tolist()

train_data_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn
)

valid_data_loader = DataLoader(
    valid_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn
)

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
device

# Train
model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
lr_scheduler = None

num_epochs = 1

loss_hist = Averager()
itr = 1

print("Train started!")
print("Loader len: ", len(train_data_loader))

for epoch in range(num_epochs):
    loss_hist.reset()

    for images, targets, image_ids in train_data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)  # model class can takes 2 params and return loss_dict
        # also can do : outputs = model(images)

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        loss_hist.send(loss_value)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if itr % 100 == 0:
            print("Iteration #{itr} loss: {loss_value}")

        itr += 1

    # update the learning rate
    if lr_scheduler is not None:
        lr_scheduler.step()

    print("Epoch #{epoch} loss: {loss_hist.value}")


model.eval()
torch.save(model.state_dict(), DIR_MODEL_SAVE+'/fasterrcnn.pth')