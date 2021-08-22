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
from matplotlib import pyplot as plt



def train_one_epoch(epoch_num, train_data_loader):
    print("Epoch - {} Started".format(epoch_num))
    st = time.time()

    retinanet.train()

    epoch_loss = []

    for iter_num, data in enumerate(train_data_loader):

        # Reseting gradients after each iter
        optimizer.zero_grad()

        # Forward
        classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot'].cuda().float()])

        # Calculating Loss
        classification_loss = classification_loss.mean()
        regression_loss = regression_loss.mean()

        loss = classification_loss + regression_loss

        if bool(loss == 0):
            continue

        # Calculating Gradients
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

        # Updating Weights
        optimizer.step()

        # Epoch Loss
        epoch_loss.append(float(loss))

        if iter_num % 100 == 0:
            print(
                'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                    epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(epoch_loss)))

        del classification_loss
        del regression_loss

    # Update the learning rate
    # if lr_scheduler is not None:
    # lr_scheduler.step()

    et = time.time()
    print("\n Total Time - {}\n".format(int(et - st)))

def valid_one_epoch(epoch_num, valid_data_loader):
    print("Epoch - {} Started".format(epoch_num))
    st = time.time()

    epoch_loss = []

    for iter_num, data in enumerate(valid_data_loader):

        with torch.no_grad():

            # Forward
            classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot'].cuda().float()])

            # Calculating Loss
            classification_loss = classification_loss.mean()
            regression_loss = regression_loss.mean()
            loss = classification_loss + regression_loss

            # Epoch Loss
            epoch_loss.append(float(loss))

            if iter_num % 100 == 0:
                print(
                    'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                        epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(epoch_loss)))

            del classification_loss
            del regression_loss

    et = time.time()
    print("\n Total Time - {}\n".format(int(et - st)))

    # Save Model after each epoch
    torch.save(retinanet, DIR_MODEL_SAVE+"/retina.pth")



df = get_data_labeling(lable_path_train)

image_ids = df['image_id'].unique()
valid_df = df[df['image_id'].isin(image_ids[-1500:])]
train_df = df[df['image_id'].isin(image_ids[:-1500])]

# Dataset Object
train_dataset = GWD(train_df, DIR_TRAIN, mode = "train", transforms = T.Compose([Augmenter(), Normalizer(), Resizer()]))
valid_dataset = GWD(valid_df, DIR_TRAIN, mode = "valid", transforms = T.Compose([Normalizer(), Resizer()]))

# DataLoaders
train_data_loader = DataLoader(
    train_dataset,
    batch_size = 4,
    shuffle = True,
    num_workers = 4,
    collate_fn = collater
)

valid_data_loader = DataLoader(
    valid_dataset,
    batch_size = 4,
    shuffle = True,
    num_workers = 4,
    collate_fn = collater
)


test_data_loader = DataLoader(
    valid_dataset,
    batch_size = 1,
    shuffle = True,
    num_workers = 4,
    collate_fn = collater
)

### Utilize GPU if available

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
torch.cuda.empty_cache()

### I am using Pre-trained Resnet50 as backbone

retinanet = model.resnet18(num_classes = 2, pretrained = True, model_dir=DIR_MODEL_SAVE)

# Defininig Optimizer
optimizer = torch.optim.Adam(retinanet.parameters(), lr = 0.0001)

# Learning Rate Scheduler
#lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 5, gamma=0.5)

retinanet.to(device);
epochs = 1

print('training..')

for epoch in range(epochs):
    # Call train function
    train_one_epoch(epoch, train_data_loader)

    # Call valid function
    valid_one_epoch(epoch, valid_data_loader)


# Eval

### Sample Results
retinanet.eval()
unnormalize = UnNormalizer()

for iter_num, data in enumerate(test_data_loader):

    if iter_num > 10:
        break

    # Getting Predictions
    scores, classification, transformed_anchors = retinanet(data['img'].cuda().float())

    idxs = np.where(scores.cpu() > 0.5)
    img = np.array(255 * unnormalize(data['img'][0, :, :, :])).copy()

    img[img < 0] = 0
    img[img > 255] = 255

    img = np.transpose(img, (1, 2, 0))

    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    for j in range(idxs[0].shape[0]):
        bbox = transformed_anchors[idxs[0][j], :]
        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = int(bbox[2])
        y2 = int(bbox[3])

        cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

    ax.imshow(img)

    continue

