# ====================================================
# Library
# ====================================================
import requests
import shutil
from pathlib import Path
from contextlib import contextmanager
from collections import defaultdict, Counter

import scipy as sp
import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from tqdm.auto import tqdm
from functools import partial

from PIL import Image

import cv2
from skimage import io
import torch
from torch import nn
import os
from datetime import datetime
import time
import random
import torchvision
from torchvision import transforms
import pandas as pd
import numpy as np
from tqdm import tqdm
import timm
from glob import glob
from sklearn.model_selection import GroupKFold, StratifiedKFold

import sklearn
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss
from sklearn import metrics
import warnings
import pydicom

from torch.utils.data import Dataset,DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from torch.cuda.amp import autocast, GradScaler
from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F

from torch.optim import Adam, SGD
import torchvision.models as models
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau

from albumentations import (
    Compose, OneOf, Normalize, Resize, RandomResizedCrop, RandomCrop, HorizontalFlip, VerticalFlip,
    RandomBrightness, RandomContrast, RandomBrightnessContrast, Rotate, ShiftScaleRotate, Cutout,
    IAAAdditiveGaussianNoise, Transpose
    )
from albumentations.pytorch import ToTensorV2
from albumentations import ImageOnlyTransform

# from efficientnet_pytorch import EfficientNet

from scipy.ndimage.interpolation import zoom

# API
from bottle import post, run, request, response, route
import pandas as pd
import joblib
import json

from datetime import date, datetime
import time

from os import path

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# ====================================================
# CFG
# ====================================================
class CFG:
    num_workers=4
    model_name='resnext50_32x4d'  # 'resnext50_32x4d'
    size=400 ###
    scheduler='CosineAnnealingWarmRestarts' # ['ReduceLROnPlateau', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts']
    T_0=10 # CosineAnnealingWarmRestarts
    epochs=10
    lr=1e-4
    min_lr=1e-6
    batch_size=16 # 32
    weight_decay=1e-6
    gradient_accumulation_steps=1
    max_grad_norm=1000
    seed=42
    target_size=4
    target_col='label'
    n_fold=5
    trn_fold=[0, 1, 2, 3, 4]
    train=True
    inference=False
    print_freq=100
    smoothing=0.05


# ====================================================
# MODEL
# ====================================================
class CustomResNext(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.model = models.resnext50_32x4d(pretrained=pretrained)

        n_features = self.model.fc.in_features
        self.model.fc = nn.Linear(n_features, CFG.target_size)

    def forward(self, x):
        x = self.model(x)
        return x


# ====================================================
# Transforms
# ====================================================
def get_transforms():
    return Compose([
            Resize(CFG.size, CFG.size),
            Normalize(
                mean=[0.5507, 0.5094, 0.4702],
                std=[0.2580, 0.2618, 0.2764],
            ),
            ToTensorV2(),
    ])


def collect_image(id):
    # collect image
    url = 'https://../images/{}/{}.jpg?rule=gallery'.format(id[:2], id)
    image = requests.get(url).content

    if len(image) > 0:
        nparr = np.fromstring(image, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img_np
    else:
        return 0


def make_pred(model_, id_):
    image = collect_image(id_)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    augmented = get_transforms()(image=image)
    image = augmented['image']

    image.unsqueeze_(0)
    image = image.to(device)
    outputs = model_(image)
    preds = torch.max(outputs, 1)[1]
    return preds, mapping[preds.item()]


@post('/predict/id')
def predict_id():
    try:
        o = json.load(request.body)
        pred = make_pred(o)
        return pred
    except:
        print('exception')


@route('/health')
def health_check():
    return 'OK'


names = ['good_flat', 'granny_flat', 'empty_flat', 'empty_flat_good']
mapping = {0: names[0], 1: names[1], 2: names[2], 3: names[3]}

model = CustomResNext(pretrained=False)
model.load_state_dict(torch.load(path.realpath(path.curdir)+'/models/model_26_04_v3.pth', map_location=torch.device('cpu')))
model.to(device)
model.eval()


run(host='0.0.0.0', port=8080, reloader=True)
