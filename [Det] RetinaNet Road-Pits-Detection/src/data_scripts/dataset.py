import numpy as np
import pandas as pd
import cv2
import re
import glob
import torch
from torch.utils.data import DataLoader, Dataset

from transforms import get_train_transform, get_valid_transform
import albumentations as A


class GWD(Dataset):

    def __init__(self, dataframe, image_dir, mode="train", transforms=None):

        super().__init__()
        self.image_ids = dataframe['image_id'].unique()
        self.df = dataframe
        self.image_dir = image_dir
        self.mode = mode
        self.transforms = transforms

    def __getitem__(self, index: int):

        # Retriving image id and records from df
        image_id = self.image_ids[index]
        records = self.df[self.df['image_id'] == image_id]

        # Loading Image
        image = cv2.imread(f'{self.image_dir}/{image_id}', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        # If mode is set to train, then only we create targets
        if self.mode == "train" or self.mode == "valid":

            # Converting xmin, ymin, w, h to x1, y1, x2, y2
            boxes = np.zeros((records.shape[0], 5))
            boxes[:, 0:4] = records[['x', 'y', 'w', 'h']].values
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
            boxes[:, 4] = 1  # This is for label, as we have only 1 class, it is always 1

            # Applying Transforms
            sample = {'img': image, 'annot': boxes}

            if self.transforms:
                sample = self.transforms(sample)

            return sample

        elif self.mode == "test":

            # We just need to apply transoforms and return image
            if self.transforms:
                sample = {'img': image}
                sample = self.transforms(sample)

            return sample

    def __len__(self) -> int:
        return self.image_ids.shape[0]


def dataframe_from_yolov4_format(lable_path):
    truth = {}
    f = open(lable_path, 'r', encoding='utf-8')
    for line in f.readlines():
        data = line.split(" ")
        truth[data[0]] = []
        for i in data[1:]:
            truth[data[0]].append([int(j) for j in i.split(',')])

    data = truth.items()
    data = np.array(list(data))
    df = pd.DataFrame(data)
    return df


def expand_bbox(x):
    r = np.array(re.findall("([0-9]+[.]?[0-9]*)", x))
    if len(r) == 0:
        r = [-1, -1, -1, -1, -1]
    return r


def get_data_labeling(lable_path):
    df = dataframe_from_yolov4_format(lable_path)
    df = df.explode([1]).reset_index(drop=True)
    df.columns = ['image_id', 'bbox']
    df.bbox = df.bbox.astype(str)


    # chech all images exist
    names = glob.glob1(DIR_TRAIN, "*.jpg")
    df = df.assign(exists=0)
    df.loc[df.image_id.isin(names), 'exists'] = 1
    df = df[df.exists==1]
    df = df.drop('exists', axis=1)

    df['x1'] = -1
    df['y1'] = -1
    df['x2'] = -1
    df['y2'] = -1
    df['cls'] = -1

    df[['x1', 'y1', 'x2', 'y2', 'cls']] = np.stack(df['bbox'].apply(lambda x: expand_bbox(x)))
    df.drop(columns=['bbox'], inplace=True)
    df['x1'] = df['x1'].astype(np.float)
    df['y1'] = df['y1'].astype(np.float)
    df['x2'] = df['x2'].astype(np.float)
    df['y2'] = df['y2'].astype(np.float)
    df['cls'] = df['cls'].astype(np.float)

    df.cls = 1
    df['width'] = 640
    df['height'] = 640

    return df


