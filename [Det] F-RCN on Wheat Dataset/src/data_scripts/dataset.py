import numpy as np
import pandas as pd
import cv2
import re
import glob
import torch
from torch.utils.data import DataLoader, Dataset

from transforms import get_train_transform, get_valid_transform
import albumentations as A


class WheatDataset(Dataset):

    def __init__(self, dataframe, image_dir, transforms=None):
        super().__init__()

        self.image_ids = dataframe['image_id'].unique()
        self.df = dataframe
        self.image_dir = image_dir
        self.transforms = transforms

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        records = self.df[self.df['image_id'] == image_id]

        image = cv2.imread(f'{self.image_dir}/{image_id}.jpg', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        boxes = records[['x', 'y', 'w', 'h']].values

        # [834. 222.  56.  36.] -> [ 834.  222.  890.  258.]
        # x,y,w,h -> x1,y1,x2,y2
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        # w * h
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)

        # there is only one class. 1 - target class value
        labels = torch.ones((records.shape[0],), dtype=torch.int64)

        # suppose all instances are not crowd
        iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        # target['masks'] = None
        target['image_id'] = torch.tensor([index])
        target['area'] = area
        target['iscrowd'] = iscrowd

        if self.transforms:
            sample = {
                'image': image,
                'bboxes': target['boxes'],
                'labels': labels
            }
            sample = self.transforms(**sample)
            image = sample['image']

            target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)

        return image, target, image_id

    def __len__(self) -> int:
        return self.image_ids.shape[0]



def expand_bbox(x):
    r = np.array(re.findall("([0-9]+[.]?[0-9]*)", x))
    if len(r) == 0:
        r = [-1, -1, -1, -1, -1]
    return r


def get_data_labeling(lable_path):
    train_df = pd.read_csv(lable_path)
    train_df['x'] = -1
    train_df['y'] = -1
    train_df['w'] = -1
    train_df['h'] = -1

    train_df[['x', 'y', 'w', 'h']] = np.stack(train_df['bbox'].apply(lambda x: expand_bbox(x)))
    train_df.drop(columns=['bbox'], inplace=True)
    train_df['x'] = train_df['x'].astype(np.float)
    train_df['y'] = train_df['y'].astype(np.float)
    train_df['w'] = train_df['w'].astype(np.float)
    train_df['h'] = train_df['h'].astype(np.float)
    display(train_df.head(3))

    image_ids = train_df['image_id'].unique()
    valid_ids = image_ids[-665:]
    train_ids = image_ids[:-665]
    len(valid_ids), len(train_ids)

    train_df = train_df[train_df['image_id'].isin(train_ids)]
    valid_df = train_df[train_df['image_id'].isin(valid_ids)]

    return train_df, valid_df


