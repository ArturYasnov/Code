import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

# Albumentations
def get_train_transform():
    return A.Compose([
        A.Resize(height=256, width=256, p=1.0),
        A.Flip(0.5),
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

def get_valid_transform():
    return A.Compose([
        A.Resize(height=256, width=256, p=1.0),
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
