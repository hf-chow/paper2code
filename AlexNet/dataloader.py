import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image

class ImageNetDataset(Dataset):
    """
    We don't really need to create a custom dataset since ImageNet datasets
    is already included in the Pytorch API, but we are doing it anyway to 
    understand the Dataset class in Pytorch.

    A custom dataset class is a subclass of the default Dataset class. It is 
    basically a map pointing to the location where the labels and the data 
    reside. As such the custom class should overwrite the __getitem__() method.
    Which in this case includes: img_path, image and label.
    Overwriting __len__() is helpful since we will subsample from the train 
    set and split it into train and test sets.
    """
    def __init__(self, 
                 annotations_file,
                 img_dir,
                 transform=None,
                 target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
"""
train_labels_path = "/Users/nara/personal/paper2code/AlexNet/utils/train_labels.csv"
train_images_path = "/Users/nara/personal/datasets/imagenet-mini/train/"

val_labels_path = "/Users/nara/personal/paper2code/AlexNet/utils/val_labels.csv"
val_images_path = "/Users/nara/personal/datasets/imagenet-mini/val/"

train_data = ImageNetDataset(train_labels_path, train_images_path)
val_data = ImageNetDataset(val_labels_path, val_images_path)
"""
