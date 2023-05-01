from dataloader import ImageNetDataset
from transform import transform_img
from model import AlexNet
from torchvision.transforms import Lambda

train_labels_path = "/Users/nara/personal/paper2code/AlexNet/utils/train_labels.csv"
train_images_path = "/Users/nara/personal/datasets/imagenet-mini/train/"

val_labels_path = "/Users/nara/personal/paper2code/AlexNet/utils/val_labels.csv"
val_images_path = "/Users/nara/personal/datasets/imagenet-mini/val/"

train_data = ImageNetDataset(train_labels_path,
                             train_images_path,
                             transform=transform_img,
                             target_transform=Lambda(lambda y: torch.zeros(
                                 10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)))
val_data = ImageNetDataset(val_labels_path,
                           val_images_path,
                           transform=transform_img,
                           target_transform=Lambda(lambda y: torch.zeros(
                                 10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)))

