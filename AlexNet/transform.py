import torchvision.transforms as transforms

"""
According to the original AlexNet paper, the authors implemented the following 
augmentation techniques:
    - horizontal reflection
    - random sized crops from 256x256 to 224x224

We will write another tansforms recipe later that something else
"""

def transform_img(img):
    alexnet_transform = transforms.Compose([transforms.Resize(size=(255,255)),
                                            transforms.RandomHorizontalFlip(0.5),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                 std=[0.229, 0.224, 0.225])
                                            ])

    return alexnet_transform(img)
