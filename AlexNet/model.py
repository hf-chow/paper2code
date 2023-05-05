from torch import nn, flatten

class AlexNet(nn.Module):
    """
    Similar to the Dataset class, a custom architecture is defined by
    subclassing the nn.Module class. In particular, we need to overwrite the
    definition for __init__()
    """
    def __init__(self):
        super().__init__()          #Inheriting the init from the superclass
        self.net = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(64, 192, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(192, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.AdaptiveAvgPool2d((6,6)),
                nn.Flatten(),
                nn.Dropout(p=0.5),
                nn.Linear(256*6*6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(4096, 1000),
                )

    def forward(self, x):
        x = self.net(x)
        return x
