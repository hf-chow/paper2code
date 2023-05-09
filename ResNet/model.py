from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self,
                 kernel_size,
                 in_channels,
                 out_channels, 
                 stride,
                 downsample=None):
        super().__init__()
        self.net = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride),
                nn.BatchNorm2d(out_channels),
                )
        self.downsample = downsample
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        x = self.net(x)
        if self.downsample:
            residual = self.downsample(x)
        x += residual
        x = nn.ReLU(x)
        return x

class ResNet(nn.Module):
    def __init__(self, block, layers, out_channels):
        super().__init__()
        self.in_channels = 64
        self.out_channels = out_channels
        self.conv1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2),
                nn.BatchNorm2d(64),
                nn.ReLU()
                )
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2)
        self.conv2 = self._make_layer(block, 64, layers[0], stride=1)
        self.conv3 = self._make_layer(block, 128, layers[1], stride=2)
        self.conv4 = self._make_layer(block, 256, layers[2], stride=2)
        self.conv5 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc = nn.Linear(512, self.out_channels)

    def _make_layer(self, block, planes, num_blocks, stride):
        downsample = None

        if stride != 1 or self.in_channels !=planes:
            downsample = nn.Sequential(
                    nn.Conv2d(self.in_channels, planes, kernel_size=1, stride=2),
                    nn.BatchNorm2d(planes))

        _layers = []
        _layers.append(block(3,
                           self.in_channels,
                           planes,
                           stride,
                           downsample))
        for _ in range(1, num_blocks):
            _layers.append(block(3,
                                planes,
                                planes,
                                stride=1))
        return nn.Sequential(*_layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        x = nn.Flatten(x)
        x = self.fc(x)
        return x

resnet34 = ResNet(ResidualBlock, [3, 4, 6, 3], 10)
print(resnet34)
