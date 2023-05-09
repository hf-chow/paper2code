from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self,
                 kernel_size,
                 in_channels,
                 out_channels, 
                 stride,
                 downsample):
        super().__init__()
        self.net = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride),
                nn.BatchNorm2d(out_channels),
                )
        self.downsample = downsample

    def forward(self, x):
        residual = x
        x = self.net(x)
        if self.downsample:
            residual = self.downsample(x)
        x += residual
        x = nn.ReLU(x)
        return x

