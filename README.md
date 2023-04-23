### Ideas proposed in the paper
-Data augmentation
-Dropout
-ReLu activation
-Conv2D

### Architecture
- 1st layer: 224x224x3; 96 kernels 11x11x3 stride 4
- Normalized & Max Pooling
- 2nd layer: 256 kernels 5x5x48
- Normalized & Max Pooling
- 3rd layer: 384 kernels 3x3x256
- 4th layer: 384 kernels 3x3x192
- 5th layer: 256 kernels 3x3x192
- Normalized & Max Pooling
- 6th layer: 4096 fully connected
- 7th layer: 4096 fully connected
- Output: 1000 categories
