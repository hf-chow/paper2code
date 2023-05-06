import torch
from torch import nn
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader

import random
import numpy as np
import matplotlib.pyplot as plt

MNIST_DIR = "/Users/nara/personal/datasets"
train_dataset = MNIST(MNIST_DIR, train=True, download=True,
                      transform=transforms.Compose([
                          transforms.Resize((28,28)),
                          transforms.ToTensor(),
                          transforms.Normalize(mean=0.1307, std=0.3081)]))

test_dataset = MNIST(MNIST_DIR, train=False, download=True,
                      transform=transforms.Compose([
                          transforms.Resize((28,28)),
                          transforms.ToTensor(),
                          transforms.Normalize(mean=0.1307, std=0.3081)]))

train_dataloader = DataLoader(train_dataset, batch_size=64)
test_dataloader = DataLoader(test_dataset, batch_size=64)

"""
for X, y in train_dataloader:
    print(X.shape)
    print(y.shape)
    break

rand_samples = [random.randint(0, len(train_dataloader)) for i in range(9)]

fig = plt.figure(figsize=(10,10))

for i in range(1, len(rand_samples)+1):
    img, label = train_dataset[rand_samples[i-1]]
    fig.add_subplot(3, 3, i)
    plt.title(label)
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")

plt.show()
"""

device = ("cuda" if torch.cuda.is_available()
          else "mps"
          if torch.backends.mps.is_available()
          else "cpu"
        )

print(f"Using {device} for training")

class LeNet(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
                nn.Conv2d(1, 6, kernel_size=5, padding=2),
                nn.Sigmoid(),
                nn.AvgPool2d(2, stride=2),
                nn.Conv2d(6, 16, kernel_size=5),
                nn.Sigmoid(),
                nn.AvgPool2d(2, stride=2),
                nn.Flatten(),
                nn.Linear(5*5*16, 120),
                nn.Sigmoid(),
                nn.Linear(120, 84),
                nn.Sigmoid(),
                nn.Linear(84, 10)
                )
    def forward(self, x):
        x = self.net(x)
        return x

model = LeNet().to(device)

loss_fn = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=1e-3)

def train(dataloader, model, optim, loss_fn):
    size = len(dataloader.dataset)
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        pred = model(X)
        loss = loss_fn(pred, y)

        optim.zero_grad()
        loss.backward()
        optim.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

 
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
 

epochs = 10

for i in range(epochs):
     print(f"Epoch {i+1}\n-------------------------------")
     train(train_dataloader, model, optim, loss_fn)
     test(test_dataloader, model, loss_fn)
print("Training Complete")
