from dataloader import ImageNetDataset
from transform import transform_img
from model import AlexNet
from torchvision.transforms import Lambda
import torch
from torch.utils.data import DataLoader

train_labels_path = "/Users/nara/personal/paper2code/AlexNet/train_labels.csv"
train_images_path = "/Users/nara/personal/datasets/imagenet-mini/train/"

val_labels_path = "/Users/nara/personal/paper2code/AlexNet/val_labels.csv"
val_images_path = "/Users/nara/personal/datasets/imagenet-mini/val/"

device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using {device} for training")

BATCH_SIZE = 64
    
train_data = ImageNetDataset(train_labels_path,
                             train_images_path,
                             transform=transform_img,
                             )

val_data = ImageNetDataset(val_labels_path,
                           val_images_path,
                           transform=transform_img,
                           )

model = AlexNet().to(device)

loss_fn = torch.nn.CrossEntropyLoss()
optim = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE)
val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE)

for X, y in train_dataloader:
    print(f"{X.shape}")
    print(f"{y.shape}")

def train(dataloader, model, loss_fn, optim):
    size = len(dataloader.dataset)
    model.train()
    for batch, data in enumerate(dataloader):
        print(data)
        X, y = data[0].to(device), data[1].to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        optim.zero_grad()
        loss.backward()
        optim.step()

    if batch % 100 == 0:
        loss, current = loss.item(), (batch + 1) * len(X)
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def val(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    val_loss, correct = 0, 0
    with torch.no_grad():
        for data in dataloader:
            X, y = data[0].to(device), data[1].to(device)
            pred = model(X)
            val_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            test_loss /= num_batches
            correct /= size
            print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epoch = 10

for i in range(epoch):
    print(f"Epoch {i+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optim)
    test(val_dataloader, model, loss_fn)
print("Complete")
