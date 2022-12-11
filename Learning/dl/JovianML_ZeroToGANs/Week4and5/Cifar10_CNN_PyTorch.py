import torch
import torchvision
import matplotlib.pyplot as plt
import os
import tarfile
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets.utils import download_url
from torch.utils.data import random_split

dataset_url = 'http://files.fast.ai/data/cifar10.tgz'
download_url(dataset_url, root='./data/')

with tarfile.open('./data/cifar10.tgz', 'r:gz') as tar:
    def is_within_directory(directory, target):
        
        abs_directory = os.path.abspath(directory)
        abs_target = os.path.abspath(target)
    
        prefix = os.path.commonprefix([abs_directory, abs_target])
        
        return prefix == abs_directory
    
    def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
    
        for member in tar.getmembers():
            member_path = os.path.join(path, member.name)
            if not is_within_directory(path, member_path):
                raise Exception("Attempted Path Traversal in Tar File")
    
        tar.extractall(path, members, numeric_owner=numeric_owner) 
        
    
    safe_extract(tar, path="./data/")

data_dir = './data/cifar10'

dataset = torchvision.datasets.ImageFolder(data_dir, transform=torchvision.transforms.ToTensor())

val_size = 5000
train_size = len(dataset) - val_size

train_data, val_data = random_split(dataset, [train_size, val_size])
batch_size = 128

train_dataldr = DataLoader(train_data, batch_size, shuffle=True, num_workers=3, pin_memory=True)
val_dataldr = DataLoader(val_data, batch_size * 2, shuffle=True, num_workers=3, pin_memory=True)


class ImageClassifiationBase(torch.nn.Module):
    def train_step(self, batch):
        X_train, y_train = batch
        y_pred = self(X_train)
        loss = F.cross_entropy(y_train, y_train)
        return loss

    def val_step(self, batch):
        X_val, y_val = batch
        y_val_pred = self(X_val)
        val_loss = F.cross_entropy(y_val_pred, y_val)
        val_acc = accuracy(y_val_pred, y_val)
        return {'val_loss': val_loss.detach(), 'val_acc': val_acc}

    def val_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        batch_accs = [x['val_acc'] for x in outputs]

        epoch_loss = torch.stack(batch_losses.mean())
        epoch_acc = torch.stack(batch_accs.mean())

        return {'val_loss': epoch_loss, 'val_acc': epoch_acc}

    def epoch_end(self, epoch, result):
        print("Epoch #{}\tLoss: {:.2f}\tAccuracy: {:.2f}".format(epoch, result['val_loss'], result['val_acc']))


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class Cifar10CNNModel(ImageClassifiationBase):
    def __init__(self):
        self.arch = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),

            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),

            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),  # Output Dim: 256x4x4

            torch.nn.Flatten(),
            torch.nn.Linear(256 * 4 * 4, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 10)

        )

    def forward(self, X):
        return self.arch(X)


model = Cifar10CNNModel()
print(model)  # Model summary


def get_default_device():
    if torch.cuda.is_available():
        torch.device('cuda')
    else:
        torch.device('cpu')


device = get_default_device()


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(d) for d in data]
    else:
        return data.to(device, non_blocking=True)


class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)


def plot_losses(history):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs');


train_dataldr = DeviceDataLoader(train_dataldr, device)
val_dataldr = DeviceDataLoader(val_dataldr, device)


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    return history


model = to_device(Cifar10CnnModel(), device)

num_epochs = 10
opt_func = torch.optim.Adam
lr = 0.001

history = fit(num_epochs, lr, model, train_dataldr, val_dataldr, opt_func)
plot_losses(history)