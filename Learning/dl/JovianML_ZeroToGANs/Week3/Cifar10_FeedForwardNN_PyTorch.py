import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import CIFAR10

dataset = CIFAR10(root='data/', train=True, transform=torchvision.transforms.ToTensor())
test_dataset = CIFAR10(root='data/', train=False, transform=torchvision.transforms.ToTensor())

val_size = 5000
train_size = len(dataset) - val_size
num_cls = len(dataset.classes)
batch_size = 128

train_data, val_data = random_split(dataset, [train_size, val_size])

train_dataldr = DataLoader(train_data, batch_size, shuffle=True, num_workers=3, pin_memory=True)

# Batch size can be larger for val/test because there's no backprop involved here.
# No gradient calculation/storing => GPU can afford more memory for accommodating larger batches.

val_dataldr = DataLoader(val_data, batch_size * 2, shuffle=True, num_workers=3, pin_memory=True)
test_dataldr = DataLoader(test_dataset, batch_size * 2, shuffle=True, num_workers=3, pin_memory=True)


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class ImageClassificationBase(torch.nn.Module):

    def train_step(self, batch):
        X, y = batch
        y_pred = self(X)
        loss = F.cross_entropy(y_pred, y)
        return loss

    def val_step(self, batch):
        X_val, y_val = batch
        y_val_pred = self(X_val)
        val_loss = F.cross_entropy(y_val_pred, y_val)
        val_acc = accuracy(y_val_pred, y_val)
        return {'val_loss': val_loss.detach(), 'val_acc': val_acc}

    def val_epoch_end(self, outputs):
        batch_losses = [out['val_loss'] for out in outputs]
        batch_accs = [out['val_acc'] for out in outputs]

        epoch_loss = torch.stack(batch_losses.mean())
        epoch_acc = torch.stack(batch_accs.mean())

        return {'epoch_val_loss': epoch_loss.item(), 'epoch_val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch #{}\tLoss:{:.2f}\tAcc{:.2f}".format(epoch, result['epoch_val_loss'], result['epoch_val_acc']))


input_units = 3 * 32 * 32
output_units = 10
hidden_units = [128, 64, 32]


class Cifar10Classifier(ImageClassificationBase):

    def __init__(self):
        self.linear1 = torch.nn.Linear(in_features=input_units, out_features=hidden_units[0])
        self.linear2 = torch.nn.Linear(hidden_units[0], hidden_units[1])
        self.linear3 = torch.nn.Linear(hidden_units[1], hidden_units[2])
        self.linear4 = torch.nn.Linear(hidden_units[2], output_units)

    def forward(self, X_batch):
        X = X_batch.view(X_batch.size(0), -1)  # Flattened input

        h = self.linear1(X)
        h = F.relu(h)
        h = self.linear2(h)
        h = F.relu(h)
        h = self.linear3(h)
        h = F.relu(h)

        o = self.linear4(h)


def evaluate(model, val_ldr):
    val_results = [model.val_step(batch) for batch in val_ldr]
    return model.validation_epoch_end(val_results)


def fit(epochs, lr, model, train_ldr, val_ldr, optim=torch.optim.SGD):
    history = []
    optimizer = optim(model.parameters(), lr)
    for epoch in epochs:
        for batch in train_ldr:
            loss = model.train_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        result = evaluate(model, val_ldr)
        model.epoch_end(epoch, result)
        history.append(result)

    return history


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


def plot_loss_history(history):
    losses = [x['val_loss'] for x in history]
    plt.plot(losses, '-x')
    plt.xlabel('epoch')
    plt.ylabel('val_loss')
    plt.title('Epoch-wise Loss')


def plot_acc_history(history):
    accs = [x['val_acc'] for x in history]
    plt.plot(accs, '-x')
    plt.xlabel('epoch')
    plt.ylabel('val_acc')
    plt.title('Epoch-wise Accuracy')


train_dataldr = DeviceDataLoader(train_dataldr, device)
val_dataldr = DeviceDataLoader(val_dataldr, device)
test_dataldr = DeviceDataLoader(test_dataldr, device)

model = Cifar10Classifier()

model = to_device(model, device)

history = fit(epochs=100, model=model, train_ldr=train_dataldr, val_ldr=val_dataldr)

evaluate(model, test_dataldr)
