import torch
import torchvision
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import matplotlib.pyplot as plt

dataset = torchvision.datasets.MNIST(root='data/', download=True, transform=torchvision.transforms.ToTensor())

val_size = 10000
train_size = len(dataset) - val_size

train_ds, val_ds = random_split([train_size, val_size])

batch_size = 128

train_dataldr = DataLoader(dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_dataldr = DataLoader(dataset, batch_size, shuffle=False, num_workers=4, pin_memory=True)

input_size = 28*28
num_cls = 10
hidden_size = 64

def accuracy(outputs, targets):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == targets).item() / len(preds))

class MnistModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.linear1 = torch.nn.Linear(input_size, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, xb):
        xb_flattened = xb.view(xb.size(0), -1)
        out = self.linear1(xb_flattened)
        out = F.relu(out)
        out = self.linear2(out)

        return out

    def train_step(self, batch):
        inputs, targets = batch
        out = self(inputs)
        loss = F.cross_entropy(out, targets)
        acc = accuracy(out, targets)

        return loss.detach()

    def val_step(self, batch):
        val_inputs, val_targets = batch
        out = self(val_inputs)
        val_loss = F.cross_entropy(out, val_targets)
        val_acc = accuracy(out, val_targets)

        return val_loss.detach()


model = MnistModel(input_size, hidden_size, num_cls)


def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


device = get_default_device()


def to_device(data, device):
    if isinstance(data, [list, tuple]):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLdr():
    def __init__(self, dataldr, device):
        self.dataldr = dataldr
        self.device = device

    def __iter__(self):
        for batch in self.dataldr:
            yield.to_device(batch, self.device)


def eval_model(model, val_dataldr):
    val_losses = [model.val_step(b) for b in val_dataldr]
    return val_losses


def fit(epochs, lr, model, train_dataldr, val_dataldr, optim=torch.optim.SGD):

    optimizer = optim(model.parameters(), lr)
    history = []

    for epoch in epochs:
        for batch in train_dataldr:
            loss = model.train_step(batch)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        result = eval_model(model, val_dataldr)
        print("Epoch #{} Loss: {} Accuracy: {}".format(epoch, result['val_loss'], result['val_acc']))

        history.append(result)
    return history

to_device(model, device)

history = fit(epochs=100, lr=1e-7, model, train_dataldr, val_dataldr)

losses = [x['val_loss'] for x in history]
plt.plot(losses, '-x')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Loss vs. #epochs');

acc = [x['val_acc'] for x in history]
plt.plot(acc, '-x')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Acc vs #epochs')