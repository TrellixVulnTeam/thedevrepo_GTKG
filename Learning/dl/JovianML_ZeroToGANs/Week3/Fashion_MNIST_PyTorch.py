import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.datasets import FashionMNIST
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader

fmnist = FashionMNIST(root='data/', download=True, transform=torchvision.transforms.ToTensor)

val_size=10000
train_size=len(fmnist) - val_size

train_ds, val_ds = random_split(fmnist, [train_size, val_size])

input_size = 28*28
hidden_size = 64
num_cls = 10
batch_size = 64

train_dataldr = DataLoader(fmnist, batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_dataldr = DataLoader(fmnist, batch_size, shuffle=True, num_workers=4, pin_memory=True)

def accuracy(outputs, targets):
    _, pred = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(pred == targets).item() / len(pred))

class FMnistModel(torch.nn.Module):
    def __init__(self, in_size, hidden_units, out_size):
        self.linear1 = torch.nn.Linear(in_size, hidden_units)
        self.linear2 = torch.nn.Linear(hidden_units, out_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)

        out = self.linear1(x)
        out = F.relu(out)

        out = self.linear2(out)
        return out

    def training_step(self, batch):
        X_train, y_train = batch
        y_pred = self(X_train)
        loss = F.cross_entropy(y_pred, y_train)
        return loss.detach()

    def val_step(self, batch):
        X_val, y_val = batch
        y_pred = self(X_val)
        val_loss = F.cross_entropy(y_pred, y_val)

        val_acc = accuracy(y_pred, y_val)

        return {'val_loss':val_loss.detach(), 'val_acc':val_acc.item()}


model = FMnistModel(input_size, hidden_size, num_cls)

def eval_model(model, val_dataldr):
    val_losses = [model.val_step(batch) for batch in val_dataldr]
    return val_losses

def fit(epochs, lr=1e-7, model, training_dataldr, val_dataldr, optim=torch.optim.SGD):
    history = []
    optimizer = optim(model.parameters(), lr)

    for epoch in epochs:
        for batch in training_dataldr:
            training_loss = model.training_step(batch)
            training_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        val_results = eval_model(model, val_dataldr)
        print("Validation results:\t Loss: {}\tAcc: {}".format(val_results['val_loss'], val_results['val_acc']))

        history.append(val_results)

    return history

# PENDING --  device assignment and run !
def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLdr():
    def __init__(self, dataldr, device):
        self.dataldr = dataldr
        self.device  = device

    def __iter__(self):
        for batch in self.dataldr:
            yield to_device(batch, self.device)

    def __len__(self):
        return len(self.dataldr)


device = get_default_device()

train_dataldr = DeviceDataLdr(train_dataldr, device)
val_dataldr = DeviceDataLdr(val_dataldr, device)

to_device(model, device)

history += fit(epochs=100, lr=1e-7, model, train_dataldr, val_dataldr)

losses = [x['val_loss'] for x in history]
plt.plot(losses, '-x')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Loss vs. No. of epochs');

accs = [x['val_acc'] for x in history]
plt.plot(accs, '-x')
plt.xlabel('epoch')
plt.ylabel('acc')
plt.title('Accuracy vs. No. of epochs');



def predict_image(img, model):
    xb = to_device(img.unsqueeze(0), device)
    yb = model(xb)
    _, preds  = torch.max(yb, dim=1)
    return preds[0].item()


img, label = fmnist[0]
plt.imshow(img[0], cmap='gray')
print('Label:', fmnist.classes[label], ', Predicted:', fmnist.classes[predict_image(img, model)])