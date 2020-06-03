# !pip install jovian --upgrade --quieta

import torch
import torchvision
import jovian
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.datasets.utils import download_url
from torch.utils.data import DataLoader, TensorDataset, random_split

# Hyperparams
batch_size = 64
learning_rate = 5e-7

# Constants
DATASET_URL = 'Https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv'
DATA_FILENAME = 'BostonHousing.csv'
TARGET_COLUMN = 'medv'

input_size = 13
output_size = OUTPUT_SIZE = 1

download_url(DATASET_URL, 'data/.')
data = pd.read_csv('data/BostonHousing.csv')

print(data.head())

inputs = data.drop(['medv'], axis=1).values
targets = data['medv'].values

print("Input shape: {}, Target shape: {}".format(inputs.shape, targets.shape))

dataset = TensorDataset(torch.tensor(inputs, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32))
train_ds, val_ds = random_split(dataset, [(inputs.shape[0] - 100), 100])

train_loader = DataLoader(train_ds, batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size * 2)


# Model

class HousingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(input_size, output_size)

    def forward(self, xb):
        out = self.linear(xb)

        return out

    def training_step(self, batch):
        xs, ys = batch
        out = self(xs)
        loss = F.mse_loss(out, ys)

        return loss

    def validation_step(self, batch):
        val_inputs, val_targets = batch
        out = self(val_targets)
        val_loss = F.mse_loss(val_targets, out)

        return {'val_loss': val_loss.detach()}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()

        return {'epoch_val_loss': epoch_loss.item()}

    def epoch_end(self, epoch, result):
        print("Epoch #{}, Val loss: {:.4f}".format(epoch, result['epoch_val_loss']))


model = HousingModel()


# Training

def fit_model(epochs, lr, model, train_loader, val_loader, optim=torch.optim.SGD):
    history = []
    optimizer = optim(model.parameters(), lr)

    for epoch in range(epochs):
        # training
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # validation
        result = eval_model(model, val_loader)
        model.epoch_end(epoch, result)
        history.append(result)

    return history


def eval_model(model, val_loader):
    outs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outs)


history = fit_model(30, learning_rate, model, train_loader, val_loader)

val_losses = [result['epoch_val_loss'] for result in history]
plt.plot(val_losses, '-x')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss by Epoch')


def get_prediction(x, model):
    xb = x.unsquezze(0)
    return model(xb).item()


x,y = val_ds[10]
y_pred = get_prediction(x, model)

print("Prediction: {}, Target: {}".format(y_pred, y))

torch.save(model.state_dict(), 'artifacts/housing_linear.pth')

jovian.commit(project='housing-price-prediction', environment=None, outputs=['housing_linear.pth'])

