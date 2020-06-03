import torch
import torchvision
import jovian
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import random_split, DataLoader

batch_size = 128
learning_rate = 5e-7

input_size = 28 * 28
num_cls = 10

jovian.reset()
jovian.log_hyperparams(batch_size=batch_size, learning_rate=learning_rate)

mnist = MNIST(root='data/', train=True, transform=torchvision.transforms.ToTensor(), download=True)

train_ds, val_ds = random_split(mnist, [50000, 10000])
test_ds = MNIST(root='data/', train=False, transform=torchvision.transforms.ToTensor())

train_loader = DataLoader(train_ds, batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size * 2)
test_loader = DataLoader(test_ds, batch_size * 2)

# Sample image plot

some_image, some_label = train_ds[0]
plt.imshow(some_image[0], cmap='gray')
print('Label: ', some_label)


# Model

class MnistModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(input_size, num_cls)

    def forward(self, xb):
        xb = xb.reshape(-1, input_size)
        out = self.linear(xb)

        return out

    def training_step(self, batch):
        train_img, train_lbl = batch
        out = self(train_img)
        loss = F.cross_entropy(out, train_lbl)

        return loss

    def validation_step(self, batch):
        val_img, val_lbl = batch
        out = self(val_img)
        val_loss = F.cross_entropy(out, val_lbl)
        val_acc = accuracy(out, val_lbl)

        return {'val_loss': val_loss.detach(), 'val_acc': val_acc.detach()}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_acc = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_acc).mean()

        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch: #{}, val_loss: {}, val_acc: {}".format(epoch, result['val_loss'], result['val_acc']))


model = MnistModel()


def accuracy(outputs, lbls):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == lbls).item() / len(preds))


def fit_model(model, epochs, learning_rate, train_loader, val_loader, optim=torch.optim.SGD):
    history = []
    optimizer = optim(model.parameters(), learning_rate)
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
    out = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(out)



history = fit_model(model, 30, learning_rate, train_loader, val_loader)


val_acc = [result['val_acc'] for result in history]
plt.plot(val_acc, '-x')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy with epochs')

test_set_results = eval_model(model, test_loader)
print("Test accuracy: {}, Test loss: {}".format(test_set_results['val_acc'], test_set_results['val_loss']))

jovian.log_metrics(test_acc=test_set_results['val_acc'], test_loss=test_set_results['val_loss'])


def get_prediction(x, model):
    xb = x.unsquezze(0)
    yb = model(xb)
    _, preds  = torch.max(yb, dim=1)
    return preds[0].item()

some_random_digit, some_random_lbl = test_ds[123]
plt.imshow(some_random_digit[0], cmap='gray')
print('Prediction: {}, Target: {}'.format(get_prediction(some_random_digit, model), some_random_lbl) )

torch.save(model.state_dict(), 'mnist_logistic.pth')
jovian.commit(project='mnist-classification', environment=None, outputs=['mnist-logisitc.pth'])