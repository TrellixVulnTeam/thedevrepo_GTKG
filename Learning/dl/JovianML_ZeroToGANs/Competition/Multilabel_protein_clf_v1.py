# Imports

import os
import pandas as pd
import torch
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import torch.nn.functional as F

# Static paths
BASE_DIR = '../input/jovian-pytorch-z2g'
DATA_DIR = os.path.join(BASE_DIR, 'Human protein atlas')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
TRAIN_CSV = os.path.join(DATA_DIR, 'train.csv')
SUBMISSION_CSV = os.path.join(BASE_DIR, 'submission.csv')

# Data

labels = {
    0: 'Mitochondria',
    1: 'Nuclear bodies',
    2: 'Nucleoli',
    3: 'Golgi apparatus',
    4: 'Nucleoplasm',
    5: 'Nucleoli fibrillar center',
    6: 'Cytosol',
    7: 'Plasma membrane',
    8: 'Centrosome',
    9: 'Nuclear speckles'
}
num_cls = len(labels)
val_pc = 0.1
batch_size = 256

# Helpers & Utils

"""
encode_label

As it's a multilabel problem, the one-hot encoding is applied to the label.
The label here is a string of class indexes. This is split and the respective indexes of the OH vector is flagged.
"""


def encode_label(label):
    target_vector = torch.zeros(num_cls)
    for lbl in str(label).split(' '):
        target_vector[int(lbl)] = 1

    return target_vector


"""
decode_label

Here, the multi-label vector is decoded to it's corresponding class(es).
"""


def decode_label(target_vec, lbls, threshold=0.5, as_text=False):
    result = []
    for i, v in enumerate(target_vec):
        if v >= threshold:
            if as_text:
                result.append(lbls[i])
            else:
                result.append(str(i))

    return ' '.join(result)


"""
show_sample

Plot the image and print the label(s).
If `invert=True`, plot the image in contrast mode (suits dark images).
"""


def show_sample(img, lbl, invert=False):
    if invert:
        plt.imshow(1 - img.permute((1, 2, 0)))
    else:
        plt.imshow(img.permute((1, 2, 0)))

    print(decode_label(lbl, labels))


"""
get_default_device()

Returns the default device available.
"""


def get_default_device():
    if torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


"""
F1_score

Return the F1-score for the prediction.
"""


def f1_score(pred, lbl, threshold=0.5, beta=1, tol=1e-12):
    prob = pred > threshold
    lbl = lbl > threshold

    TP = (prob & lbl).sum(1).float()
    TN = ((~prob) & (~lbl)).sum(1).float()
    FP = ((~prob) & lbl).sum(1).float()
    FN = (prob & (~lbl)).sum(1).float()

    precision = torch.mean(TP / (TP + FP + tol))
    recall = torch.mean(TP / (TP + FN + tol))
    f1 = ((1 + beta ** 2) * precision * recall) / (beta ** 2 * precision + recall + tol)
    return f1.mean(0)


"""
predict_single

Predict the class and plot a single input image.
"""


def predict_single(image):
    xb = image.unsqueeze(0)
    xb = to_device(xb, device)
    preds = model(xb)
    prediction = preds[0]
    print("Prediction: ", prediction)
    show_sample(image, prediction)


"""
to_device

Assign data to device appropriately
"""


def to_device(data, dvc):
    if isinstance(data, (list, tuple)):
        return [to_device(ele, dvc) for ele in data]
    else:
        return data.to(dvc, non_blocking=True)


# Dataset / Dataloader creation

class HumanProteinDataset(Dataset):
    def __init__(self, csv_file, data_dir, tsfm=None):
        self.df = pd.read_csv(csv_file)
        self.data_dir = data_dir
        self.transform = tsfm

    def __len__(self):
        return len(self.df)

    """
    __getitem__

    Reads the image based on `item_idx` from csv.
    Encodes the corresponding label as One-Hot vector.

    Returns <img, label>
    """

    def __getitem__(self, item_idx):
        obs = self.df.loc[item_idx]
        img_id, img_lbl = obs['Image'], obs['Label']
        img_file = os.path.join(self.data_dir, str(img_id) + '.png')

        img = Image.open(img_file, 'r')

        if self.transform:
            img = self.transform(img)

        return img, encode_label(img_lbl)


transform = torchvision.transforms.Compose([torchvision.transforms.Resize(128), torchvision.transforms.ToTensor()])
dataset = HumanProteinDataset(TRAIN_CSV, TRAIN_DIR, tsfm=transform)
test_dataset = HumanProteinDataset(SUBMISSION_CSV, TEST_DIR, tsfm=transform)

show_sample(*dataset[0], invert=False)
show_sample(*dataset[0])

val_size = int(val_pc * len(dataset))
train_size = len(dataset) - val_size

train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
val_dl = DataLoader(train_ds, batch_size=batch_size * 2, shuffle=True, num_workers=2, pin_memory=True)
test_dl = DataLoader(test_dataset, batch_size, num_workers=2, pin_memory=True)


class DeviceDataLoader():
    def __init__(self, dl, dvc):
        self.dl = dl
        self.device = dvc

    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)


device = get_default_device()

train_dl = DeviceDataLoader(train_dl, device)
val_dl = DeviceDataLoader(val_dl, device)
test_dl = DeviceDataLoader(test_dl, device)


# Model

class MultiLabelImageClassificationBase(torch.nn.Module):

    def train_step(self, batch):
        X_train, y_train = batch
        y_pred = self(X_train)
        loss = F.binary_cross_entropy(y_pred, y_train)

        return loss

    def val_step(self, batch):
        X_val, y_val = batch
        y_val_pred = self(X_val)
        val_loss = F.binary_cross_entropy(y_val_pred, y_val)
        val_score = f1_score(y_val_pred, y_val)

        return {'val_loss': val_loss, 'val_score': val_score}

    def val_epoch_end(self, outputs):
        batch_losses = [batch['val_loss'] for batch in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_score = [batch['val_score'] for batch in outputs]
        epoch_score = torch.stack(batch_score).mean()

        return {'val_loss': epoch_loss.item(), 'val_score': epoch_score.item()}

    def epoch_end(self, epoch, results):
        print("Epoch #{}\tTrain Loss: {:.2f}\tVal Loss: {:.2f}\tScore: {:.2f}".format(epoch, results['train_loss'],
                                                                                      results['val_loss'],
                                                                                      results['val_score']))


class ProteinCnnModel(MultiLabelImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),

            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),

            torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),

            torch.nn.Flatten(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, num_cls),

            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


class ProteinResNetModel(MultiLabelImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = torchvision.models.resnet34(pretrained=True)
        last_layer = self.network.fc.in_features
        self.network.fc = torch.nn.Linear(last_layer, num_cls)

    def forward(self, x):
        return torch.sigmoid(self.network(x))


model = ProteinResNetModel()

print(model)
to_device(model, device);


# Training
@torch.no_grad()
def eval_model(model, val_dataldr):
    model.eval()
    val_outputs = [model.val_step(batch) for batch in val_dataldr]
    return model.val_epoch_end(val_outputs)


def fit(epochs, lr, model, train_dataldr, val_dataldr, optim=torch.optim.Adam):
    torch.cuda.empty_cache()

    history = []
    optimizer = optim(model.parameters(), lr)

    for epoch in range(epochs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        model.train()
        train_losses = []

        for batch in train_dataldr:
            loss = model.train_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        result = eval_model(model, val_dataldr)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)

        history.append(result)
        end.record()

        torch.cuda.synchronize()

        print(start.elapsed_time(end))

    return history


# Prediction & Submission
num_epochs = 10
learning_rate = 5e-3

history = fit(epochs=num_epochs, lr=learning_rate, model=model, train_dataldr=train_dl, val_dataldr=val_dl)

weights_fname = 'protein-resnet.pth'
torch.save(model.state_dict(), weights_fname)


@torch.no_grad()
def predict_dl(dl, model):
    torch.cuda.empty_cache()

    batch_probs = []
    for xb, _ in dl:
        probs = model(xb)
        batch_probs.append(probs.cpu().detach())
    batch_probs = torch.cat(batch_probs)
    return [decode_label(x, labels) for x in batch_probs]


test_preds = predict_dl(test_dl, model)

submission_df = pd.read_csv(SUBMISSION_CSV)
submission_df.Label = test_preds

print(submission_df.head())
sub_fname = 'resnet_submission.csv'
submission_df.to_csv(sub_fname, index=False)