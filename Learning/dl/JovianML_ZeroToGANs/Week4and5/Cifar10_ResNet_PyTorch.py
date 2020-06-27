"""
Following concepts are explored:

Data normalization
Data augmentation
Residual connections
Batch normalization
Learning rate scheduling
Weight Decay
Gradient clipping
"""

import os
import torch
import torchvision
import tarfile
from torch.utils.data import DataLoader
from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split

