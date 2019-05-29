import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


data = pd.read_csv("data/training.csv")

test = 2

# class BixiDataset(Dataset):
