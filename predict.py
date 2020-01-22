import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from config import Config
from dataset import CarDataset
import numpy as np

from model import CentResnet

PATH = Config.PATH
train = pd.read_csv(PATH + 'train.csv')
train_images_dir = PATH + 'train_images/{}.jpg'
df_train, df_dev = train_test_split(train, test_size=0.01, random_state=42)
dev_dataset = CarDataset(df_dev, train_images_dir)

device = Config.device

model = CentResnet(8).to(device)
model.load_state_dict(torch.load(Config.save_path))
img, mask, regr = dev_dataset[0]

plt.figure(figsize=(16, 16))
plt.title('Input image')
plt.imshow(np.rollaxis(img, 0, 3))
plt.show()

plt.figure(figsize=(16, 16))
plt.title('Ground truth mask')
plt.imshow(mask)
plt.show()

output = model(torch.tensor(img[None]).to(device))
logits = output[0, 0].data.cpu().numpy()

plt.figure(figsize=(16, 16))
plt.title('Model predictions')
plt.imshow(logits)
plt.show()

plt.figure(figsize=(16, 16))
plt.title('Model predictions thresholded')
plt.imshow(logits > 0)
plt.show()
