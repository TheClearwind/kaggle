import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm

from config import Config
from dataset import CarDataset
from model2 import CentResnet
from utils import extract_coords, coords2str, imread, preprocess_image

PATH = Config.PATH
device = Config.device
predictions = []
train_images_dir = PATH + 'train_images/{}.jpg'

train = pd.read_csv(PATH + 'train.csv')
df_train, df_dev = train_test_split(train, test_size=0.2, random_state=118)
model = CentResnet().to(device)
# model.load_state_dict(torch.load(Config.save_path))
model.eval()
if __name__ == '__main__':
    import numpy as np

    reg_cout = 0
    pre_cout = 0
    for idx in range(len(df_dev)):
        data = df_dev.iloc[idx]
        imageid, prestr = data["ImageId"], data["PredictionString"]
        path = PATH + 'train_images/{}.jpg'.format(imageid)
        img0 = imread(path, True, True)
        img = preprocess_image(img0)
        img = np.rollaxis(img, 2, 0)
        reg_cout += len(prestr.split()) // 7
        img = img[np.newaxis, :]
        img = torch.tensor(img).to(device)
        with torch.no_grad():
            out = model(img)
            coords = extract_coords(out[0].cpu().numpy(), threshold=0)
            pre_cout += len(coords)
        print("reg cout:{} pre count:{}".format(reg_cout, pre_cout))
