import torch
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm

from config import Config
from dataset import CarDataset
from model import CentResnet
from utils import extract_coords, coords2str

PATH = Config.PATH
device = Config.device
predictions = []
test_images_dir = PATH + 'test_images/{}.jpg'
test = pd.read_csv(PATH + 'sample_submission.csv')
df_test = test
test_dataset = CarDataset(df_test, test_images_dir, False)
test_loader = DataLoader(dataset=test_dataset, batch_size=4, shuffle=False, num_workers=4)
model = CentResnet(8).to(device)
model.load_state_dict(torch.load(Config.save_path))
model.eval()
if __name__ == '__main__':

    for img, _, _ in tqdm(test_loader):
        with torch.no_grad():
            output = model(img.to(device))
        output = output.data.cpu().numpy()
        for out in output:
            coords = extract_coords(out)
            s = coords2str(coords)
            predictions.append(s)
    test = pd.read_csv(PATH + 'sample_submission.csv')
    test['PredictionString'] = predictions
    test.to_csv('submission.csv', index=False)
