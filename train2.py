from sklearn.model_selection import train_test_split
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from dataset import CarDataset
from model2 import CentResnet
import pandas as pd
import torch
import matplotlib

matplotlib.use('Agg')

SWITCH_LOSS_EPOCH = Config.SWITCH_LOSS_EPOCH
PATH = Config.PATH
train_images_dir = PATH + 'train_images/{}.jpg'

train = pd.read_csv(PATH + 'train.csv')

BATCH_SIZE = 2

df_train, df_dev = train_test_split(train, test_size=0.2, random_state=118)

# Create dataset objects
train_dataset = CarDataset(df_train, train_images_dir)
dev_dataset = CarDataset(df_dev, train_images_dir)

# Create data generators - they will produce batches
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
dev_loader = DataLoader(dataset=dev_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

n_epochs = 12  # 6
device = Config.device
model = CentResnet().to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.001)
# optimizer =  RAdam(model.parameters(), lr = 0.001)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=max(n_epochs, 10) * len(train_loader) // 3, gamma=0.1)

history = pd.DataFrame()


def _sigmoid(x):
    y = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
    return y


def focal_loss(pred, gt):
    pred = _sigmoid(pred)
    pos_inds = gt.eq(1).float()
    pos_inds = pos_inds.unsqueeze(1)
    # print(pos_inds.size())
    neg_inds = gt.lt(1).float().unsqueeze(1)

    neg_weights = torch.pow(1 - gt, 4).unsqueeze(1)

    loss = 0
    # print(neg_weights)
    pos_loss = torch.log(pred + 1e-7) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred + 1e-7) * torch.pow(pred, 2) * neg_weights * neg_inds

    # .float().sum()
    pos_loss = pos_loss.view(pred.size(0), -1).sum(-1)
    neg_loss = neg_loss.view(gt.size(0), -1).sum(-1)
    # neg_loss.sum(-1)
    num_pos = pos_inds.sum()
    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss)  # / num_pos
    num_pos = pos_inds.view(gt.size(0), -1).sum(-1)
    # print('loss',loss.size(),pos_loss.size(),loss.size(),'loss_sum',loss.sum(-1).mean(0),num_pos.size())
    return loss.mean(0)


def criterion(prediction, mask, regr, weight=0.4, size_average=True):
    # Binary mask loss
    pred_mask = torch.sigmoid(prediction[:, 0])
    mask_loss = mask * torch.log(pred_mask + 1e-12) + (1 - mask) * torch.log(1 - pred_mask + 1e-12)
    mask_loss = -mask_loss.mean(0).sum()
    # Regression L1 loss
    pred_regr = prediction[:, 1:]
    regr_loss = (torch.abs(pred_regr - regr).sum(1) * mask).sum(1).sum(1) / mask.sum(1).sum(1)
    regr_loss = regr_loss.mean(0)

    # Sum
    loss = weight * mask_loss + (1 - weight) * regr_loss
    if not size_average:
        loss *= prediction.shape[0]
    return loss, mask_loss, regr_loss


def train(epoch, history=None):
    model.train()
    t = tqdm(train_loader)
    for batch_idx, (img_batch, mask_batch, regr_batch) in enumerate(t):
        img_batch = img_batch.to(device)
        mask_batch = mask_batch.to(device)
        regr_batch = regr_batch.to(device)

        optimizer.zero_grad()
        output = model(img_batch)
        if epoch < SWITCH_LOSS_EPOCH:
            loss, mask_loss, regr_loss = criterion(output, mask_batch, regr_batch, 1)
        else:
            loss, mask_loss, regr_loss = criterion(output, mask_batch, regr_batch, 0.5)

        t.set_description(f'train_loss (l={loss:.3f})(m={mask_loss:.2f}) (r={regr_loss:.4f}')

        if history is not None:
            history.loc[epoch + batch_idx / len(train_loader), 'train_loss'] = loss.data.cpu().numpy()

        loss.backward()

        optimizer.step()
        exp_lr_scheduler.step()

    print('Train Epoch: {} \tLR: {:.6f}\tLoss: {:.6f}\tMaskLoss: {:.6f}\tRegLoss: {:.6f}'.format(
        epoch,
        optimizer.state_dict()['param_groups'][0]['lr'],
        loss.data,
        mask_loss.data,
        regr_loss.data))


def evaluate(epoch, history=None):
    model.eval()
    loss = 0
    valid_loss = 0
    valid_mask_loss = 0
    valid_regr_loss = 0
    with torch.no_grad():
        for img_batch, mask_batch, regr_batch in dev_loader:
            img_batch = img_batch.to(device)
            mask_batch = mask_batch.to(device)
            regr_batch = regr_batch.to(device)

            output = model(img_batch)

            if epoch < SWITCH_LOSS_EPOCH:
                loss, mask_loss, regr_loss = criterion(output, mask_batch, regr_batch, 1, size_average=False)
                valid_loss += loss.data
                valid_mask_loss += mask_loss.data
                valid_regr_loss += regr_loss.data
            else:
                loss, mask_loss, regr_loss = criterion(output, mask_batch, regr_batch, 0.5, size_average=False)
                valid_loss += loss.data
                valid_mask_loss += mask_loss.data
                valid_regr_loss += regr_loss.data

    valid_loss /= len(dev_loader.dataset)
    valid_mask_loss /= len(dev_loader.dataset)
    valid_regr_loss /= len(dev_loader.dataset)

    if history is not None:
        history.loc[epoch, 'dev_loss'] = valid_loss.cpu().numpy()
        history.loc[epoch, 'mask_loss'] = valid_mask_loss.cpu().numpy()
        history.loc[epoch, 'regr_loss'] = valid_regr_loss.cpu().numpy()

    print('Dev loss: {:.4f}'.format(valid_loss))


import gc
import matplotlib.pyplot as plt

history = pd.DataFrame()
if __name__ == '__main__':

    for epoch in range(n_epochs):
        torch.cuda.empty_cache()
        gc.collect()
        train(epoch, history)
        evaluate(epoch, history)
        torch.save(model.state_dict(), './models/model_res_%d.pth' % epoch)

    series1 = history.dropna()['mask_loss']
    plt.plot(series1.index, series1, label='mask loss')
    series2 = history.dropna()['regr_loss']
    plt.plot(series2.index, 30 * series2, label='regr loss')
    series3 = history.dropna()['dev_loss']
    plt.plot(series3.index, series3, label='dev loss')
    plt.legend()
    plt.savefig("loss.jpg")
