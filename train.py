import scipy.io as sio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import time
from model.defeat_detect import DefeatDetectModel
from dataset import DefectDataset, my_collate
from torch.utils.data import DataLoader
from myloss import MyLoss
import utils
import os

device = torch.device('cuda')

# Hyperparameters
lr = 1e-4
epochs = 5

# Data parameters
file_name = 'data/data_{}.mat'
file_num = len(os.listdir('data'))
train_ratio = 0.9
scan_num = 4
gridN = 512

# Model
use_pretrained = True
pretrain_model_name = 'model_v1.pth'

model = DefeatDetectModel(gridN).to(device)
if use_pretrained:
    model.load_state_dict(torch.load(pretrain_model_name, weights_only=True))
else:
    model.apply(utils.weight_init)

optimizer = optim.Adam(model.parameters(), lr=lr)
loss_fn = MyLoss(lambda_coord=2, lambda_r=5, lambda_noobj=0.05).to(device)

start_time = time.time()

train_data = DefectDataset(file_name, file_num * train_ratio)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True, collate_fn=my_collate)

for epoch in range(epochs):
    # Training
    loop = tqdm.tqdm(train_loader, leave=True)
    loop.set_description(f'Epoch {epoch + 1}/{epochs}')

    loss_history = []

    model.train()

    for phase, label in train_loader:

        phase, label = phase.to(device), label

        pred = model(phase)
        loss = loss_fn(pred, label)
        loss_history.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loop.set_postfix({
            'loss': '%.4f' % loss.item(),
            'time': time.time() - start_time
        })
        loop.update()

    loop.set_postfix({
        'loss': '%.4f' % np.mean(loss_history),
        'time': time.time() - start_time
    })
    loop.close()

torch.save(model.state_dict(), 'model.pth')

# Draw loss history
import matplotlib.pyplot as plt
plt.plot(loss_fn.xy_loss_hist, label='xy_loss')
plt.plot(loss_fn.r_loss_hist, label='r_loss')
plt.plot(loss_fn.coord_loss_hist, label='coord_loss')
plt.plot(loss_fn.condobj_loss_hist, label='condobj_loss')
plt.plot(loss_fn.condnoobj_loss_hist, label='condnoobj_loss')
plt.plot(loss_fn.loss_hist, label='loss')
plt.legend()
plt.show()

