import scipy.io as sio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import time
from model.defeat_detect import DefeatDetectModel
from myloss import MyLoss
import utils
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
lr = 1e-4
epochs = 10

# Data parameters
file_name = 'data/final_data_sim_{:04d}.mat'
file_num = 32
train_ratio = 0.35
scan_num = 32
td_num = 32
gridN = 512
freq_dim = 64


model = DefeatDetectModel(td_num, gridN, freq_dim).to(device)
model.apply(utils.weight_init)

optimizer = optim.Adam(model.parameters(), lr=lr)
# loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]).to(device))
loss_fn = MyLoss(lambda_pos=2, lambda_rad=1, lambda_unpair=5).to(device)

start_time = time.time()

datas = []

# Load data
print('Loading data...')
for file_idx in tqdm.tqdm(range(int(file_num * train_ratio))):
    datas.append(sio.loadmat(file_name.format(file_idx)))
print('Data loaded.')

for epoch in range(epochs):
    # Training
    loop = tqdm.tqdm(total = file_num * train_ratio * scan_num)
    loop.set_description(f'Epoch {epoch + 1}/{epochs}')

    loss_history = []

    model.train()

    for file_idx in range(int(file_num * train_ratio)):

        # Load data
        data = datas[file_idx]

        wave = torch.tensor(np.abs(data['wave']), dtype=torch.float32).to(device)
        phase = torch.tensor(data['phase'], dtype=torch.float32).to(device)
        label = torch.tensor(data['defeat'] / gridN).to(device)
        for i in range(len(label)):
            label[i][0] = 1 - label[i][0]

        # Go through each scan
        for scan_idx in range(scan_num):
            pred = model(wave[scan_idx], phase[scan_idx])
            loss = loss_fn(pred.squeeze(), label)
            loss_history.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            

            loop.set_postfix({
                'file': file_idx,
                'scan': scan_idx,
                'loss': '%.4f' % loss.item(),
                'time': time.time() - start_time
            })
            loop.update()

    loop.set_postfix({
        'file': file_idx,
        'scan': scan_idx,
        'loss': '%.4f' % np.mean(loss_history),
        'time': time.time() - start_time
    })
    loop.close()

torch.save(model.state_dict(), 'model.pth')
