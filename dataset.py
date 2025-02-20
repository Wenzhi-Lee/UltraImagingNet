import torch.utils.data as data
import torch
import scipy.io as sio
import numpy as np
import tqdm

def label_gen(raw_label):
    label = []
    gridN = 512
    boxNum = 8
    boxLen = gridN / boxNum
    for defect in raw_label:
        x, y, r = gridN - defect[0], defect[1], defect[2]
        boxRow = int(x // boxLen)
        boxCol = int(y // boxLen)
        label_temp = np.array([x - boxRow * boxLen, y - boxCol * boxLen, r])
        label_temp /= boxLen
        label.append([boxRow, boxCol, label_temp[0], label_temp[1], label_temp[2]])
    return label

def my_collate(data_batch):
    o_data, o_label = [], []
    for data, label in data_batch:
        o_data.append(data)
        o_label.append(label)
    
    return torch.tensor(np.array(o_data)), o_label

class DefectDataset(data.Dataset):
    
    def __init__(self, file_name, file_num, start_idx=0):
        self.datas = []
        self.label = []
        
        print('Loading data...')
        file_num = int(file_num)
        start_idx = int(start_idx)
        for file_idx in tqdm.tqdm(range(start_idx, start_idx + file_num)):
            data = sio.loadmat(file_name.format(file_idx))
            phase = data['phase']
            # phase = np.mean(phase, axis=1)
            phase = (phase - np.min(phase)) / (np.max(phase) - np.min(phase))
            self.datas.extend(phase)
            self.label.extend([label_gen(data['defeat'])] * len(phase))
        print('Data loaded.')
    
    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, idx):
        return self.datas[idx], self.label[idx]

