import torch
import torch.nn as nn

class MyLoss(nn.Module):
    def __init__(self, lambda_pos = 0.5, lambda_rad = 0.5, lambda_unpair = 0.5):
        super(MyLoss, self).__init__()
        self.lambda_pos = lambda_pos
        self.lambda_rad = lambda_rad
        self.lambda_unpair = lambda_unpair
    
    def forward(self, pred, label):
        # pred: (20, 4)
        # label: (num, 3)
        loss_pair, loss_unpair = 0, 0

        pair_mask = torch.zeros(20).to(label.device)

        # Pairing loss
        for y in label:
            min_cost = float('inf')
            paired_idx = 0

            for i, y_hat in enumerate(pred):
                
                cost = self.pair_cost(y, y_hat)

                if cost < min_cost:
                    min_cost = cost
                    paired_idx = i
            
            loss_pair += self.pair_loss(y, pred[paired_idx])
            pair_mask[paired_idx] = 1
        
        # Unpaired loss
        for i, mask in enumerate(pair_mask):
            if mask == 0:
                loss_unpair += (self.prob_loss(pred[i][3], 0) + self.lambda_rad * pred[i][2]) * self.lambda_unpair

        return loss_pair / label.shape[0] + loss_unpair / (20 - label.shape[0])
        
    
    def pair_cost(self, y, y_hat):
        # y: (3,)
        # y_hat: (4,)
        
        dist = torch.sqrt((y[0] - y_hat[0]) ** 2 + (y[1] - y_hat[1]) ** 2)

        radius_diff = torch.abs(y[2] - y_hat[2])

        return self.lambda_pos * dist + self.lambda_rad * radius_diff

    def pair_loss(self, y, y_hat):
        # y: (3,)
        # y_hat: (4,)
        
        dist = torch.sqrt((y[0] - y_hat[0]) ** 2 + (y[1] - y_hat[1]) ** 2)

        radius_diff = torch.abs(y[2] - y_hat[2])

        prob = self.prob_loss(y_hat[3], 1)

        loss = self.lambda_pos * dist + self.lambda_rad * radius_diff + prob

        return loss

    def prob_loss(self, y_hat, y):
        return -y * torch.log(y_hat + 1e-6) - (1 - y) * torch.log(1 - y_hat + 1e-6)
