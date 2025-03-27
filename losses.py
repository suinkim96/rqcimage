import torch
import torch.nn as nn

class RandomizedCheckLoss(nn.Module):
    def __init__(self):
        super(RandomizedCheckLoss, self).__init__()

    def forward(self, output, target, tau):
        """
        output: 복원된 이미지 텐서, shape: [B, C, H, W]
        target: 원본 이미지 텐서, shape: [B, C, H, W]
        tau: 각 픽셀마다의 quantile 값 텐서, shape: [B, 1, H, W] 또는 broadcast 가능한 shape
        """
        error = target - output  # error shape: [B, C, H, W]
        # 각 픽셀마다 quantile loss 계산:
        # if error >= 0  --> loss = tau * error
        # if error < 0   --> loss = (tau - 1) * error, which is equal to (1-tau)*abs(error)
        loss = torch.where(error >= 0, tau * error, (tau - 1) * error)
        return loss.mean()
    
