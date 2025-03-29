import torch
import torch.nn as nn

# class RandomizedCheckLoss(nn.Module):
#     def __init__(self):
#         super(RandomizedCheckLoss, self).__init__()

#     def forward(self, output, target, tau):
#         """
#         output: 복원된 이미지 텐서, shape: [B, C, H, W]
#         target: 원본 이미지 텐서, shape: [B, C, H, W]
#         tau: 각 픽셀마다의 quantile 값 텐서, shape: [B, 1, H, W] 또는 broadcast 가능한 shape
#         """
#         error = target - output  # error shape: [B, C, H, W]
#         # 각 픽셀마다 quantile loss 계산:
#         # if error >= 0  --> loss = tau * error
#         # if error < 0   --> loss = (tau - 1) * error, which is equal to (1-tau)*abs(error)
#         loss = torch.where(error >= 0, tau * error, (tau - 1) * error)
#         return loss.mean()
    
class RandomizedCheckLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(RandomizedCheckLoss, self).__init__()
        self.reduction = reduction

    def forward(self, output, target, tau, reduction=None):
        """
        output: 복원된 이미지 텐서, shape: [B, C, H, W]
        target: 원본 이미지 텐서, shape: [B, C, H, W]
        tau: 각 픽셀마다의 quantile 값 텐서, shape: [B, 1, H, W] 또는 broadcast 가능한 shape
        reduction: 'none', 'mean', 또는 'sum'. 지정하지 않으면 초기 설정(self.reduction)을 사용.
        """
        if reduction is None:
            reduction = self.reduction

        error = target - output  # [B, C, H, W]
        loss = torch.where(error >= 0, tau * error, (tau - 1) * error)
        
        if reduction == 'none':
            return loss
        elif reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            raise ValueError(f"Invalid reduction type: {reduction}")
