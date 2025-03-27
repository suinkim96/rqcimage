import random
import os
import numpy as np
import math
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageFilter


# 1. 전처리: 랜덤 블러 적용 transform (input에만 적용)
class RandomBlur(object):
    def __init__(self, probability=0.7, radius_range=(1.0, 3.0)):
        self.probability = probability
        self.radius_range = radius_range

    def __call__(self, img):
        if random.random() < self.probability:
            radius = random.uniform(*self.radius_range)
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))
        return img

class GaussianNoise(object):
    def __init__(self, sigma_type='constant', sigma_range=15):
        """
        Args:
            sigma_type (str): 노이즈 강도를 결정하는 방식. 'constant', 'random', 'choice' 중 하나.
            sigma_range: 
                - 'constant'일 경우 고정 값 (예: 15),
                - 'random'일 경우 (min, max) 튜플 (예: (10, 20)),
                - 'choice'일 경우 선택 가능한 값들의 리스트 (예: [10, 15, 20]).
        """
        self.sigma_type = sigma_type
        self.sigma_range = sigma_range

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): 입력 이미지.
        Returns:
            PIL.Image: Gaussian noise가 항상 추가된 이미지.
        """
        # sigma 값 결정 (확률 조건 없이 항상 적용)
        if self.sigma_type == 'constant':
            sigma_value = self.sigma_range
        elif self.sigma_type == 'random':
            sigma_value = random.uniform(self.sigma_range[0], self.sigma_range[1])
        elif self.sigma_type == 'choice':
            sigma_value = random.choice(self.sigma_range)
        else:
            raise ValueError(f'Unsupported sigma_type: {self.sigma_type}')

        # 이미지를 numpy float32 배열로 변환 (범위 0~1)
        img_array = np.array(img).astype(np.float32) / 255.0

        # 노이즈 생성 (평균 0, 표준편차 = sigma_value/255)
        noise = np.random.randn(*img_array.shape).astype(np.float32) * (sigma_value / 255.0)
        img_noisy = img_array + noise

        # 값 범위 클리핑
        img_noisy = np.clip(img_noisy, 0, 1)

        # 다시 0~255 범위로 복원 후 uint8로 변환하여 PIL 이미지 생성
        img_noisy = (img_noisy * 255.0).astype(np.uint8)
        return Image.fromarray(img_noisy)
    
# 패치 추출 함수 (예시 구현; 실제 구현은 다를 수 있음)
def extract_patches(tensor_img, patch_size=48, overlap=9, p_max=60):
    """
    tensor_img: Tensor (C x H x W)
    patch_size: 추출할 패치 크기
    overlap: 패치 간 겹침 크기
    p_max: 이미지가 충분히 큰지 판별하는 임계값
    """
    C, H, W = tensor_img.shape
    patches = []
    # 이미지가 충분히 큰 경우에만 패치로 분할 (첫 번째 코드와 동일한 조건)
    if H > p_max and W > p_max:
        # H와 W에 대해 시작 좌표 계산 (슬라이딩 윈도우: patch_size-overlap 간격)
        h_coords = list(np.arange(0, H - patch_size, patch_size - overlap, dtype=int))
        w_coords = list(np.arange(0, W - patch_size, patch_size - overlap, dtype=int))
        # 마지막 패치를 위해 보정
        if h_coords[-1] != H - patch_size:
            h_coords.append(H - patch_size)
        if w_coords[-1] != W - patch_size:
            w_coords.append(W - patch_size)
        # 슬라이딩 윈도우로 패치 추출
        for i in h_coords:
            for j in w_coords:
                patch = tensor_img[:, i:i+patch_size, j:j+patch_size]
                patches.append(patch)
        patches = torch.stack(patches)
    else:
        # 이미지가 작으면 원본 전체를 하나의 패치로 처리
        patches = tensor_img.unsqueeze(0)
    return patches


def patch_pair_collate_fn(batch):
    """
    batch: list of tuples (input_patch, target_patch)
    각각을 분리하여 스택함.
    """
    inputs = torch.stack([pair[0] for pair in batch])
    targets = torch.stack([pair[1] for pair in batch])
    return inputs, targets

def calculate_psnr(target, output, max_val=1.0):
    """
    target, output: tensor [C, H, W]
    """
    mse = torch.mean((target - output) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 10 * math.log10(max_val**2 / mse.item())
    return psnr

def calculate_batch_psnr(target, output, max_val=1.0):
    """
    target: tensor [batch_size, C, H, W]
    output: tensor [batch_size, C, H, W]
    max_val: 이미지의 최대 값, 보통 [0,1] 범위이면 1.0
    """
    mse = torch.mean((target - output) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 10 * torch.log10(max_val**2 / mse)
    return psnr.item()
