import os
from PIL import Image
from utils import RandomBlur, extract_patches
from torchvision import transforms
from torch.utils.data import Dataset

class TestDataset(Dataset):
    def __init__(self, image_dir, transform_target=None, transform_input=None, grayscale=False):
        """
        image_dir: 테스트 이미지가 저장된 폴더 경로.
        transform_target: 원본 이미지를 tensor로 변환하는 transform.
        transform_input: 블러링이 적용된 이미지를 생성하는 transform.
        grayscale: True이면 이미지를 grayscale("L")로 로드하여 채널 1로 만듦.
        """
        self.image_dir = image_dir
        self.image_paths = [os.path.join(image_dir, f) 
                            for f in os.listdir(image_dir) 
                            if f.lower().endswith(('jpg','jpeg','png'))]
        self.transform_target = transform_target if transform_target is not None else transforms.ToTensor()
        self.transform_input = transform_input if transform_input is not None else transforms.Compose([
            RandomBlur(probability=0.7, radius_range=(1, 3)),
            transforms.ToTensor()
        ])
        self.grayscale = grayscale

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mode = "L" if self.grayscale else "RGB"
        img = Image.open(img_path).convert(mode)
        target = self.transform_target(img)
        input_img = self.transform_input(img)
        return input_img, target
    
class TrainDataset(Dataset):
    def __init__(self, image_dir, patch_size=48, overlap=9, p_max=60,
                 transform_target=None, transform_input=None, grayscale=False):
        """
        image_dir: 이미지가 저장된 폴더 경로.
        patch_size: 추출할 패치 크기 (첫 번째 코드와 동일하게 512)
        overlap: 패치 간 겹침 영역 (예: 96)
        p_max: 이미지가 충분히 큰지 판단하는 임계값 (예: 800)
        transform_target: 원본 이미지를 tensor로 변환하는 transform (target)
        transform_input: 랜덤 블러를 포함한, input 이미지를 생성하는 transform
        grayscale: True이면 이미지를 grayscale("L")로 로드하여 채널 1로 만듦.
        """
        self.image_dir = image_dir
        self.image_paths = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith(('jpg', 'jpeg', 'png'))
        ]
        self.transform_target = transform_target if transform_target is not None else transforms.ToTensor()
        self.transform_input = transform_input if transform_input is not None else transforms.Compose([
            # 예시로 RandomBlur transform; 실제 구현에 맞게 수정할 것
            # RandomBlur(probability=0.7, radius_range=(1, 3)),
            transforms.ToTensor()
        ])
        self.patch_size = patch_size
        self.overlap = overlap
        self.p_max = p_max
        self.grayscale = grayscale
        
        # 미리 각 이미지에서 input과 target 패치를 추출하여 저장
        self.patch_pairs = []  # list of tuples: (input_patch, target_patch)
        for img_path in self.image_paths:
            mode = "L" if self.grayscale else "RGB"
            img = Image.open(img_path).convert(mode)
            # 원본 target 이미지 (예: 단순히 tensor 변환)
            target_img = self.transform_target(img)  # shape: C x H x W
            # input 이미지는 랜덤 블러 적용
            input_img = self.transform_input(img)  # shape: C x H x W
            
            # 두 이미지 모두 첫 번째 코드와 동일한 방식의 patch extraction
            target_patches = extract_patches(target_img, self.patch_size, self.overlap, self.p_max)
            input_patches = extract_patches(input_img, self.patch_size, self.overlap, self.p_max)
            # 두 텐서의 패치 개수는 동일함
            num_patches = target_patches.shape[0]
            for i in range(num_patches):
                self.patch_pairs.append( (input_patches[i], target_patches[i]) )
                
    def __len__(self):
        return len(self.patch_pairs)
    
    def __getitem__(self, idx):
        return self.patch_pairs[idx]
    



class DIV64TrainDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, patch_size=48, stride=40,
                 transform_target=None, transform_input=None, grayscale=False):
        """
        hr_dir: 고해상도 이미지가 저장된 폴더 (DIV2K_train_HR)
        lr_dir: 저해상도 이미지가 저장된 폴더 (DIV2K_train_LR_bicubic)
        patch_size: 추출할 패치 크기 (예: 48)
        stride: 패치 추출 간격 (예: 40)
        transform_target: 고해상도 이미지를 tensor로 변환하는 transform (target)
        transform_input: 저해상도 이미지를 tensor로 변환하는 transform
        grayscale: True이면 이미지를 grayscale("L")로 로드하여 채널 1로 변환.
        """
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir

        self.hr_image_paths = sorted([
            os.path.join(hr_dir, f)
            for f in os.listdir(hr_dir)
            if f.lower().endswith(('jpg', 'jpeg', 'png'))
        ])
        self.lr_image_paths = sorted([
            os.path.join(lr_dir, f)
            for f in os.listdir(lr_dir)
            if f.lower().endswith(('jpg', 'jpeg', 'png'))
        ])

        self.transform_target = transform_target if transform_target is not None else transforms.ToTensor()
        self.transform_input = transform_input if transform_input is not None else transforms.ToTensor()
        self.patch_size = patch_size
        self.stride = stride
        self.grayscale = grayscale

        # 각 이미지별로 패치 쌍들을 저장 (이미지마다 추출되는 패치 수가 다름)
        self.patch_pairs = []  # list of tuples: (input_patches, target_patches) per image
        for hr_path, lr_path in zip(self.hr_image_paths, self.lr_image_paths):
            mode = "L" if self.grayscale else "RGB"
            hr_img = Image.open(hr_path).convert(mode)
            lr_img = Image.open(lr_path).convert(mode)
            # LR 이미지를 HR 이미지 크기로 업샘플링 (bicubic)
            lr_img = lr_img.resize(hr_img.size, Image.BICUBIC)
            # transform 적용
            target_img = self.transform_target(hr_img)
            input_img = self.transform_input(lr_img)
            # 각 이미지에서 patch 추출
            target_patches = extract_patches(target_img, self.patch_size, self.stride)
            input_patches = extract_patches(input_img, self.patch_size, self.stride)
            # 두 텐서의 patch 개수는 이미지마다 다를 수 있음
            self.patch_pairs.append((input_patches, target_patches))

    def __len__(self):
        # 전체 이미지 수를 반환합니다.
        return len(self.patch_pairs)

    def __getitem__(self, idx):
        # 한 이미지에서 추출된 모든 patch들을 반환합니다.
        input_patches, target_patches = self.patch_pairs[idx]
        return input_patches, target_patches
