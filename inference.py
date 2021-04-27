import os
import pandas as pd
from PIL import Image
from efficientnet_pytorch import EfficientNet
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, Normalize
from RandAugment import RandAugment

import argparse
from model import *

def seed_everything(seed):
    """
    동일한 조건으로 학습을 할 때, 동일한 결과를 얻기 위해 seed를 고정시킵니다.
    
    Args:
        seed: seed 정수값
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)



class TestDataset(Dataset):
    def __init__(self, img_paths, transform):
        self.img_paths = img_paths
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)

class TestDataAugmentation:
    def __init__(self, type, **args):
        self.size = 384
        augmentation=getattr(self,type)
        augmentation()

    
    def center(self):
        self.transform = transforms.Compose([
        transforms.CenterCrop(self.size),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
   
    def randagument(self):
        self.transform = transforms.Compose([
        transforms.CenterCrop(self.size),
        RandAugment(1,15),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    def baseline(self):
        self.transform = transforms.Compose([
            Resize((512, 384), Image.BILINEAR),
            ToTensor(),
            Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
        ])
    
    def __call__(self, image):
        return self.transform(image) 


@torch.no_grad()
def inference(config):
    # meta 데이터와 이미지 경로를 불러옵니다.
    model_name = 'efficientnet-b3'
    model_path = config.f
    print(model_path)
    test_dir = '/opt/ml/input/data/eval'

    submission = pd.read_csv(os.path.join(test_dir, 'info.csv'))
    image_dir = os.path.join(test_dir, 'images')

    # Test Dataset 클래스 객체를 생성하고 DataLoader를 만듭니다.
    image_paths = [os.path.join(image_dir, img_id) for img_id in submission.ImageID]

    transform = TestDataAugmentation(type='center')
    dataset = TestDataset(image_paths, transform)

    use_cuda = torch.cuda.is_available()

    loader = DataLoader(
        dataset,
        batch_size=34,
        num_workers=4,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MyEfficientNet3(num_classes=18).to(device)
    #model = MyFnNet(num_classes=18).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()


    # 모델이 테스트 데이터셋을 예측하고 결과를 저장합니다.
    all_predictions = []
    with torch.no_grad():
        for images in loader:
            images = images.to(device)
            pred = model(images)
            pred = pred.argmax(dim=-1)
            all_predictions.extend(pred.cpu().numpy())
    submission['ans'] = all_predictions

    # 제출할 파일을 저장합니다.
    submission.to_csv(os.path.join("./", 'submission.csv'), index=False)
    print('test inference is done!')

if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--f', type=str, default="mask-model.pt", help='model file')
    config = parser.parse_args()
    random.seed(777)
    inference(config)