# Dataset
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from autoaugment import ImageNetPolicy

import os
from PIL import Image
import numpy as np
from glob import glob
import re
from RandAugment import RandAugment

class MaskDataset(Dataset):
  def __init__(self, data_root, transform=None):
    #이미지 리스트(파일 경로)
    #이미지 개수, 이미지 크기, 이미지 변형 값을 설정
    super(MaskDataset, self).__init__()

    self.imgs = self._load_image_data(data_root)
    self.len = len(self.imgs)
    self.transform = transform

  def _load_image_data(self, data_root):
    imgs = []
    for dir in glob(data_root + '/*'):
        imgs.extend(glob(dir+'/*'))
    return imgs

  def __getitem__(self, index):
    img_path = self.imgs[index]
  
    # Image Loading
    img = Image.open(img_path)
    img = img.convert('RGB')

    if self.transform:
      img = self.transform(img)

    label = self._get_label_from_img_path(img_path)

    return img, label
  def __len__(self):
    return self.len
  
  def _get_label_from_img_path(self, img_path):
 
    img_path = re.split("_|/|\.", img_path)
    gender=img_path[8]
    age= img_path[10]
    mask= img_path[11][0]

    gender_label = {'male':0, 'female':3}
    mask_label={'m':0, 'i':6,'n':12}
    age = int(age)

    if age < 30: age = 0
    elif age < 58: age =1
    else : age =2

    label = gender_label[gender] + mask_label[mask]+ age//30
    label = gender_label[gender] + mask_label[mask]+ age

    return label


class DataAugmentation:
    def __init__(self, type, **args):
      augmentation=getattr(self,type)
      size = 384
      self.size = size
      augmentation()
    
    def randaugment(self):
      self.transform =transforms.Compose([
        transforms.CenterCrop(self.size), 
        transforms.RandomHorizontalFlip(p=0.5),
        RandAugment(1,15),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
      ])

    def imagenet(self):
      self.transform =transforms.Compose([
        transforms.CenterCrop(self.size),
        transforms.RandomHorizontalFlip(p=0.5),
        ImageNetPolicy(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
      ])

    def resize(self):
      #중앙을 중심으로
      #  이미지를 자르고, resize 함
      self.transform =transforms.Compose([
        transforms.RandomResizedCrop(self.size), 
        transforms.ColorJitter(brightness=(0.5, 1.5), 
                               contrast=(0.5, 1.5), 
                               saturation=(0.5, 1.5)),
        transforms.RandomAffine(10),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
      ])

    def center(self):
      #중앙을 중심으로
      #  이미지를 자르고, resize 함
      self.transform =transforms.Compose([
        transforms.CenterCrop(self.size), 
        transforms.ColorJitter(brightness=(0.5, 1.5), 
                               contrast=(0.5, 1.5), 
                               saturation=(0.5, 1.5)),
        transforms.RandomAffine(10),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
      ])

    def normal(self):
      #중앙을 중심으로
      #  이미지를 자르고, resize 함
      self.transform =transforms.Compose([
        transforms.CenterCrop(self.size), 
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
      ])


    def sample_augment(self):
      self.transform =transforms.Compose([
        transform.RandomResizedCrop(size, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333)),
        transforms.CenterCrop(28),
        transforms.RandomAffine(10),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
          transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
      ])

    # # randomcrop_albumentation 사용 ver2
    # randomcrop_train_transforms = A.Compose([
    #     A.Resize(256,256),
    #     A.RandomResizedCrop(256,256, scale=(0.4, 1.0) , ratio = (0.75 , 1)) ,          
    #     A.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
    #     A.pytorch.transforms.ToTensor()
    # ])

    def __call__(self, image):
        return self.transform(image)



if __name__ =="__main__":
    data_root = '/opt/ml/input/data/train/images'
  
    transform = DataAugmentation(type='resize')
    test_dataset = MaskDataset(data_root, transform=transform)
    print(len(test_dataset))
    img,  label = test_dataset[1]
    print(label)
