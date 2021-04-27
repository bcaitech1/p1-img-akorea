from efficientnet_pytorch import EfficientNet
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
from dataset import MaskDataset, DataAugmentation
import time, os
import copy, random
from adamp import AdamP
import wandb
from torchsampler import ImbalancedDatasetSampler
import argparse
import json
from importlib import import_module
from sklearn.metrics import f1_score
from loss import create_criterion
from sklearn.model_selection import train_test_split
import math

torch.cuda.empty_cache()

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


def train_model(config, wandb):
    
    seed_everything(config.seed)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_module = getattr(import_module("model"), config.model)  
    model = model_module(
        num_classes=18
    ).to(device)
    
    #model = torch.nn.DataParallel(model)
   
    ########  DataSet

    transform = DataAugmentation(type=config.transform) #center_384_1
    dataset = MaskDataset(config.data_dir, transform=transform)


    len_valid_set = int(config.data_ratio *len(dataset))
    len_train_set = len(dataset) - len_valid_set
    dataloaders, batch_num = {}, {}
    
    train_dataset , valid_dataset = torch.utils.data.random_split(dataset ,
                                                                    [len_train_set, len_valid_set])
    if config.random_split ==0:
        print("tbd")

    sampler =None
    
    if config.sampler =='ImbalancedDatasetSampler':   
        sampler =ImbalancedDatasetSampler(train_dataset)
    
    use_cuda = torch.cuda.is_available()
   

    dataloaders['train'] = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=config.batch_size,
                                            sampler=sampler,
                                            shuffle=False, num_workers=4,
                                            pin_memory=use_cuda)

  
    dataloaders['valid'] = torch.utils.data.DataLoader(valid_dataset,
                                            batch_size=config.batch_size,
                                            shuffle=False, num_workers=4,
                                            pin_memory=use_cuda)

    batch_num['train'], batch_num['valid'] = len(dataloaders['train']), len(dataloaders['valid'])



    #Loss
    criterion = create_criterion(config.criterion)
    
    
    #Optimizer
    optimizer= optim.SGD(model.parameters(), lr=config.lr, momentum=0.9)
    
    if config.optim == "AdamP":
        optimizer = AdamP(model.parameters(), lr=config.lr, betas=(0.9, 0.999), weight_decay=config.weight_decay)
    elif config.optim  == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=config.lr , weight_decay=config.weight_decay)
    


    #Scheduler 
    # Decay LR by a factor of 0.1 every 7 epochs
    #exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    if config.lr_scheduler == "cosine":
        print('cosine')
        Q = math.floor(len(train_dataset)/config.batch_size+1)*config.epochs/7
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max = Q)
        #ConsineAnnealingWarmRestarts


    since = time.time()
    low_train = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_loss, train_acc, valid_loss, valid_acc = [], [], [], []
    num_epochs = config.epochs
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss, running_corrects, num_cnt = 0.0, 0, 0
            runnnig_f1 =0
            
            # Iterate over data.
            idx =0
            for inputs, labels in dataloaders[phase]:
                idx+=1
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    else:
                        runnnig_f1 += f1_score(labels.data.detach().cpu(), preds.detach().cpu(), average='macro')

            
                # statistics
                val_loss = loss.item() * inputs.size(0)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                num_cnt += len(labels)
                if idx  % 100 == 0:
                    _loss = loss.item() /config.batch_size
                    print(
                        f"Epoch[{epoch}/{config.epochs}]({idx}/{batch_num[phase]}) || "
                        f"{phase} loss {_loss:4.4} ")
                
            if phase == 'train':
                scheduler.step()
            
            epoch_loss = float(running_loss / num_cnt)
            epoch_acc  = float((running_corrects.double() / num_cnt).cpu()*100)
            epoch_f1 = float(runnnig_f1/ num_cnt)
            if phase == 'train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
                if config.wandb:
                    wandb.log({"Train acc":epoch_acc })
            else:
                valid_loss.append(epoch_loss)
                valid_acc.append(epoch_acc)
                if config.wandb:
                    wandb.log({"Valid acc":epoch_acc })
                    wandb.log({"F1 Score":epoch_f1 })
                    
                
                

            print('{} Loss: {:.2f} Acc: {:.1f} f1 :{:.3f}'.format(phase, epoch_loss, epoch_acc, epoch_f1))
           
            # deep copy the model
            if phase == 'valid':
                if epoch_acc > best_acc:
                    best_idx = epoch
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    print('==> best model saved - %d / %.1f'%(best_idx, best_acc))
                    low_train = 0
                elif epoch_acc < best_acc:
                    print('==> model finish')
                    low_train  +=1
 
        if low_train  > 0 and epoch > 4:
            break

        if phase == 'valid':
            if epoch_acc < 80:
                print('Stop valid is so low')
                break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best valid Acc: %d - %.1f' %(best_idx, best_acc))


    # load best model weights
    model.load_state_dict(best_model_wts)
    #torch.save(model.state_dict(), 'mask_model.pt')
    torch.save(model.state_dict(), config.name+'.pt')
    print('model saved')
    if config.wandb:
        wandb.finish()
    return model, best_idx, best_acc, train_loss, train_acc, valid_loss, valid_acc

def parse_args(parser, name):
    with open(os.path.join("./config/", name+".json"), 'r', encoding='utf-8') as f:
        config = json.load(f)

    parser.add_argument('--seed', type=int, default=config["seed"], help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=config["epochs"], help='number of epochs to train (default: 1)')
    parser.add_argument('--transform', type=str, default=config["transform"], help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument('--batch_size', type=int, default=config["batch_size"], help='input batch size for training (default: 64)')
    parser.add_argument('--model', type=str, default=config["model"], help='model type (default: BaseModel)')
    parser.add_argument('--optim', type=str, default=config["optim"], help='optimizer type (default: SGD)')
    parser.add_argument('--lr', type=float, default=config["lr"], help='learning rate (default: 1e-3)')
    parser.add_argument('--data_ratio', type=float, default=config["data_ratio"], help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default=config["criterion"], help='criterion type (default: cross_entropy)')
    parser.add_argument('--weight_decay', type=float, default=config["weight_decay"], help='tbd')
    
    parser.add_argument('--lr_scheduler', type=str, default=config["lr_scheduler"], help='lr_scheduler')
    parser.add_argument('--random_split', type=float, default=config["random_split"], help='random_split')
    parser.add_argument('--sampler', type=str, default=config["sampler"], help='sampler')
    parser.add_argument('--test', type=str, default=config["test"], help='test')
    


    args = parser.parse_args()
    return args



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    conf_name = "base"

    config=parse_args(parser ,conf_name)    
    run_wandb = True
    
    if run_wandb:
        wandb.init(project="mask")
        wandb.config.update(config) 
        config.name = wandb.run.name 
    else:
        config.name = conf_name 
    
    config.epochs = 120

    config.wandb = run_wandb
    config.data_dir = '/opt/ml/input/data/etrain/images' # 데이터 위치
    print(config)
    model_ft = train_model(config, wandb)

#https://greeksharifa.github.io/references/2020/06/10/wandb-usage/
#wandb agent asally/mask-classification/mvnwtglf