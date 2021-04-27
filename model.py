import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
import torchvision.models as models
import timm

def conv(in_num, out_num, kernel_size=3, padding=1, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_num,out_num,kernel_size=kernel_size,stride=stride),
        nn.BatchNorm2d(out_num),
        nn.LeakyReLU())


class ResidualBlock(nn.Module):
    def __init__(self, in_channel):
        super(ResidualBlock,self).__init__()
        reduce_channel = in_channel//2
        self.layer1 = conv_batch(in_channel,reduce_channel,kernel_size=1, padding =0)         
        self.layer2 = conv_batch(reduce_channel,in_channel)
    
    def forward(self, x):
        residual  = x 
        out = self.layer1(x)
        out=  self.layer2(out)
        out += residual
        return out

class MyDarknet(nn.Module):
    def __init__(self, block, num_classes):
        super(MyDarknet,self).__init__()
        
        self.conv1 = conv_batch(3, 32)
        self.conv2 = conv_batch(32, 64, stride=2)
        self.residual_block1 = self.make_layer(block, in_channels=64, num_blocks=1)
        self.conv3 = conv_batch(64, 128, stride=2)
        self.residual_block2 = self.make_layer(block, in_channels=128, num_blocks=2)
        self.conv4 = conv_batch(128, 256, stride=2)
        self.residual_block3 = self.make_layer(block, in_channels=256, num_blocks=8)
        self.conv5 = conv_batch(256, 512, stride=2)
        self.residual_block4 = self.make_layer(block, in_channels=512, num_blocks=8)
        self.conv6 = conv_batch(512, 1024, stride=2)
        self.residual_block5 = self.make_layer(block, in_channels=1024, num_blocks=4)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, self.num_classes)


    def make_layer(self,block,in_channel, num_blocks):
        layers=[]
        for i in range(0, num_blocks):
            layers.append(block(in_channel))
        return nn.Sequential(*layers)
    
    def forward(self,x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.residual_block1(out)
        out = self.conv3(out)
        out = self.residual_block2(out)
        out = self.conv4(out)
        out = self.residual_block3(out)
        out = self.conv5(out)
        out = self.residual_block4(out)
        out = self.conv6(out)
        out = self.residual_block5(out)
        out = self.global_avg_pool(out)
        out = out.view(-1, 1024)
        out = self.fc(out)
        
        return out

class MyEfficientNet(nn.Module):
    def __init__(self, num_classes, model='efficientnet-b0'):
        super(MyEfficientNet, self).__init__()
        self.backbone = EfficientNet.from_pretrained(model, num_classes=num_classes)
    def forward(self, x):
        x = self.backbone(x)
        return x

class MyEfficientNet1(MyEfficientNet):
    def __init__(self, num_classes, model='efficientnet-b1'):
        super(MyEfficientNet1, self).__init__(num_classes, model)

class MyEfficientNet2(MyEfficientNet):
    def __init__(self, num_classes, model='efficientnet-b2'):
        super(MyEfficientNet2, self).__init__(num_classes, model)

class MyEfficientNet3(MyEfficientNet):
    def __init__(self, num_classes, model='efficientnet-b3'):
        super(MyEfficientNet3, self).__init__(num_classes, model)       

class MyResNet18(nn.Module):
    def __init__(self, num_classes):
        super(MyResNet18, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        return x

class MyResNet34(nn.Module):
    def __init__(self, num_classes):
        super(MyResNet34, self).__init__()
        self.backbone = models.resnet34(pretrained=True)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        return x

class MyFnNet(nn.Module):
    def __init__(self, num_classes):
        super(MyFnNet, self).__init__()
        self.backbone = timm.create_model("dm_nfnet_f0", pretrained=True)
        num_ftrs = self.backbone.head.fc.in_features
        self.backbone.head.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        return x

if __name__ =="__main__":
    #MyDarknet(ResidualBlock, 18)
    MyFnNet(18)