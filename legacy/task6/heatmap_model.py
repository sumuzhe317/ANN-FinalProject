import torch
import os
import cv2
import math
import time
import matplotlib.pyplot as plt
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights,ResNeXt101_32X8D_Weights

from PIL import Image
from utils.dataloader import BirdsDataset, StanfordDogsDataset
from utils.utils import save_checkpoint, _init_fn, set_seed, draw_features
from utils.config import getConfig, getDatasetConfig

class heatmap_model(nn.Module):
 
    def __init__(self,config, num_classes, savepath):
        super(heatmap_model, self).__init__()
        model = models.resnext101_32x8d(weights=None,progress=True)
        # model config
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.7)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=25, gamma=0.1)

        # gpu config
        use_gpu = torch.cuda.is_available() and config.use_gpu
        gpu_ids = [int(r) for r in config.gpu_ids.split(',')]
        if use_gpu:
            if config.multi_gpu:
                model = model.cuda()
                model = torch.nn.DataParallel(model, device_ids=gpu_ids)
            else:
                model = model.cuda()
        model.load_state_dict(torch.load(config.resume)['state_dict'])
        device = torch.device("cuda" if use_gpu else "cpu")
        self.model = model.module
        self.model.eval()
        self.savepath = savepath
        self.num_classes = num_classes
 
    def forward(self, x):
        if True: # draw features or not
            x = self.model.conv1(x)
            draw_features(8,8,x.cpu().numpy(),"{}/f1_conv1.png".format(self.savepath))
 
            x = self.model.bn1(x)
            draw_features(8, 8, x.cpu().numpy(),"{}/f2_bn1.png".format(self.savepath))
 
            x = self.model.relu(x)
            draw_features(8, 8, x.cpu().numpy(), "{}/f3_relu.png".format(self.savepath))
 
            x = self.model.maxpool(x)
            draw_features(8, 8, x.cpu().numpy(), "{}/f4_maxpool.png".format(self.savepath))
 
            x = self.model.layer1(x)
            draw_features(16, 16, x.cpu().numpy(), "{}/f5_layer1.png".format(self.savepath))
 
            x = self.model.layer2(x)
            draw_features(16, 32, x.cpu().numpy(), "{}/f6_layer2.png".format(self.savepath))
 
            x = self.model.layer3(x)
            draw_features(32, 32, x.cpu().numpy(), "{}/f7_layer3.png".format(self.savepath))
 
            x = self.model.layer4(x)
            draw_features(32, 32, x.cpu().numpy()[:, 0:1024, :, :], "{}/f8_layer4_1.png".format(self.savepath))
            draw_features(32, 32, x.cpu().numpy()[:, 1024:2048, :, :], "{}/f8_layer4_2.png".format(self.savepath))
 
            x = self.model.avgpool(x)
            plt.plot(np.linspace(1, 2048, 2048), x.cpu().numpy()[0, :, 0, 0])
            plt.savefig("{}/f9_avgpool.png".format(self.savepath))
            plt.clf()
            plt.close()
 
            x = torch.flatten(x, 1)
            x = self.model.fc(x)
            plt.plot(np.linspace(1, self.num_classes, self.num_classes), x.cpu().numpy()[0, :])
            plt.savefig("{}/f10_fc.png".format(self.savepath))
            plt.clf()
            plt.close()
        else :
            # See note [TorchScript super()]
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
 
        return x