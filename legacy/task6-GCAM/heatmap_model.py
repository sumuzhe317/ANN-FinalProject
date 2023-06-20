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

import numpy as np
import torch.nn.functional as F
import torchvision.models as models
from torchvision.transforms.functional import normalize, resize, to_tensor, to_pil_image
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image

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
        x = self.model.forward(x)
        return x