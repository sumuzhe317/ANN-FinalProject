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
from utils.utils import save_checkpoint, _init_fn, set_seed
from utils.config import getConfig, getDatasetConfig
from legacy.task6.heatmap_model import heatmap_model
import torch.nn.functional as F
from torchvision.transforms.functional import normalize, resize, to_tensor, to_pil_image
from matplotlib import cm


def preprocess_train(image):
    width, height = image.size
    if width > height and width > 512:
        height = math.floor(512 * height / width)
        width = 512
    elif width < height and height > 512:
        width = math.floor(512 * width / height)
        height = 512
    pad_values = (
        (512 - width) // 2 + (0 if width % 2 == 0 else 1),
        (512 - height) // 2 + (0 if height % 2 == 0 else 1),
        (512 - width) // 2,
        (512 - height) // 2,
    )
    return transforms.Compose([
        transforms.RandomGrayscale(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize((height, width)),
        transforms.Pad(pad_values),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.Lambda(lambda x: x[:3]),  # Remove the alpha channel if it's there
    ])(image)

def preprocess_test(image):
    width, height = image.size
    if width > height and width > 512:
        height = math.floor(512 * height / width)
        width = 512
    elif width < height and height > 512:
        width = math.floor(512 * width / height)
        height = 512
    pad_values = (
        (512 - width) // 2 + (0 if width % 2 == 0 else 1),
        (512 - height) // 2 + (0 if height % 2 == 0 else 1),
        (512 - width) // 2,
        (512 - height) // 2,
    )
    return transforms.Compose([
        transforms.RandomGrayscale(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize((height, width)),
        transforms.Pad(pad_values),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.Lambda(lambda x: x[:3]),  # Remove the alpha channel if it's there
    ])(image)

def preprocess_heatmap(image):
    width, height = image.size
    if width > height and width > 512:
        height = math.floor(512 * height / width)
        width = 512
    elif width < height and height > 512:
        width = math.floor(512 * width / height)
        height = 512
    pad_values = (
        (512 - width) // 2 + (0 if width % 2 == 0 else 1),
        (512 - height) // 2 + (0 if height % 2 == 0 else 1),
        (512 - width) // 2,
        (512 - height) // 2,
    )
    return transforms.Compose([
        transforms.Resize((height, width)),
        transforms.Pad(pad_values),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.Lambda(lambda x: x[:3]),  # Remove the alpha channel if it's there
    ])(image)

def train():
    # set seed
    set_seed(config.seed)
    
    num_epochs = config.epochs

    best_acc = 0
    best_loss = None
    warm_up_iter = 0
    T_max = 50
    lr_max = 0.05	# 最大值
    lr_min = 0.001	# 最小值

    num_classes = dataset_classes

    # model config
    model = models.resnext101_32x8d(weights=ResNeXt101_32X8D_Weights.IMAGENET1K_V2,progress=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)

    # log config
    writer = SummaryWriter(config.log_path)

    # gpu config
    use_gpu = torch.cuda.is_available() and config.use_gpu
    gpu_ids = [int(r) for r in config.gpu_ids.split(',')]
    if use_gpu:
        if config.multi_gpu:
            model = model.cuda()
            model = torch.nn.DataParallel(model, device_ids=gpu_ids)
        else:
            model = model.cuda()
    device = torch.device("cuda" if use_gpu else "cpu")

    # train
    for epoch in range(num_epochs):

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            if best_loss is None:
                best_loss = epoch_loss + 1
            if phase == 'val':
                save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'best_prec1': epoch_acc,
                    'optimizer': optimizer.state_dict(),
                }, is_best=epoch_acc > best_acc, path=config.checkpoint_path)
                best_loss = epoch_loss if epoch_loss < best_loss else best_loss
                best_acc = epoch_acc if epoch_acc > best_acc else best_acc

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Record loss and accuracy into TensorBoard
            writer.add_scalar(f'{phase}/loss', epoch_loss, epoch)
            writer.add_scalar(f'{phase}/accuracy', epoch_acc, epoch)

        print()

    for phase in ['test']:
        model.eval()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)


            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]

        print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # Record loss and accuracy into TensorBoard
        writer.add_scalar(f'{phase}/loss', epoch_loss, 0)
        writer.add_scalar(f'{phase}/accuracy', epoch_acc, 0)

    writer.close()

def test():
    # set seed
    set_seed(config.seed)
    
    num_epochs = config.epochs

    best_acc = 0
    best_loss = None
    warm_up_iter = 0
    T_max = 50
    lr_max = 0.05	# 最大值
    lr_min = 0.001	# 最小值

    num_classes = dataset_classes

    # model config
    model = models.resnext101_32x8d(weights=None,progress=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.7)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)

    # log config
    writer = SummaryWriter(config.log_path)

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

    # test
    for phase in ['test']:
        model.eval()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)


            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]

        print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # Record loss and accuracy into TensorBoard
        writer.add_scalar(f'{phase}/loss', epoch_loss, 0)
        writer.add_scalar(f'{phase}/accuracy', epoch_acc, 0)

    writer.close()

hook_a = None
def _hook_a(module,inp,out):
    global hook_a
    hook_a = out

hook_g = None
def _hook_g(module,inp,out):
    global hook_g
    hook_g = out[0]


def gen_heatmap():
    # set seed
    set_seed(config.seed)
    
    num_classes = dataset_classes
    savepath=config.heatmap_path
    if not os.path.exists(savepath):
        os.mkdir(savepath)

    # model config
    model = models.resnext101_32x8d(weights=None,progress=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.7)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)

    # log config
    writer = SummaryWriter(config.log_path)

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


    # 加载输入图像
    image_path = config.input_img
    image = Image.open(image_path).convert('RGB')

    # 数据预处理
    input_image = preprocess_heatmap(image).unsqueeze(0).to(device)

    with torch.no_grad():
        son_modules = dict(model.named_modules())
        for name in son_modules.keys():
            if name != 'module.layer4':
                continue
            # print(submodule_dict)
            target_layer = son_modules[name]
            print(target_layer)

            hook1 = target_layer.register_forward_hook(_hook_a)
            hook2 = target_layer.register_backward_hook(_hook_g)

            scores = model(input_image)
            class_idx = 0 # class 232 corresponding to the border collie
            loss = scores[:,class_idx].sum()
            loss.requires_grad_(True)   #加入此句就行了
            loss.backward()

            weights = hook_g.squeeze(0).mean(dim=(1,2))
            cam = (weights.view(*weights.shape, 1, 1) * hook_a.squeeze(0)).sum(0)
            cam = F.relu(cam)
            cam.sub_(cam.flatten(start_dim=-2).min(-1).values.unsqueeze(-1).unsqueeze(-1))
            cam.div_(cam.flatten(start_dim=-2).max(-1).values.unsqueeze(-1).unsqueeze(-1))
            cam = cam.data.cpu().numpy()

            heatmap = to_pil_image(cam, mode='F')
            overlay = heatmap.resize(input_image.size, resample=Image.BICUBIC)
            cmap = cm.get_cmap('jet')
            overlay = (255 * cmap(np.asarray(overlay) ** 2)[:, :, :3]).astype(np.uint8)
            alpha = .7
            result = (alpha * np.asarray(input_image) + (1 - alpha) * overlay).astype(np.uint8)
            plt.imshow(result)
            plt.savefig("{}/{}".format(savepath,name))
            plt.clf()
            plt.close()
            hook1.remove()
            hook2.remove()

if __name__ == "__main__":
    config = getConfig()
    if config.action == 'train':
        dataset, dataloaders, dataset_sizes, dataset_classes = getDatasetConfig(config=config,dataset_name=config.dataset,project_root=os.getcwd(),preprocess_train=preprocess_train, preprocess_test=preprocess_test)
        train()
    elif config.action == 'test':
        dataset, dataloaders, dataset_sizes, dataset_classes = getDatasetConfig(config=config,dataset_name=config.dataset,project_root=os.getcwd(),preprocess_train=preprocess_train, preprocess_test=preprocess_test)
        test()
    elif config.action == 'gen_heatmap':
        dataset, dataloaders, dataset_sizes, dataset_classes = getDatasetConfig(config=config,dataset_name=config.dataset,project_root=os.getcwd(),preprocess_train=preprocess_train, preprocess_test=preprocess_test)
        gen_heatmap()