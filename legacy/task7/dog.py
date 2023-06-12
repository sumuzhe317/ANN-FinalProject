import torch
import os
import math
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import datasets, transforms, models
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights,ResNeXt101_32X8D_Weights

from utils.dataloader import BirdsDataset, StanfordDogsDataset
from utils.utils import save_checkpoint, _init_fn, set_seed
from utils.config import getConfig, getDatasetConfig
from legacy.task7.FGSM import fgsm_attack

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

def train():
    # set seed
    set_seed(config.seed)
    epsilons = 0.1
    
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
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.7)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

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
                if phase == 'train':
                    inputs.requires_grad = True

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        inputs_grad = inputs.grad.data
                        perturbed_inputs = fgsm_attack(inputs, epsilons, inputs_grad, device)
                        optimizer.zero_grad()
                        outputs = model(perturbed_inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
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

if __name__ == "__main__":
    config = getConfig()
    if config.action == 'train':
        dataset, dataloaders, dataset_sizes, dataset_classes = getDatasetConfig(config=configdataset_name=config.dataset,project_root=os.getcwd(),preprocess_train=preprocess_train, preprocess_test=preprocess_test)
        train()
    else:
        dataset, dataloaders, dataset_sizes, dataset_classes = getDatasetConfig(config=configdataset_name=config.dataset,project_root=os.getcwd(),preprocess_train=preprocess_train, preprocess_test=preprocess_test)
        test()