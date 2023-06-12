from collections import OrderedDict
import os
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader

from utils.dataloader import BirdsDataset, StanfordDogsDataset
from utils.utils import save_checkpoint, _init_fn, set_seed
from utils.config import getConfig, getDatasetConfig
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import ViT_B_16_Weights,ViT_B_32_Weights # ViT_B_16_Weights.DEFAULT

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
        transforms.Resize((height, width)),
        transforms.Pad(pad_values),
        transforms.RandomResizedCrop((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomGrayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
        transforms.Resize((height, width)),
        transforms.Pad(pad_values),
        transforms.Resize((256,256)),
        transforms.CenterCrop((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Lambda(lambda x: x[:3]),  # Remove the alpha channel if it's there
    ])(image)

def main():
    # 设置随机种子
    torch.manual_seed(2023)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 模型配置
    num_classes = dataset_classes

    # 创建Vision Transformer模型
    model = models.vision_transformer.vit_b_32(weights=ViT_B_32_Weights.DEFAULT, progress=True, image_size=224)
    heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
    if model.representation_size is None:
        heads_layers["head"] = nn.Linear(model.hidden_dim, num_classes)
    else:
        heads_layers["pre_logits"] = nn.Linear(model.hidden_dim, model.representation_size)
        heads_layers["act"] = nn.Tanh()
        heads_layers["head"] = nn.Linear(model.representation_size, num_classes)
    model.heads = nn.Sequential(heads_layers)
    if hasattr(model.heads, "pre_logits") and isinstance(model.heads.pre_logits, nn.Linear):
        fan_in = model.heads.pre_logits.in_features
        nn.init.trunc_normal_(model.heads.pre_logits.weight, std=math.sqrt(1 / fan_in))
        nn.init.zeros_(model.heads.pre_logits.bias)
    if isinstance(model.heads.head, nn.Linear):
        nn.init.zeros_(model.heads.head.weight)
        nn.init.zeros_(model.heads.head.bias)

    # 将模型放在GPU上（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.7)

    # 训练和验证
    num_epochs = 10

    for epoch in range(num_epochs):
        # 训练
        model.train()
        train_loss = 0.0
        train_correct = 0

        for images, labels in dataloaders['train']:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_correct += torch.sum(preds == labels.data)

        train_loss = train_loss / dataset_sizes['train']
        train_acc = train_correct.double() / dataset_sizes['train']

        # 验证
        model.eval()
        val_loss = 0.0
        val_correct = 0

        with torch.no_grad():
            for images, labels in dataloaders['val']:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                val_correct += torch.sum(preds == labels.data)

        val_loss = val_loss / dataset_sizes['val']
        val_acc = val_correct.double() / dataset_sizes['val']

        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}")

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
    # 创建Vision Transformer模型
    model = models.vision_transformer.vit_b_32(weights=ViT_B_32_Weights.DEFAULT, progress=True, image_size=224)
    heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
    if model.representation_size is None:
        heads_layers["head"] = nn.Linear(model.hidden_dim, num_classes)
    else:
        heads_layers["pre_logits"] = nn.Linear(model.hidden_dim, model.representation_size)
        heads_layers["act"] = nn.Tanh()
        heads_layers["head"] = nn.Linear(model.representation_size, num_classes)
    model.heads = nn.Sequential(heads_layers)
    if hasattr(model.heads, "pre_logits") and isinstance(model.heads.pre_logits, nn.Linear):
        fan_in = model.heads.pre_logits.in_features
        nn.init.trunc_normal_(model.heads.pre_logits.weight, std=math.sqrt(1 / fan_in))
        nn.init.zeros_(model.heads.pre_logits.bias)
    if isinstance(model.heads.head, nn.Linear):
        nn.init.zeros_(model.heads.head.weight)
        nn.init.zeros_(model.heads.head.bias)
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
    # 创建Vision Transformer模型
    model = models.vision_transformer.vit_b_32(weights=None, progress=True, image_size=224)
    heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
    if model.representation_size is None:
        heads_layers["head"] = nn.Linear(model.hidden_dim, num_classes)
    else:
        heads_layers["pre_logits"] = nn.Linear(model.hidden_dim, model.representation_size)
        heads_layers["act"] = nn.Tanh()
        heads_layers["head"] = nn.Linear(model.representation_size, num_classes)
    model.heads = nn.Sequential(heads_layers)
    if hasattr(model.heads, "pre_logits") and isinstance(model.heads.pre_logits, nn.Linear):
        fan_in = model.heads.pre_logits.in_features
        nn.init.trunc_normal_(model.heads.pre_logits.weight, std=math.sqrt(1 / fan_in))
        nn.init.zeros_(model.heads.pre_logits.bias)
    if isinstance(model.heads.head, nn.Linear):
        nn.init.zeros_(model.heads.head.weight)
        nn.init.zeros_(model.heads.head.bias)
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
        dataset, dataloaders, dataset_sizes, dataset_classes = getDatasetConfig(config=config,dataset_name=config.dataset,project_root=os.getcwd(),preprocess_train=preprocess_train, preprocess_test=preprocess_test)
        train()
    else:
        dataset, dataloaders, dataset_sizes, dataset_classes = getDatasetConfig(config=config,dataset_name=config.dataset,project_root=os.getcwd(),preprocess_train=preprocess_train, preprocess_test=preprocess_test)
        test()