import os
import argparse
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import math

from utils.utils import set_seed, _init_fn, get_time
from utils.dataloader import *


def getConfig():
    parser = argparse.ArgumentParser()

    # train or test
    # action = parser.add_subparsers()
    # action.add_parser('train', action='store_true', help='run train')
    # action.add_parser('test', action='store_true', help='run test')
    parser.add_argument('action', choices=('train', 'test', 'gen_heatmap','train_stage_1','train_stage_2'))
    # dataset
    parser.add_argument('--dataset', metavar='DIR',
                        help='name of the dataset')
    parser.add_argument('--image-size', '-i', default=512, type=int,
                        metavar='N', help='image size (default: 512)')
    parser.add_argument('--input-size', '-cs', default=448, type=int,
                        metavar='N', help='the input size of the model (default: 448)')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    # optimizer config
    parser.add_argument('--optim', default='sgd', type=str,
                        help='the name of optimizer(adam,sgd)')
    parser.add_argument('--scheduler', default='step', type=str,
                        help='the name of scheduler(step,plateau)')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,
                        metavar='W', help='weight decay (default: 1e-5)')

    # model config
    parser.add_argument('--model-name', default='resnext101_32x8d', type=str,
                        help='model name')

    # training config
    parser.add_argument('--use-gpu', action="store_true", default=False,
                        help='whether use gpu or not, default True')
    parser.add_argument('--multi-gpu', action="store_true", default=False,
                        help='whether use multiple gpus or not, default True')
    parser.add_argument('--gpu-ids', default='0,1',
                        help='gpu id list(eg: 0,1,2...)')
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=16, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--checkpoint-path', default='checkpoint/unknown_set/'+get_time(), type=str, metavar='checkpoint_path',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--log-path', default='runs/unknown_set/'+get_time(),type=str,help='path to store the log data using tensorboard')
    parser.add_argument('--heatmap-path', default='runs/unknown_set_heatmap/'+get_time(),type=str,help='path to store the log data using tensorboard')
    parser.add_argument('--input-img', type=str,help='the path of input image to generate heat map')
    parser.add_argument('--seed', default=2023, type=int, help='global seed to set')

    args = parser.parse_args()

    return args

def getDatasetConfig(config, dataset_name, project_root, preprocess_train, preprocess_test):
    assert dataset_name in ['bird', 'dog', 'bird_unsupervised', 'dog_unsupervised', 'bird_cropped', 'dog_cropped', 'bird_square_cropped', 'dog_square_cropped', 'bird_gen', 'dog_gen'], 'No dataset named %s!' % dataset_name    # check the dataset name
    dataset = dict()
    dataloaders = dict()
    dataset_sizes = dict()
    if dataset_name == 'bird':
        dataset = {
            'train_set' : BirdsDataset(root_dir=os.path.join(project_root, "data/CUB-200-2011/CUB_200_2011"), split="train",transform=preprocess_train),
            'val_set' : BirdsDataset(root_dir=os.path.join(project_root, "data/CUB-200-2011/CUB_200_2011"), split="test", transform=preprocess_test),
            'test_set' : BirdsDataset(root_dir=os.path.join(project_root, "data/CUB-200-2011/CUB_200_2011"), split="test", transform=preprocess_test),
        }
        dataloaders = {
            'train': DataLoader(dataset['train_set'], batch_size=config.batch_size, shuffle=True, num_workers=8,worker_init_fn=_init_fn, drop_last=True),
            'val': DataLoader(dataset['val_set'], batch_size=config.batch_size, shuffle=True, num_workers=8,worker_init_fn=_init_fn, drop_last=True),
            'test': DataLoader(dataset['test_set'], batch_size=config.batch_size, shuffle=True, num_workers=8,worker_init_fn=_init_fn, drop_last=True)
        }
        dataset_sizes = {
            'train': len(dataset['train_set']),
            'val': len(dataset['val_set']),
            'test': len(dataset['test_set'])
        }
        dataset_classes = 200
    elif dataset_name == "bird_unsupervised":
        dataset = {
            'train_set' : UnsupervisedBirdsDataset(root_dir=os.path.join(project_root, "data/CUB-200-2011/CUB_200_2011"), split="train",transform=preprocess_train),
            'val_set' : UnsupervisedBirdsDataset(root_dir=os.path.join(project_root, "data/CUB-200-2011/CUB_200_2011"), split="test", transform=preprocess_test),
            'test_set' : UnsupervisedBirdsDataset(root_dir=os.path.join(project_root, "data/CUB-200-2011/CUB_200_2011"), split="test", transform=preprocess_test),
        }
        dataloaders = {
            'train': DataLoader(dataset['train_set'], batch_size=config.batch_size, shuffle=True, num_workers=8,worker_init_fn=_init_fn, drop_last=True),
            'val': DataLoader(dataset['val_set'], batch_size=config.batch_size, shuffle=True, num_workers=8,worker_init_fn=_init_fn, drop_last=True),
            'test': DataLoader(dataset['test_set'], batch_size=config.batch_size, shuffle=True, num_workers=8,worker_init_fn=_init_fn, drop_last=True)
        }
        dataset_sizes = {
            'train': len(dataset['train_set']),
            'val': len(dataset['val_set']),
            'test': len(dataset['test_set'])
        }
        dataset_classes = 200
    elif dataset_name == "bird_cropped":
        dataset = {
            'train_set' : CroppedBirdsDataset(root_dir=os.path.join(project_root, "data/CUB-200-2011/CUB_200_2011"), split="train",transform=preprocess_train),
            'val_set' : CroppedBirdsDataset(root_dir=os.path.join(project_root, "data/CUB-200-2011/CUB_200_2011"), split="test", transform=preprocess_test),
            'test_set' : CroppedBirdsDataset(root_dir=os.path.join(project_root, "data/CUB-200-2011/CUB_200_2011"), split="test", transform=preprocess_test),
        }
        dataloaders = {
            'train': DataLoader(dataset['train_set'], batch_size=config.batch_size, shuffle=True, num_workers=8,worker_init_fn=_init_fn, drop_last=True),
            'val': DataLoader(dataset['val_set'], batch_size=config.batch_size, shuffle=True, num_workers=8,worker_init_fn=_init_fn, drop_last=True),
            'test': DataLoader(dataset['test_set'], batch_size=config.batch_size, shuffle=True, num_workers=8,worker_init_fn=_init_fn, drop_last=True)
        }
        dataset_sizes = {
            'train': len(dataset['train_set']),
            'val': len(dataset['val_set']),
            'test': len(dataset['test_set'])
        }
        dataset_classes = 200
    elif dataset_name == "bird_square_cropped":
        dataset = {
            'train_set' : SquareCroppedBirdsDataset(root_dir=os.path.join(project_root, "data/CUB-200-2011/CUB_200_2011"), split="train",transform=preprocess_train),
            'val_set' : SquareCroppedBirdsDataset(root_dir=os.path.join(project_root, "data/CUB-200-2011/CUB_200_2011"), split="test", transform=preprocess_test),
            'test_set' : SquareCroppedBirdsDataset(root_dir=os.path.join(project_root, "data/CUB-200-2011/CUB_200_2011"), split="test", transform=preprocess_test),
        }
        dataloaders = {
            'train': DataLoader(dataset['train_set'], batch_size=config.batch_size, shuffle=True, num_workers=8,worker_init_fn=_init_fn, drop_last=True),
            'val': DataLoader(dataset['val_set'], batch_size=config.batch_size, shuffle=True, num_workers=8,worker_init_fn=_init_fn, drop_last=True),
            'test': DataLoader(dataset['test_set'], batch_size=config.batch_size, shuffle=True, num_workers=8,worker_init_fn=_init_fn, drop_last=True)
        }
        dataset_sizes = {
            'train': len(dataset['train_set']),
            'val': len(dataset['val_set']),
            'test': len(dataset['test_set'])
        }
        dataset_classes = 200
    elif dataset_name == "bird_gen":
        dataset = {
            'train_set' : GenBirdsDataset(root_dir=os.path.join(project_root, "data/CUB-200-2011/CUB_200_2011"), split="train",transform=preprocess_train),
            'val_set' : GenBirdsDataset(root_dir=os.path.join(project_root, "data/CUB-200-2011/CUB_200_2011"), split="test", transform=preprocess_test),
            'test_set' : GenBirdsDataset(root_dir=os.path.join(project_root, "data/CUB-200-2011/CUB_200_2011"), split="test", transform=preprocess_test),
        }
        dataloaders = {
            'train': DataLoader(dataset['train_set'], batch_size=config.batch_size, shuffle=True, num_workers=8,worker_init_fn=_init_fn, drop_last=True),
            'val': DataLoader(dataset['val_set'], batch_size=config.batch_size, shuffle=True, num_workers=8,worker_init_fn=_init_fn, drop_last=True),
            'test': DataLoader(dataset['test_set'], batch_size=config.batch_size, shuffle=True, num_workers=8,worker_init_fn=_init_fn, drop_last=True)
        }
        dataset_sizes = {
            'train': len(dataset['train_set']),
            'val': len(dataset['val_set']),
            'test': len(dataset['test_set'])
        }
        dataset_classes = 200
    elif dataset_name == "dog":
        dataset = {
            'train_set' : StanfordDogsDataset(root_dir=os.path.join(project_root, "data/StanfordDogs"), split="train",transform=preprocess_train),
            'val_set' : StanfordDogsDataset(root_dir=os.path.join(project_root, "data/StanfordDogs"), split="test", transform=preprocess_test),
            'test_set' : StanfordDogsDataset(root_dir=os.path.join(project_root, "data/StanfordDogs"), split="test", transform=preprocess_test),
        }
        dataloaders = {
            'train': DataLoader(dataset['train_set'], batch_size=config.batch_size, shuffle=True, num_workers=8,worker_init_fn=_init_fn, drop_last=True),
            'val': DataLoader(dataset['val_set'], batch_size=config.batch_size, shuffle=True, num_workers=8,worker_init_fn=_init_fn, drop_last=True),
            'test': DataLoader(dataset['test_set'], batch_size=config.batch_size, shuffle=True, num_workers=8,worker_init_fn=_init_fn, drop_last=True)
        }
        dataset_sizes = {
            'train': len(dataset['train_set']),
            'val': len(dataset['val_set']),
            'test': len(dataset['test_set'])
        }
        dataset_classes = 120
    elif dataset_name == "dog_unsupervised":
        dataset = {
            'train_set' : UnsupervisedStanfordDogsDataset(root_dir=os.path.join(project_root, "data/StanfordDogs"), split="train",transform=preprocess_train),
            'val_set' : UnsupervisedStanfordDogsDataset(root_dir=os.path.join(project_root, "data/StanfordDogs"), split="test", transform=preprocess_test),
            'test_set' : UnsupervisedStanfordDogsDataset(root_dir=os.path.join(project_root, "data/StanfordDogs"), split="test", transform=preprocess_test),
        }
        dataloaders = {
            'train': DataLoader(dataset['train_set'], batch_size=config.batch_size, shuffle=True, num_workers=8,worker_init_fn=_init_fn, drop_last=True),
            'val': DataLoader(dataset['val_set'], batch_size=config.batch_size, shuffle=True, num_workers=8,worker_init_fn=_init_fn, drop_last=True),
            'test': DataLoader(dataset['test_set'], batch_size=config.batch_size, shuffle=True, num_workers=8,worker_init_fn=_init_fn, drop_last=True)
        }
        dataset_sizes = {
            'train': len(dataset['train_set']),
            'val': len(dataset['val_set']),
            'test': len(dataset['test_set'])
        }
        dataset_classes = 120
    elif dataset_name == "dog_cropped":
        dataset = {
            'train_set' : CroppedStanfordDogsDataset(root_dir=os.path.join(project_root, "data/StanfordDogs"), split="train",transform=preprocess_train),
            'val_set' : CroppedStanfordDogsDataset(root_dir=os.path.join(project_root, "data/StanfordDogs"), split="test", transform=preprocess_test),
            'test_set' : CroppedStanfordDogsDataset(root_dir=os.path.join(project_root, "data/StanfordDogs"), split="test", transform=preprocess_test),
        }
        dataloaders = {
            'train': DataLoader(dataset['train_set'], batch_size=config.batch_size, shuffle=True, num_workers=8,worker_init_fn=_init_fn, drop_last=True),
            'val': DataLoader(dataset['val_set'], batch_size=config.batch_size, shuffle=True, num_workers=8,worker_init_fn=_init_fn, drop_last=True),
            'test': DataLoader(dataset['test_set'], batch_size=config.batch_size, shuffle=True, num_workers=8,worker_init_fn=_init_fn, drop_last=True)
        }
        dataset_sizes = {
            'train': len(dataset['train_set']),
            'val': len(dataset['val_set']),
            'test': len(dataset['test_set'])
        }
        dataset_classes = 120
    elif dataset_name == "dog_square_cropped":
        dataset = {
            'train_set' : SquareCroppedStanfordDogsDataset(root_dir=os.path.join(project_root, "data/StanfordDogs"), split="train",transform=preprocess_train),
            'val_set' : SquareCroppedStanfordDogsDataset(root_dir=os.path.join(project_root, "data/StanfordDogs"), split="test", transform=preprocess_test),
            'test_set' : SquareCroppedStanfordDogsDataset(root_dir=os.path.join(project_root, "data/StanfordDogs"), split="test", transform=preprocess_test),
        }
        dataloaders = {
            'train': DataLoader(dataset['train_set'], batch_size=config.batch_size, shuffle=True, num_workers=8,worker_init_fn=_init_fn, drop_last=True),
            'val': DataLoader(dataset['val_set'], batch_size=config.batch_size, shuffle=True, num_workers=8,worker_init_fn=_init_fn, drop_last=True),
            'test': DataLoader(dataset['test_set'], batch_size=config.batch_size, shuffle=True, num_workers=8,worker_init_fn=_init_fn, drop_last=True)
        }
        dataset_sizes = {
            'train': len(dataset['train_set']),
            'val': len(dataset['val_set']),
            'test': len(dataset['test_set'])
        }
        dataset_classes = 120
    elif dataset_name == "dog_gen":
        dataset = {
            'train_set' : GenStanfordDogsDataset(root_dir=os.path.join(project_root, "data/StanfordDogs"), split="train",transform=preprocess_train),
            'val_set' : GenStanfordDogsDataset(root_dir=os.path.join(project_root, "data/StanfordDogs"), split="test", transform=preprocess_test),
            'test_set' : GenStanfordDogsDataset(root_dir=os.path.join(project_root, "data/StanfordDogs"), split="test", transform=preprocess_test),
        }
        dataloaders = {
            'train': DataLoader(dataset['train_set'], batch_size=config.batch_size, shuffle=True, num_workers=8,worker_init_fn=_init_fn, drop_last=True),
            'val': DataLoader(dataset['val_set'], batch_size=config.batch_size, shuffle=True, num_workers=8,worker_init_fn=_init_fn, drop_last=True),
            'test': DataLoader(dataset['test_set'], batch_size=config.batch_size, shuffle=True, num_workers=8,worker_init_fn=_init_fn, drop_last=True)
        }
        dataset_sizes = {
            'train': len(dataset['train_set']),
            'val': len(dataset['val_set']),
            'test': len(dataset['test_set'])
        }
        dataset_classes = 120
    return dataset, dataloaders, dataset_sizes, dataset_classes

if __name__ == '__main__':
    config = getConfig()
    config = vars(config)

    def preprocess(image):
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
    
    dataset, dataloaders, dataset_sizes, dataset_classes = getDatasetConfig(config,config["dataset"],"/mnt/sda/2022-0526/home/scc/zty/code/ANN/finalproject/new_test_project",preprocess,preprocess)
    # for k,v in config.items():
    #     print(k,v)
    # config.
    print(config)
    print(dataset, dataloaders, dataset_sizes, dataset_classes)
