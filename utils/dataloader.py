import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import scipy.io

class StanfordDogsDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        if split == "train":
            data_split_file = os.path.join(root_dir, "train_list.mat")
        elif split == "test":
            data_split_file = os.path.join(root_dir, "test_list.mat")
        else:
            raise ValueError("Invalid split type. Expected 'train' or 'test', but got {}".format(split))

        data_split = scipy.io.loadmat(data_split_file)
        self.file_list = data_split["file_list"]
        temp_labels = data_split["labels"].squeeze() - 1
        self.labels = np.ndarray(shape=temp_labels.size,dtype=np.int64)
        for i in range(temp_labels.size):
            self.labels[i] = temp_labels[i]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, "Images", self.file_list[idx][0][0])
        img = Image.open(img_name).convert("RGB")

        if self.transform:
            img = self.transform(img)

        label = self.labels[idx]
        return img, label
    
class SquareCroppedStanfordDogsDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        if split == "train":
            data_split_file = os.path.join(root_dir, "train_list.mat")
        elif split == "test":
            data_split_file = os.path.join(root_dir, "test_list.mat")
        else:
            raise ValueError("Invalid split type. Expected 'train' or 'test', but got {}".format(split))

        data_split = scipy.io.loadmat(data_split_file)
        self.file_list = data_split["file_list"]
        temp_labels = data_split["labels"].squeeze() - 1
        self.labels = np.ndarray(shape=temp_labels.size,dtype=np.int64)
        for i in range(temp_labels.size):
            self.labels[i] = temp_labels[i]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, "Images", self.file_list[idx][0][0])
        img_name = img_name.replace("Images", "squareCroppedImages")
        img = Image.open(img_name).convert("RGB")

        if self.transform:
            img = self.transform(img)

        label = self.labels[idx]
        return img, label
    
class CroppedStanfordDogsDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        if split == "train":
            data_split_file = os.path.join(root_dir, "train_list.mat")
        elif split == "test":
            data_split_file = os.path.join(root_dir, "test_list.mat")
        else:
            raise ValueError("Invalid split type. Expected 'train' or 'test', but got {}".format(split))

        data_split = scipy.io.loadmat(data_split_file)
        self.file_list = data_split["file_list"]
        temp_labels = data_split["labels"].squeeze() - 1
        self.labels = np.ndarray(shape=temp_labels.size,dtype=np.int64)
        for i in range(temp_labels.size):
            self.labels[i] = temp_labels[i]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, "Images", self.file_list[idx][0][0])
        img_name = img_name.replace("Images", "croppedImages")
        img = Image.open(img_name).convert("RGB")

        if self.transform:
            img = self.transform(img)

        label = self.labels[idx]
        return img, label

class UnsupervisedStanfordDogsDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        if split == "train":
            data_split_file = os.path.join(root_dir, "train_list.mat")
        elif split == "test":
            data_split_file = os.path.join(root_dir, "test_list.mat")
        else:
            raise ValueError("Invalid split type. Expected 'train' or 'test', but got {}".format(split))

        data_split = scipy.io.loadmat(data_split_file)
        self.file_list = data_split["file_list"]
        temp_labels = data_split["labels"].squeeze() - 1
        self.labels = np.ndarray(shape=temp_labels.size,dtype=np.int64)
        for i in range(temp_labels.size):
            self.labels[i] = temp_labels[i]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, "Images", self.file_list[idx][0][0])
        imgL = Image.open(img_name).convert("RGB")
        imgR = Image.open(img_name).convert("RGB")

        if self.transform:
            imgL = self.transform(imgL)
            imgR = self.transform(imgR)

        label = self.labels[idx]
        return imgL, imgR, label

class GenStanfordDogsDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        if split == "train":
            data_split_file = os.path.join(root_dir, "train_list.mat")
        elif split == "test":
            data_split_file = os.path.join(root_dir, "test_list.mat")
        else:
            raise ValueError("Invalid split type. Expected 'train' or 'test', but got {}".format(split))

        data_split = scipy.io.loadmat(data_split_file)
        self.file_list = data_split["file_list"]
        self.labels = data_split["labels"].squeeze() - 1
        gen_split_file = os.path.join(root_dir, "gen_list.txt")
        # 读取 gen_list.txt 文件
        with open(gen_split_file, 'r') as file:
            lines = file.readlines()

        # 初始化两个空的 numpy.ndarray
        arr1 = np.empty(1,dtype='<U100')
        arr2 = np.empty(1,dtype=object)
        arr3 = np.empty((len(lines),1),dtype=object)
        arr4 = np.empty(len(lines),dtype=np.uint8)
        # 遍历文件的每一行
        for i, line in enumerate(lines):
            # 去除空白符并按空格分割字符串
            parts = line.strip().split(' ')
            arr1 = np.empty(1,dtype='<U'+str(len(parts[0])))
            arr1[0] = parts[0]
            arr2[0] = arr1.copy()
            arr3[i] = arr2.copy()
            arr4[i] = int(parts[1])
        self.file_list = np.concatenate((self.file_list,arr3))
        temp_labels = np.concatenate((self.labels,arr4))
        self.labels = np.ndarray(shape=temp_labels.size,dtype=np.int64)
        for i in range(temp_labels.size):
            self.labels[i] = temp_labels[i]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, "genImages", self.file_list[idx][0][0])
        img = Image.open(img_name).convert("RGB")

        if self.transform:
            img = self.transform(img)

        label = self.labels[idx]
        return img, label

class BirdsDataset(Dataset):
    def __init__(self, root_dir, split, transform=None):
        assert split in {"train", "test"}, "Invalid split! Please choose between 'train' and 'test'"
        
        self.root_dir = root_dir
        self.transform = transform
        self.images_df = pd.read_csv(os.path.join(root_dir, 'images.txt'), sep=' ', names=['image_id', 'image_name'])
        self.labels_df = pd.read_csv(os.path.join(root_dir, 'image_class_labels.txt'), sep=' ', names=['image_id', 'class_id'])
        self.train_test_df = pd.read_csv(os.path.join(root_dir, 'train_test_split.txt'), sep=' ', names=['image_id', 'is_training_image'])

        self.data_info = pd.merge(self.images_df, self.labels_df, on='image_id')
        self.data_info = pd.merge(self.data_info, self.train_test_df, on='image_id')
        
        if split == "train":
            self.data_info = self.data_info[self.data_info.is_training_image == 1]
        else: # test
            self.data_info = self.data_info[self.data_info.is_training_image == 0]

        self.data_info.reset_index(drop=True, inplace=True)
        
    def __len__(self):
        return len(self.data_info)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, 'images', self.data_info.iloc[idx]['image_name'])
        image = Image.open(img_path).convert('RGB')
        label = self.data_info.iloc[idx]['class_id']

        if self.transform:
            image = self.transform(image)

        return image, label - 1 
    
class CroppedBirdsDataset(Dataset):
    def __init__(self, root_dir, split, transform=None):
        assert split in {"train", "test"}, "Invalid split! Please choose between 'train' and 'test'"
        
        self.root_dir = root_dir
        self.transform = transform
        self.images_df = pd.read_csv(os.path.join(root_dir, 'images.txt'), sep=' ', names=['image_id', 'image_name'])
        self.labels_df = pd.read_csv(os.path.join(root_dir, 'image_class_labels.txt'), sep=' ', names=['image_id', 'class_id'])
        self.train_test_df = pd.read_csv(os.path.join(root_dir, 'train_test_split.txt'), sep=' ', names=['image_id', 'is_training_image'])

        self.data_info = pd.merge(self.images_df, self.labels_df, on='image_id')
        self.data_info = pd.merge(self.data_info, self.train_test_df, on='image_id')
        
        if split == "train":
            self.data_info = self.data_info[self.data_info.is_training_image == 1]
        else: # test
            self.data_info = self.data_info[self.data_info.is_training_image == 0]

        self.data_info.reset_index(drop=True, inplace=True)
        
    def __len__(self):
        return len(self.data_info)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, 'croppedimages', self.data_info.iloc[idx]['image_name'])
        image = Image.open(img_path).convert('RGB')
        label = self.data_info.iloc[idx]['class_id']

        if self.transform:
            image = self.transform(image)

        return image, label - 1

class SquareCroppedBirdsDataset(Dataset):
    def __init__(self, root_dir, split, transform=None):
        assert split in {"train", "test"}, "Invalid split! Please choose between 'train' and 'test'"
        
        self.root_dir = root_dir
        self.transform = transform
        self.images_df = pd.read_csv(os.path.join(root_dir, 'images.txt'), sep=' ', names=['image_id', 'image_name'])
        self.labels_df = pd.read_csv(os.path.join(root_dir, 'image_class_labels.txt'), sep=' ', names=['image_id', 'class_id'])
        self.train_test_df = pd.read_csv(os.path.join(root_dir, 'train_test_split.txt'), sep=' ', names=['image_id', 'is_training_image'])

        self.data_info = pd.merge(self.images_df, self.labels_df, on='image_id')
        self.data_info = pd.merge(self.data_info, self.train_test_df, on='image_id')
        
        if split == "train":
            self.data_info = self.data_info[self.data_info.is_training_image == 1]
        else: # test
            self.data_info = self.data_info[self.data_info.is_training_image == 0]

        self.data_info.reset_index(drop=True, inplace=True)
        
    def __len__(self):
        return len(self.data_info)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, 'squarecroppedimages', self.data_info.iloc[idx]['image_name'])
        image = Image.open(img_path).convert('RGB')
        label = self.data_info.iloc[idx]['class_id']

        if self.transform:
            image = self.transform(image)

        return image, label - 1 

class UnsupervisedBirdsDataset(Dataset):
    def __init__(self, root_dir, split, transform=None):
        assert split in {"train", "test"}, "Invalid split! Please choose between 'train' and 'test'"
        
        self.root_dir = root_dir
        self.transform = transform
        self.images_df = pd.read_csv(os.path.join(root_dir, 'images.txt'), sep=' ', names=['image_id', 'image_name'])
        self.labels_df = pd.read_csv(os.path.join(root_dir, 'image_class_labels.txt'), sep=' ', names=['image_id', 'class_id'])
        self.train_test_df = pd.read_csv(os.path.join(root_dir, 'train_test_split.txt'), sep=' ', names=['image_id', 'is_training_image'])

        self.data_info = pd.merge(self.images_df, self.labels_df, on='image_id')
        self.data_info = pd.merge(self.data_info, self.train_test_df, on='image_id')
        
        if split == "train":
            self.data_info = self.data_info[self.data_info.is_training_image == 1]
        else: # test
            self.data_info = self.data_info[self.data_info.is_training_image == 0]

        self.data_info.reset_index(drop=True, inplace=True)
        
    def __len__(self):
        return len(self.data_info)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, 'images', self.data_info.iloc[idx]['image_name'])
        imageL = Image.open(img_path).convert('RGB')
        imageR = Image.open(img_path).convert('RGB')
        label = self.data_info.iloc[idx]['class_id']

        if self.transform:
            imageL = self.transform(imageL)
            imageR = self.transform(imageR)

        return imageL, imageR, label - 1 

class GenBirdsDataset(Dataset):
    def __init__(self, root_dir, split, transform=None):
        assert split in {"train", "test"}, "Invalid split! Please choose between 'train' and 'test'"
        
        self.root_dir = root_dir
        self.transform = transform
        self.images_df = pd.read_csv(os.path.join(root_dir, 'images.txt'), sep=' ', names=['image_id', 'image_name'])
        self.labels_df = pd.read_csv(os.path.join(root_dir, 'image_class_labels.txt'), sep=' ', names=['image_id', 'class_id'])
        self.train_test_df = pd.read_csv(os.path.join(root_dir, 'train_test_split.txt'), sep=' ', names=['image_id', 'is_training_image'])
        self.gen_df = pd.read_csv(os.path.join(root_dir, 'gen_list.txt'), sep=' ', names=['image_name', 'class_id', 'is_training_image'])
        start_idx = self.images_df.__len__()+1
        end_idx = start_idx + self.gen_df.__len__()
        idx_list = np.arange(start=start_idx,stop=end_idx)
        self.gen_df.insert(loc=0,column='image_id',value=idx_list)
        
        self.data_info = pd.merge(self.images_df, self.labels_df, on='image_id')
        self.data_info = pd.merge(self.data_info, self.train_test_df, on='image_id')
        self.data_info = pd.concat((self.data_info,self.gen_df))
        self.data_info.reset_index(drop=True, inplace=True)

        if split == "train":
            self.data_info = self.data_info[self.data_info.is_training_image == 1]
        else: # test
            self.data_info = self.data_info[self.data_info.is_training_image == 0]

        self.data_info.reset_index(drop=True, inplace=True)
        
    def __len__(self):
        return len(self.data_info)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, 'genimages', self.data_info.iloc[idx]['image_name'])
        image = Image.open(img_path).convert('RGB')
        label = self.data_info.iloc[idx]['class_id']

        if self.transform:
            image = self.transform(image)

        return image, label - 1 
    
if __name__ == "__main__":
    project_root = "/mnt/sda/2022-0526/home/scc/zty/code/ANN/finalproject/new_test_project"
    mytest = GenBirdsDataset(root_dir=os.path.join(project_root, "data/CUB-200-2011/CUB_200_2011"), split="train",transform=None)
    mytest2 = GenStanfordDogsDataset(root_dir=os.path.join(project_root, "data/StanfordDogs"), split="train",transform=None)
    ytest2 = StanfordDogsDataset(root_dir=os.path.join(project_root, "data/StanfordDogs"), split="train",transform=None)