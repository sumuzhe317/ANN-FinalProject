import shutil
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import utils, datasets, transforms
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image

import torch
import os
import math
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights,ResNeXt101_32X8D_Weights

from utils.dataloader import BirdsDataset, StanfordDogsDataset
from utils.utils import save_checkpoint, _init_fn, set_seed
from utils.config import getConfig, getDatasetConfig

# Spatial size of training images. All images will be resized to this size using a transformer.
image_size = 32

# 自定义排序函数
def sort_by_digits(string):
    # 获取字符串的前八位数字部分并转换为整数进行排序
    digits = int(string[:3])
    return digits

def preprocess_train(image):
    return transforms.Compose([
        transforms.Resize((image_size,image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])(image)

def preprocess_test(image):
    return transforms.Compose([
        transforms.Resize((image_size,image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])(image)

def train():
    # Set random seed for reproducibility
    torch.manual_seed(2023)

    # Root directory for dataset
    dataroot = "/mnt/sda/2022-0526/home/scc/zty/code/ANN/finalproject/new_test_project/data/CUB-200-2011/CUB_200_2011/"

    # Number of workers for dataloader
    workers = 10

    # Batch size during training
    batch_size = 100

    # Number of channels in the training images. For color images this is 3
    nc = 3

    # Number of classes in the training images. For mnist dataset this is 10
    num_classes = 200

    # Size of z latent vector (i.e. size of generator input)
    nz = 10000

    # Size of feature maps in generator
    ngf = 64

    # Size of feature maps in discriminator
    ndf = 64

    # Number of training epochs
    num_epochs = 100

    # Learning rate for optimizers
    lr = 0.0002

    # Beta1 hyperparam for Adam optimizers
    beta1 = 0.5

    # Number of GPUs available. Use 0 for CPU mode.
    ngpu = 1

    all_dataset = dataset['train_set'] + dataset['test_set']
    print(f'Total Size of all_dataset: {len(all_dataset)}')

    dataloader = DataLoader (
        dataset=all_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers
    )

    device = torch.device('cuda:2' if (torch.cuda.is_available() and ngpu > 0) else 'cpu')
    print(device)

    imgs = {}
    for x, y in all_dataset:
        if y not in imgs:
            imgs[y] = []
        elif len(imgs[y])!=1:
            imgs[y].append(x)
        elif sum(len(imgs[key]) for key in imgs)==200:
            break
        else:
            continue
            
    imgs = sorted(imgs.items(), key=lambda x:x[0])
    imgs = [torch.stack(item[1], dim=0) for item in imgs]
    imgs = torch.cat(imgs, dim=0)

    plt.figure(figsize=(10,20))
    plt.title("Training Images")
    plt.axis('off')
    imgs = utils.make_grid(imgs, nrow=10)
    plt.imshow(imgs.permute(1, 2, 0)*0.5+0.5)
    os.makedirs("legacy/task4/figures", exist_ok=True)
    os.makedirs("legacy/task4/figures/bird", exist_ok=True)
    os.makedirs("checkpoint/task4", exist_ok=True)
    os.makedirs("checkpoint/task4/bird", exist_ok=True)
    os.makedirs("checkpoint/task4/bird/gan", exist_ok=True)
    plt.savefig('legacy/task4/figures/bird/bird_true.jpg')

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    class Generator(nn.Module):
        def __init__(self, ngpu):
            super(Generator, self).__init__()
            self.ngpu = ngpu
            self.image = nn.Sequential(
                # state size. (nz) x 1 x 1
                nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True)
                # state size. (ngf*4) x 4 x 4
            )
            self.label = nn.Sequential(
                # state size. (num_classes) x 1 x 1
                nn.ConvTranspose2d(num_classes, ngf * 4, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True)
                # state size. (ngf*4) x 4 x 4
            )
            self.main = nn.Sequential(
                # state size. (ngf*8) x 4 x 4
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                # state size. (ngf*4) x 8 x 8
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                # state size. (ngf*2) x 16 x 16
                nn.ConvTranspose2d(ngf*2, nc, 4, 2, 1, bias=False),
                nn.Tanh()
                # state size. (nc) x 32 x 32
            )

        def forward(self, image, label):
            image = self.image(image)
            label = self.label(label)
            incat = torch.cat((image, label), dim=1)
            return self.main(incat)
        
    # Create the generator
    netG = Generator(ngpu).to(device)

    # Handle multi-gpu if desired
    if device.type == 'cuda' and ngpu > 1:
        print("goto gpu")
        netG = nn.DataParallel(netG, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2.
    netG.apply(weights_init)

    class Discriminator(nn.Module):
        def __init__(self, ngpu):
            super(Discriminator, self).__init__()
            self.ngpu = ngpu
            self.image = nn.Sequential(
                # input is (nc) x 32 x 32
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True)
                # state size. (ndf) x 16 x 16
            )
            self.label = nn.Sequential(
                # input is (num_classes) x 32 x 32
                nn.Conv2d(num_classes, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True)
                # state size. (ndf) x 16 x 16
            )
            self.main = nn.Sequential(
                # state size. (ndf*2) x 16 x 16
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*4) x 8 x 8
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*8) x 4 x 4
                nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
                # state size. (1) x 1 x 1
                nn.Sigmoid()
            )

        def forward(self, image, label):
            image = self.image(image)
            label = self.label(label)
            incat = torch.cat((image, label), dim=1)
            return self.main(incat)
        
    # Create the Discriminator
    netD = Discriminator(ngpu).to(device)

    # Handle multi-gpu if desired
    if device.type == 'cuda' and ngpu > 1:
        netD = nn.DataParallel(netD, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2.
    netD.apply(weights_init)

    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Establish convention for real and fake labels during training
    real_label_num = 1.
    fake_label_num = 0.

    # Setup Adam optimizers for both G and D
    optimizerD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # Label one-hot for G
    label_1hots = torch.zeros(num_classes,num_classes)
    for i in range(num_classes):
        label_1hots[i,i] = 1
    label_1hots = label_1hots.view(num_classes,num_classes,1,1).to(device)

    # Label one-hot for D
    label_fills = torch.zeros(num_classes, num_classes, image_size, image_size)
    ones = torch.ones(image_size, image_size)
    for i in range(num_classes):
        label_fills[i][i] = ones
    label_fills = label_fills.to(device)

    # Create batch of latent vectors and laebls that we will use to visualize the progression of the generator
    fixed_noise = torch.randn(num_classes, nz, 1, 1).to(device)
    fixed_label = label_1hots[torch.arange(num_classes).repeat(1).sort().values]
    # print("torch.arange(10)",torch.arange(10))
    # print("torch.arange(10).repeat(10)",torch.arange(10).repeat(10))
    # print("torch.arange(10).repeat(10).sort()",torch.arange(10).repeat(10).sort())
    # print("torch.arange(10).repeat(10).sort().values",torch.arange(10).repeat(10).sort().values)

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    D_x_list = []
    D_z_list = []
    loss_tep = 10

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):

        beg_time = time.time()
        # For each batch in the dataloader
        for i, data in enumerate(dataloader):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            
            # Format batch
            real_image = data[0].to(device)
            b_size = real_image.size(0)

            real_label = torch.full((b_size,), real_label_num).to(device)
            fake_label = torch.full((b_size,), fake_label_num).to(device)
            
            # print(len(data[1]))
            # print(data[1])
            # print(data[1].dtype)
            # print(label_1hots.size())
            # a = input()
            G_label = label_1hots[data[1]]
            D_label = label_fills[data[1]]
            
            # Forward pass real batch through D
            output = netD(real_image, D_label).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, real_label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1).to(device)
            # Generate fake image batch with G
            fake = netG(noise, G_label)
            # Classify all fake batch with D
            output = netD(fake.detach(), D_label).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, fake_label)
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake, D_label).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, real_label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            end_time = time.time()
            run_time = round(end_time-beg_time)
            print(
                f'Epoch: [{epoch+1:0>{len(str(num_epochs))}}/{num_epochs}]',
                f'Step: [{i+1:0>{len(str(len(dataloader)))}}/{len(dataloader)}]',
                f'Loss-D: {errD.item():.4f}',
                f'Loss-G: {errG.item():.4f}',
                f'D(x): {D_x:.4f}',
                f'D(G(z)): [{D_G_z1:.4f}/{D_G_z2:.4f}]',
                f'Time: {run_time}s',
                end='\r'
            )

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())
            
            # Save D(X) and D(G(z)) for plotting later
            D_x_list.append(D_x)
            D_z_list.append(D_G_z2)
            
            # Save the Best Model
            if errG < loss_tep:
                torch.save(netG.state_dict(), 'checkpoint/task4/bird/gan/model.pt')
                loss_tep = errG

        # Check how the generator is doing by saving G's output on fixed_noise and fixed_label
        with torch.no_grad():
            fake = netG(fixed_noise, fixed_label).detach().cpu()
        img_list.append(utils.make_grid(fake, nrow=10))
        
        # Next line
        print()

    plt.figure(figsize=(20, 10))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses[::100], label="G")
    plt.plot(D_losses[::100], label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.axhline(y=0, label="0", c='g') # asymptote
    plt.legend()
    plt.savefig('legacy/task4/figures/bird/loss.jpg')

    plt.figure(figsize=(20, 10))
    plt.title("D(x) and D(G(z)) During Training")
    plt.plot(D_x_list[::100], label="D(x)")
    plt.plot(D_z_list[::100], label="D(G(z))")
    plt.xlabel("iterations")
    plt.ylabel("Probability")
    plt.axhline(y=0.5, label="0.5", c='g') # asymptote
    plt.legend()
    plt.savefig('legacy/task4/figures/bird/dx-dz.jpg')

    # Size of the Figure
    plt.figure(figsize=(20,20))

    # Plot the real images
    plt.subplot(1,2,1)
    plt.axis('off')
    plt.title("Real Images")
    imgs = utils.make_grid(imgs, nrow=10)
    plt.imshow(imgs.permute(1, 2, 0)*0.5+0.5)

    # Load the Best Generative Model
    netG = Generator(0)
    netG.load_state_dict(torch.load('checkpoint/task4/bird/gan/model.pt', map_location=torch.device(device)))
    netG.eval()

    # Generate the Fake Images
    with torch.no_grad():
        fake = netG(fixed_noise.cpu(), fixed_label.cpu())

    # Plot the fake images
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images")
    fake = utils.make_grid(fake, nrow=10)
    plt.imshow(fake.permute(1, 2, 0)*0.5+0.5)

    # Save the comparation result
    plt.savefig('legacy/task4/figures/bird/compare.jpg', bbox_inches='tight')

    # Size of the Figure
    # Create batch of latent vectors and laebls that we will use to visualize the progression of the generator
    import torchvision
    from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize

    source_gen_folder = 'data/CUB-200-2011/CUB_200_2011/images/'
    target_gen_folder = 'data/CUB-200-2011/CUB_200_2011/genimages/'
    relative_target_gen_folder = 'genimages'

    if not os.path.exists(target_gen_folder): #判断所在目录下是否有该文件名的文件夹
        os.makedirs(target_gen_folder) #创建多级目录用mkdirs，单击目录mkdir
    else:
        shutil.rmtree(target_gen_folder)
        os.makedirs(target_gen_folder)
    
    os.system('cp -r '+source_gen_folder+'* '+target_gen_folder)
    sub_folder = os.scandir(source_gen_folder)
    sub_folder = [folder.name for folder in sub_folder]
    gen_file = open(os.path.join(dataroot,'gen_list.txt'),'w')
    sub_folder = sorted(sub_folder, key=sort_by_digits)
    print(sub_folder)

    gen_num = 10
    for now_class in range(num_classes):
        for now_idx in range(gen_num):
            my_fixed_noise = torch.randn(1, nz, 1, 1).to(device)
            my_fixed_label = label_1hots[torch.arange(now_class,now_class+1).repeat(1).sort().values]

            # Load the Best Generative Model
            netG.eval()

            # Generate the Fake Images
            with torch.no_grad():
                fake = netG(my_fixed_noise.cpu(), my_fixed_label.cpu())

            # Save the comparation result
            torchresult_tensor = fake.squeeze()*0.5+0.5
            # print(type(torchresult_tensor))
            torchvision.utils.save_image(torchresult_tensor,'temp/torchresult.jpg')
            img = Image.open('temp/torchresult.jpg')
            resize = transforms.Resize([512,512])
            img = resize(img)
            save_path = os.path.join(target_gen_folder,sub_folder[now_class],'gen-'+str(now_idx)+'.jpg')
            img_file = open(save_path,'w')
            img_file.close()
            img.save(save_path)
            gen_file.write(os.path.join(sub_folder[now_class],'gen-'+str(now_idx)+'.jpg'))
            gen_file.write(' ')
            gen_file.write(str(now_class+1))
            gen_file.write(' ')
            gen_file.write(str(1))
            gen_file.write('\n')
    gen_file.close()


if __name__ == "__main__":
    config = getConfig()
    if config.action == 'train':
        dataset, dataloaders, dataset_sizes, dataset_classes = getDatasetConfig(config=config,dataset_name=config.dataset,project_root=os.getcwd(),preprocess_train=preprocess_train, preprocess_test=preprocess_test)
        train()
    else:
        dataset, dataloaders, dataset_sizes, dataset_classes = getDatasetConfig(config=config,dataset_name=config.dataset,project_root=os.getcwd(),preprocess_train=preprocess_train, preprocess_test=preprocess_test)
        print("only support train")