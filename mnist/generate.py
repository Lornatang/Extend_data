"""
# author: shiyipaisizuo
# contact: shiyipaisizuo@gmail.com
# file: generate.py
# time: 2018/8/21 15:28
# license: MIT
"""

import argparse
import os

import torch
from torch.utils import data
from torchvision import datasets, transforms
from torchvision.utils import save_image

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Setting hyper-parameters
parser = argparse.ArgumentParser()
parser.add_argument('--img_dir', type=str, default='../data/mnist/',
                    help="""input image path dir.Default: '../data/mnist/'.""")
parser.add_argument('--external_dir', type=str, default='../data/mnist/external_data/',
                    help="""input image path dir.Default: '../data/mnist/external_data/'.""")
parser.add_argument('--noise', type=int, default=100,
                    help="""Data noise. Default: 100.""")
parser.add_argument('--hidden_size', type=int, default=64,
                    help="""Hidden size. Default: 64.""")
parser.add_argument('--batch_size', type=int, default=1,
                    help="""Batch size. Default: 1.""")
parser.add_argument('--lr', type=float, default=2e-4,
                    help="""Train optimizer learning rate. Default: 2e-4.""")
parser.add_argument('--img_size', type=int, default=96,
                    help="""Input image size. Default: 96.""")
parser.add_argument('--max_epochs', type=int, default=50,
                    help="""Max epoch of train of. Default: 50.""")
parser.add_argument('--display_epoch', type=int, default=2,
                    help="""When epochs save image. Default: 2.""")
parser.add_argument('--model_dir', type=str, default='../../models/pytorch/GAN/mnist/',
                    help="""Model save path dir. Default: '../../models/pytorch/GAN/mnist/'.""")
args = parser.parse_args()

# Create a directory if not exists
if not os.path.exists(args.external_dir):
    os.makedirs(args.external_dir)

# Image processing
transform = transforms.Compose([
    transforms.Resize(args.img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])  # 3 for RGB channels

# train dataset
dataset = datasets.MNIST(root=args.img_dir,
                         transform=transform,
                         download=True,
                         train=False)

# Data loader
data_loader = data.DataLoader(dataset=dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              drop_last=True)


if torch.cuda.is_available():
    Generator = torch.load(args.model_dir + 'Generator.pth').to(device)
    Discriminator = torch.load(args.model_dir + 'Discriminator.pth').to(device)
else:
    Generator = torch.load(
        args.model_dir + 'Generator.pth', map_location='cpu')
    Discriminator = torch.load(
        args.model_dir + 'Discriminator.pth', map_location='cpu')


# criterion = nn.BCELoss()
# optimizerG = torch.optim.Adam(Generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
# optimizerD = torch.optim.Adam(Discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))


def main():
    for i, (img, _) in enumerate(data_loader):
        # 固定生成器G，训练鉴别器D
        # optimizerD.zero_grad()
        # # 让D尽可能的把真图片判别为1
        # img = img.to(device)
        # output = Discriminator(img)

        # label.data.fill_(real_label)
        # label = label.to(device)
        # errD_real = criterion(output, label)
        # errD_real.backward()
        # 让D尽可能把假图片判别为0
        # label.data.fill_(fake_label)
        # noise = torch.randn(args.batch_size, args.noise, 1, 1)
        # noise = noise.to(device)
        # 生成假图
        fake = Generator()
        # output = Discriminator(fake.detach())  # 避免梯度传到G，因为G不用更新
        # errD_fake = criterion(output, label)
        # errD_fake.backward()
        # errD = errD_fake + errD_real
        # optimizerD.step()

        # 固定鉴别器D，训练生成器G
        # optimizerG.zero_grad()
        # 让D尽可能把G生成的假图判别为1
        # label.data.fill_(real_label)
        # label = label.to(device)
        # output = Discriminator(fake)
        # errG = criterion(output, label)
        # errG.backward()
        # optimizerG.step()

        save_image(fake.data, f"{args.external_dir}/{i+1}.jpg", normalize=True)

    # Save the model checkpoints
    torch.save(Generator, args.model_dir + 'Generator.pth')
    torch.save(Discriminator, args.model_dir + 'Discriminator.pth')


if __name__ == '__main__':
    main()
