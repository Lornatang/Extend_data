"""
# author: shiyipaisizuo
# contact: shiyipaisizuo@gmail.com
# file: train.py
# time: 2018/9/01 07:27
# license: MIT
"""

import os
import argparse

import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
parser = argparse.ArgumentParser("GAN for mnist.\n")
parser.add_argument('--latent_size', type=int, default=100)  # noise size
parser.add_argument('--hidden_size', type=int, default=256)  # hidden size
parser.add_argument('--image_size', type=int, default=28 * 28)  # image size width * hight
parser.add_argument('--num_epochs', type=int, default=100)  # train epochs
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--out_dir', type=str, default='generate')  # generate data
parser.add_argument('--data_dir', type=str, default='../data/mnist')  # generate mnist data
parser.add_argument('--model_path', type=str, default='../../models/pytorch/gan/')
args = parser.parse_args()

# Create a directory if not exists
if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

# Image processing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5),  # 3 for RGB channels
                         std=(0.5, 0.5, 0.5))])

# MNIST dataset
mnist = torchvision.datasets.MNIST(root=args.data_dir,
                                   train=True,
                                   transform=transform,
                                   download=True)

# Data loader
data_loader = torch.utils.data.DataLoader(dataset=mnist,
                                          batch_size=args.batch_size,
                                          shuffle=True)

# Discriminator
D = nn.Sequential(
    nn.Linear(args.image_size, args.hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(args.hidden_size, args.hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(args.hidden_size, 1),
    nn.Sigmoid()
)

# Generator
G = nn.Sequential(
    nn.Linear(args.latent_size, args.hidden_size),
    nn.ReLU(),
    nn.Linear(args.hidden_size, args.hidden_size),
    nn.ReLU(),
    nn.Linear(args.hidden_size, args.image_size),
    nn.Tanh()
)

# Device setting
D = D.to(device)
G = G.to(device)
# D = torch.load(args.model_path + 'D.pth')
# G = torch.load(args.model_path + 'G.pth')

# Binary cross entropy loss and optimizer
criterion = nn.BCELoss().to(device)
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)


def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


# Start training
total_step = len(data_loader)
for epoch in range(1, args.num_epochs + 1):
    for i, (images, _) in enumerate(data_loader):
        images = images.reshape(args.batch_size, -1).to(device)

        # Create the labels which are later used as input for the BCE loss
        real_labels = torch.ones(args.batch_size, 1).to(device)
        fake_labels = torch.zeros(args.batch_size, 1).to(device)

        # ================================================================== #
        #                      Train the discriminator                       #
        # ================================================================== #

        # Compute BCE_Loss using real images where BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
        # Second term of the loss is always zero since real_labels == 1
        outputs = D(images)
        d_loss_real = criterion(outputs, real_labels)
        real_score = outputs

        # Compute BCELoss using fake images
        # First term of the loss is always zero since fake_labels == 0
        z = torch.randn(args.batch_size, args.latent_size).to(device)
        fake_images = G(z)
        outputs = D(fake_images)
        d_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs

        # Backprop and optimize
        d_loss = d_loss_real + d_loss_fake
        d_optimizer.zero_grad()
        g_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # ================================================================== #
        #                        Train the generator                         #
        # ================================================================== #

        # Compute loss with fake images
        z = torch.randn(args.batch_size, args.latent_size).to(device)
        fake_images = G(z)
        outputs = D(fake_images)

        # We train G to maximize log(D(G(z)) instead of minimizing log(1-D(G(z)))
        # For the reason, see the last paragraph of section 3. https://arxiv.org/pdf/1406.2661.pdf
        g_loss = criterion(outputs, real_labels)

        # Backprop and optimize
        d_optimizer.zero_grad()
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        if (i + 1) % 200 == 0:
            print(f"Epoch [{epoch}/{args.num_epochs}], Step [{i+1}/{total_step}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}, D(x): {real_score.mean().item():.2f}, D(G(z)): {fake_score.mean().item():.2f}")
            fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)
            save_image(denorm(fake_images), os.path.join(args.out_dir, 'fake_images-{}.jpg'.format(epoch + 1)))

    # Save real images
    if epoch == 1:
        images = images.reshape(images.size(0), 1, 28, 28)
        save_image(denorm(images), os.path.join(args.out_dir, 'real_images.jpg'))

    # Save the model checkpoints
    torch.save(G, '../../models/pytorch/gan/mnist/' + G.pth)
    torch.save(D, '../../models/pytorch/gan/mnist/' + D.pth)
