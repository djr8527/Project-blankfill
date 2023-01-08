import os.path
import random

from loss import PixelLoss, SSIM
import cv2
import torch
from tqdm import tqdm
import datetime
import argparse
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from torch import Tensor
import torchvision.transforms as transforms
from torchvision.utils import save_image
from UNet import Discriminator
from UNet import Generator


def setup_seed(seed):
    """
    设置随机种子
    :param seed:
    :return:
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    info = f'Total : {str(total_num / 1000 ** 2)} M, Trainable: {str(trainable_num / 1000 ** 2)} M'
    return info


def get_meshgrid() -> Tensor:
    """
    获取meshgrid，作为Unet的输入
    :return: tensor形式的meshgrid，size:[1,2,256,256]
    """
    transform = transforms.ToTensor()
    meshgrid_1 = (np.ones((256, 256)) * np.arange(256)).astype(np.uint8)
    meshgrid_2 = meshgrid_1.T

    # 两个通道叠加起来,转成张量且最外面增加1维
    meshgrid_0 = np.array([meshgrid_1, meshgrid_2])
    meshgrid_tensor = transform(meshgrid_0).permute(1, 0, 2)[None, :, :, :]

    return meshgrid_tensor


def get_gray(path) -> Tensor:
    """
    获取灰度图
    :param path: 灰度图存放路径
    :return: 返回Tensor
    """

    img = cv2.imread(path, 0)  # 读取灰度图
    if img.shape[1] == 250:  # 井像图 宽度250，转成256
        img = np.hstack((img, np.zeros([256, 6]))).astype(np.uint8)
    transform = transforms.ToTensor()
    return transform(img)[None, :, :, :]  # 变成四维张量


def get_mask() -> Tensor:
    """
    获取掩膜mask
    :return:
    """
    mask = get_gray(args.dataset_dir + '/mask.jpg')
    mask[mask > 0.5] = 1
    mask[mask < 0.5] = 0
    return mask


def train():
    setup_seed(args.seed)

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print("Training on:", device)

    # 定义输入meshgrid/掩膜mask/待修复图像img0
    meshgrid = get_meshgrid().to(device)  # 两通道的meshgrid输入
    mask = get_mask().to(device)  # 掩膜
    img0 = get_gray(args.dataset_dir + '/masked_img.jpg').to(device)  # 真实待修复图像

    # 创建生成器和判别器模型
    gen = Generator().to(device)
    dis = Discriminator().to(device)

    # 模型参数量计算
    print('Generator' + get_parameter_number(gen))
    print('Generator' + get_parameter_number(dis))

    # 定义损失以及优化器
    criterion = nn.MSELoss()
    SSIM_criterion = SSIM(device=device)
    Pixel_criterion = PixelLoss()
    g_optimizer = Adam(gen.parameters(), lr=args.g_lr)
    d_optimizer = Adam(dis.parameters(), lr=args.d_lr)

    for epoch in range(args.epoch_num):
        # 打开训练模式
        gen.train()
        dis.train()

        # 生成的图像img1
        gen_img = gen(meshgrid)
        with torch.no_grad():
            img1 = gen_img * mask  # 用mask之后的img1进行判别

        # =======================================
        # 判别器训练，损失的构建和优化，分为以下两部分
        # 1.真实的图片判断为真
        # 2.虚假的图片判断为假
        # =======================================
        d_optimizer.zero_grad()

        real_output = dis(img0)  # 期望对真实图像img0的判别结果为1
        d_real_loss = criterion(real_output, torch.ones_like(real_output))

        fake_output = dis(img1)  # 期望对生成图像img1的判别结果为0
        d_fake_loss = criterion(fake_output, torch.zeros_like(fake_output))

        d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()
        d_optimizer.step()

        # =======================================
        # 生成器器训练，损失的构建和优化
        # =======================================
        g_optimizer.zero_grad()

        fake_output = dis(img1.detach())  # 期望对生成图像img1的判别结果为1
        Loss_GAN = criterion(fake_output, torch.ones_like(fake_output))
        Loss_Pixel = Pixel_criterion(gen_img, img0, mask)
        Loss_SSIM = 1 - SSIM_criterion(gen_img*mask, img0)
        g_loss = (1 - args.alpha - args.beta) * Loss_GAN + args.beta * Loss_SSIM + args.alpha * Loss_Pixel
        g_loss.backward()
        g_optimizer.step()

        print('epoch-{} generator-loss:{} discriminator-loss:{} ssim-loss:{}'
              .format(epoch, g_loss.item(), d_loss.item(), Loss_SSIM.item()))

        if (epoch == 0) or ((epoch + 1) % 500 == 0):
            save_image(gen_img[0], args.log_dir + '/epoch-{}.jpg'.format(epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--seed', default=4999, type=int, help='随机种子')
    parser.add_argument('--dataset_dir', default='dataset/自然场景实验数据集/1', type=str, help='数据集路径')
    parser.add_argument('--log_dir', default='log/自然多轮迭代输出/1', type=str, help='多轮迭代输出路径')
    parser.add_argument('--g_lr', default=0.01, type=float, help='生成器学习率')
    parser.add_argument('--d_lr', default=0.01, type=float, help='判别器学习率')
    parser.add_argument('--epoch_num', default=2000, type=int, help='迭代次数')
    parser.add_argument('--alpha', default=0.2, type=float, help='L1损失系数')
    parser.add_argument('--beta', default=0.7, type=float, help='SSIM损失系数')

    args = parser.parse_args()

    train()

    # 训练所有的自然场景实验数据集
    # for dir in os.listdir('dataset/自然场景实验数据集'):
    #     args.dataset_dir = os.path.join('dataset/自然场景实验数据集', dir)
    #     args.log_dir = os.path.join('log/自然多轮迭代输出', dir)
    #
    #     if not os.path.exists(args.log_dir):
    #         os.mkdir(args.log_dir)
    #
    #     train()

    # 训练所有的井像静态数据集
    # for dir in os.listdir('dataset/井像静态图数据集'):
    #     args.dataset_dir = os.path.join('dataset/井像静态图数据集', dir)
    #     args.log_dir = os.path.join('log/井像多轮迭代输出', dir)
    #
    #     if not os.path.exists(args.log_dir):
    #         os.mkdir(args.log_dir)
    #
    #     train()
