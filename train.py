import os.path

from UNet import UNet
from loss import PixelLoss
import cv2
import torch
import argparse
import numpy as np
from torch.optim import Adam
from torch import Tensor
import torchvision.transforms as transforms
from torchvision.utils import save_image


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
    mask = get_gray(args.dataset_dir + '/mask.jpg')
    mask[mask > 0.5] = 1
    mask[mask < 0.5] = 0
    return mask


def train():

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    # 定义输入meshgrid/掩膜mask/待修复图像img0
    Input = get_meshgrid().to(device)
    mask = get_mask().to(device)
    img0 = get_gray(args.dataset_dir + '/masked_img.jpg').to(device)

    # 创建模型
    net = UNet().to(device)

    # 定义损失以及优化器
    criterion = PixelLoss()
    optimizer = Adam(net.parameters(), lr=args.lr)

    for epoch in range(args.epoch_num):
        net.train()  # 打开训练模式

        optimizer.zero_grad()

        Output = net(Input)
        loss = criterion(Output, img0, mask)

        loss.backward()
        optimizer.step()

        print('epoch-{} loss:{}'.format(epoch, loss.item()))

        if (epoch == 0) or ((epoch + 1) % 500 == 0):
            save_image(Output[0], args.log_dir + '/epoch-{}.jpg'.format(epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training')

    parser.add_argument('--dataset_dir', default='dataset/自然场景实验数据集/1', type=str, help='数据集路径')
    parser.add_argument('--log_dir', default='log/自然多轮迭代输出/1', type=str, help='多轮迭代输出路径')
    parser.add_argument('--lr', default=0.01, type=float, help='学习率')
    parser.add_argument('--epoch_num', default=2000, type=int, help='迭代次数')

    args = parser.parse_args()

    # train()

    # # 训练所有的自然场景实验数据集
    # for dir in os.listdir('dataset/自然场景实验数据集'):
    #     args.dataset_dir = os.path.join('dataset/自然场景实验数据集', dir)
    #     args.log_dir = os.path.join('log/自然多轮迭代输出', dir)
    #
    #     if not os.path.exists(args.log_dir):
    #         os.mkdir(args.log_dir)
    #
    #     train()

    # 训练所有的井像静态数据集
    for dir in os.listdir('dataset/井像静态图数据集'):
        args.dataset_dir = os.path.join('dataset/井像静态图数据集', dir)
        args.log_dir = os.path.join('log/井像多轮迭代输出', dir)

        if not os.path.exists(args.log_dir):
            os.mkdir(args.log_dir)

        train()
