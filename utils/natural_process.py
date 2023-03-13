import os
import cv2
import numpy as np
from torch import Tensor
import torchvision.transforms as transforms


# 生成自然图像实验的输入meshgrid
def get_meshgrid() -> Tensor:
    """
    获取meshgrid，作为自然图像实验模型Unet的输入
    :return: tensor形式的meshgrid，size:[1,2,256,256]
    """
    transform = transforms.ToTensor()
    meshgrid_1 = (np.ones((256, 256)) * np.arange(256)).astype(np.uint8)
    meshgrid_2 = meshgrid_1.T

    meshgrid_0 = np.array([meshgrid_1, meshgrid_2])
    meshgrid_tensor = transform(meshgrid_0).permute(1, 0, 2)[None, :, :, :]

    return meshgrid_tensor


# 读取带掩膜的图像
def get_masked(path, size=256) -> Tensor:
    img = cv2.imread(os.path.join(path, 'masked_img.jpg'), 0)
    if img.shape[1] == 250:  # 如果宽度250，转成256
        img = np.hstack((img, np.zeros([size, 6]))).astype(np.uint8)
    transform = transforms.ToTensor()
    return transform(img)[None, :, :, :]


# 读取掩膜
def get_mask(path, size=256) -> Tensor:
    img = cv2.imread(os.path.join(path, 'mask.jpg'), 0)
    if img.shape[1] == 250:  # 如果宽度250，转成256
        img = np.hstack((img, np.zeros([size, 6]))).astype(np.uint8)
    transform = transforms.ToTensor()

    mask = transform(img)[None, :, :, :]
    mask[mask > 0.5] = 1
    mask[mask < 0.5] = 0
    return mask


# 获取真实图像的灰度图
def get_gray(path) -> Tensor:
    img = cv2.imread(os.path.join(path, 'gray_img.jpg'), 0)
    if img.shape[1] == 250:  # 如果宽度250，转成256
        img = np.hstack((img, np.zeros([256, 6]))).astype(np.uint8)
    transform = transforms.ToTensor()
    return transform(img)[None, :, :, :]


# 获取合并后的灰度图
def get_merge(path) -> Tensor:
    img = cv2.imread(os.path.join(path, 'merge.jpg'), 0)
    if img.shape[1] == 250:  # 如果宽度250，转成256
        img = np.hstack((img, np.zeros([256, 6]))).astype(np.uint8)
    transform = transforms.ToTensor()
    return transform(img)[None, :, :, :]


# 合并原始图像与补全后图像
def merge(args):
    path0 = args.data_dir_i
    path1 = args.res_dir_i

    mask = cv2.imread(os.path.join(path0, 'mask.jpg'), 0)
    img0 = cv2.imread(os.path.join(path0, 'masked_img.jpg'), 0)
    img1 = cv2.imread(os.path.join(path1, 'epoch-{}.jpg'.format(args.epoch_num-1)), 0)

    mask[mask < 128] = 0
    mask[mask >= 128] = 1

    # 对掩膜取反得到_mask
    _mask = mask - 1
    _mask[_mask == 255] = 1

    img = (img0*mask + img1*_mask).astype(np.uint8)
    cv2.imwrite(os.path.join(path1, 'merge.jpg'), img)
