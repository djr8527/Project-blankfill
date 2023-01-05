import os
import cv2
import numpy as np


def merge0(path0, path1):
    """
    对原始图像与修复后图像合并
    :param path0: 存放原始图像的路径
    :param path1: 存放修复后图像 与 合并后图像的路径
    :return:
    """
    mask = cv2.imread(path0 + '/mask.jpg', 0)
    img0 = cv2.imread(path0 + '/masked_img.jpg', 0)
    img1 = cv2.imread(path1 + '/epoch-1999.jpg', 0)

    mask[mask < 128] = 0
    mask[mask >= 128] = 1

    # 对掩膜取反得到_mask
    _mask = mask - 1
    _mask[_mask == 255] = 1

    img = (img0*mask + img1*_mask).astype(np.uint8)
    cv2.imwrite(path1 + '/merge.jpg', img)


def merge1(path0, path1):
    """
    对井像 原始图像与修复后图像合并
    :param path0: 存放原始图像的路径
    :param path1: 存放修复后图像 与 合并后图像的路径
    :return:
    """
    mask = cv2.imread(path0 + '/mask.jpg', 0)
    img0 = cv2.imread(path0 + '/masked_img.jpg', 0)  # 256*250
    img1 = cv2.imread(path1 + '/epoch-1999.jpg', 0)

    mask = np.hsplit(mask, (250,))[0]  # 变回256*250
    img1 = np.hsplit(img1, (250,))[0]  # 变回256*250

    mask[mask < 128] = 0
    mask[mask >= 128] = 1

    # 对掩膜取反得到_mask
    _mask = mask - 1
    _mask[_mask == 255] = 1

    img = (img0*mask + img1*_mask).astype(np.uint8)
    cv2.imwrite(path1 + '/merge.jpg', img)


if __name__ == '__main__':
    # 合并 所有的 待修补图像 和 修补后图像（自然场景图）
    root = os.path.join(os.path.abspath(''), 'dataset/自然场景实验数据集')
    save_dir = os.path.join(os.path.abspath(''), 'log/自然多轮迭代输出')
    for dir in os.listdir(root):
        path0 = os.path.join(root, dir)
        path1 = os.path.join(save_dir, dir)
        merge0(path0, path1)

    # 合并 所有的 待修补图像 和 修补后图像（井像静态图）
    # root = os.path.join(os.path.abspath(''), 'dataset/井像静态图数据集')
    # save_dir = os.path.join(os.path.abspath(''), 'log/井像多轮迭代输出')
    # for dir in os.listdir(root):
    #     path0 = os.path.join(root, dir)
    #     path1 = os.path.join(save_dir, dir)
    #     merge1(path0, path1)

    # # 合并 待修补图像 和 修补后图像（一对）
    # root = os.path.join(os.path.abspath(''), 'dataset/井像静态图数据集')
    # save_dir = os.path.join(os.path.abspath(''), 'log/井像多轮迭代输出')
    # path0 = os.path.join(root, '9')
    # path1 = os.path.join(save_dir, '9')
    # merge1(path0, path1)
