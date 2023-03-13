import os
import cv2
import pandas as pd
import numpy as np
import lasio
from torch import Tensor
import torchvision.transforms as transforms


# 读取带掩膜的图像masked_img.jpg
def get_masked(args) -> Tensor:
    path = args.data_dir_i  # masked.jpg 路径
    width = args.width  # 井像原宽度
    l3 = args.l3  # 井宽需补齐列数
    size = args.size  # 井段长度256*n

    img = cv2.imread(os.path.join(path, 'masked_img.jpg'), 0)
    if img.shape[1] == width:  # 井像图 宽度width，转成width+l3
        img = np.hstack((img, np.zeros([size, l3]))).astype(np.uint8)
    transform = transforms.ToTensor()
    return transform(img)[None, :, :, :]  # 变成四维张量


# 读取掩膜mask.jpg
def get_mask(args) -> Tensor:
    path = args.data_dir_i  # mask.jpg 路径
    width = args.width  # 井像原宽度
    l3 = args.l3  # 井宽需补齐列数
    size = args.size  # 井段长度256*n

    img = cv2.imread(os.path.join(path, 'mask.jpg'), 0)
    if img.shape[1] == width:  # 井像图 宽度width，转成width+l3
        img = np.hstack((img, np.zeros([size, l3]))).astype(np.uint8)
    transform = transforms.ToTensor()

    mask = transform(img)[None, :, :, :]
    mask[mask > 0.5] = 1
    mask[mask < 0.5] = 0
    return mask


# 对井像 原始图像(有空白带的图像) 与修复后图像（填充的空白带部分）合并
def merge(args):
    path0 = args.data_dir_i  # 带空白带的图像路径
    path1 = args.res_dir_i   # 补全空白带后的图像路径
    width = args.width       # 图像宽度需要补齐列数
    l2 = args.l2             # 最后一个256块需要补齐的像素行数

    mask = cv2.imread(os.path.join(path0, 'mask.jpg'), 0)
    img0 = cv2.imread(os.path.join(path0, 'masked_img.jpg'), 0)  # 256*250
    img1 = cv2.imread(os.path.join(path1, 'epoch-{}.jpg'.format(args.epoch_num-1)), 0)

    mask = np.hsplit(mask, (width,))[0]  # W缩减至width
    if l2 != 256:
        mask = np.vsplit(mask, (-l2,))[0]  # H减少l2

    img1 = np.hsplit(img1, (width,))[0]  # W缩减至width
    if l2 != 256:
        img0 = np.vsplit(img0, (-l2,))[0]  # H减少l2
        img1 = np.vsplit(img1, (-l2,))[0]  # H减少l2

    mask[mask < 128] = 0
    mask[mask >= 128] = 1

    # 对掩膜取反得到_mask
    _mask = mask - 1
    _mask[_mask == 255] = 1

    img = (img0*mask + img1*_mask).astype(np.uint8)
    cv2.imwrite(os.path.join(path1, 'merge.jpg'), img)


# 将补全后的井段结果保存为las文件
def save_las(args):
    # 合并分割补全后的井像图,输出为las文件
    IMG = []
    dir_num = len(os.listdir(args.res_dir)) - 1
    for i in range(dir_num):
        if (i + 1) != dir_num:
            dir = str(i + 1)
        else:
            dir = 'last'

        merge_img = cv2.imread(os.path.join(args.res_dir, dir, 'merge.jpg'), 0)
        IMG.append(merge_img)

    lasFile = lasio.read(args.las_dir)
    cols = lasFile.curves.keys()
    # 带空白带的las文件
    las_before = lasFile.df()
    las_before = las_before.fillna(0)

    # 索引
    tdep = np.array([las_before.index.to_numpy()]).T

    # 补全后的numpy数据
    las_np = np.vstack(IMG).astype(np.float64)

    # 补全空白带后的df数据
    las_after = pd.DataFrame(np.hstack((tdep, las_np)), columns=cols).set_index('TDEP')

    nan_mask = lasFile.df()
    nan_mask[nan_mask.notnull()] = 0
    nan_mask[nan_mask.isnull()] = 1

    # 融合带空白带的las文件与补全图像的空白带部分的df数据
    las_merge = las_before + las_after * nan_mask

    # 修改空白带部分为补全图像，并写入到las文件中，最终的las文件为补全后的图像
    lasFile.set_data_from_df(las_merge)

    las_save_path = os.path.join(args.res_dir, 'las_merge.las')
    with open(las_save_path, mode='w') as f:
        lasFile.write(f, version=2.0)
