import os
import cv2
import shutil
import numpy as np
import lasio
import pandas as pd


def building_dataset(src, dst, n):
    """
    主要功能：根据井的las文件转成DataFrame格式，分割成长度合适的井段，构建相应的井像数据集(原图masked_img.jpg+mask.jpg)
    csv 文件行数为rows，列数为cols(w)
    每一井段长度为256*n,最后一井段需要补齐成256的倍数
    每一井段宽度为w,为了下采样需要补齐成32的倍数
    :param src: 井的las文件的目录
    :param dst: 存放井像数据集的目录
    :param n:井段长度为256*n行像素点，
    :return:
    """
    # las文件读取
    try:
        las_data = lasio.read(src)
    except FileNotFoundError:
        print("las文件未找到！请确认las文件路径是否正确！")
        raise

    # 清空dst文件下的子文件夹以及子文件
    try:
        shutil.rmtree(dst)
    except FileNotFoundError:
        # 如果dst文件夹不存在则创建dst文件夹
        if not os.path.exists(dst):
            os.makedirs(dst)

    well = pd.DataFrame(las_data.data, columns=las_data.curves.keys()).fillna(-9999)  # 将空值填充为-9999
    rows = well.shape[0]
    cols = well.shape[1]-1

    blockNum = rows // 256 // n
    # 为了方便下采样需要分别对井段的长度与宽度方向上补充像素进行对齐
    l1 = rows//256 % n                                      # 最后一个井段长度为256*(l1+1),(l1+1) < n
    l2 = 256-rows % 256                                     # 最后一个井段，对于最后的256行像素需要补充的像素行数l2
    l3 = [0 if (cols % 32 == 0) else (32-cols % 32)][0]     # 对于井段宽度需要补充的像素列数l3
    width = cols+l3  # 对齐后的井段宽度

    # 根据 井段数 制作 井数据集以及对应的mask
    for i in range(blockNum):
        path = os.path.join(dst, str(i + 1))                # 各井段存放路径
        if not os.path.exists(path):
            os.makedirs(path)

        # 256*rows的 well
        well_data = well.iloc[i * 256 * n:(i + 1) * 256 * n, 1:]
        well_np = well_data.mask(well_data == -9999.0, 0).to_numpy().astype(np.uint8)
        cv2.imwrite(os.path.join(path, 'masked_img.jpg'), well_np)

        # 256*rows的 mask转成256*width
        mask_data = well_data.mask(well_data != -9999.0, 255)
        mask_data = mask_data.mask(mask_data == -9999.0, 0).to_numpy()
        mask = np.hstack((mask_data, np.zeros([256*n, l3]))).astype(np.uint8)
        cv2.imwrite(os.path.join(path, 'mask.jpg'), mask)
        if l1 != 0 and i+1 == blockNum:
            path = os.path.join(dst, 'last')
            if not os.path.exists(path):
                os.makedirs(path)

            well_data = well.iloc[(i + 1) * 256 * n:, 1:]
            well_np = well_data.mask(well_data == -9999.0, 0).to_numpy().astype(np.uint8)
            well_np = np.vstack((well_np, np.zeros([l2, cols]))).astype(np.uint8)  # 补齐成256的倍数
            cv2.imwrite(os.path.join(path, 'masked_img.jpg'), well_np)

            # 保存最后一个井段的相关补齐数据，即需要补齐的256的块数l1，以及最后一个块需要补齐的行数l2
            df = pd.DataFrame([l1, l2])
            df.to_csv(os.path.join(path, 'last.csv'))

            mask_data = well_data.mask(well_data != -9999.0, 255)
            mask_data_1 = mask_data.mask(mask_data == -9999.0, 0).to_numpy()
            mask_data_2 = np.hstack((mask_data_1, np.zeros([256 * (l1+1) - l2, l3]))).astype(np.uint8)
            mask = np.vstack((mask_data_2, np.zeros([l2, width]))).astype(np.uint8)
            cv2.imwrite(os.path.join(path, 'mask.jpg'), mask)  # 保存路径 'xx/{}/mask.jpg'.format(i)

    return cols, l3


if __name__ == '__main__':
    data_src = '../las/ma3/ma3-stat.las'
    data_store_dst = '../dataset/well/stat/ma3'
    width, l3 = building_dataset(data_src, data_store_dst, 20)
    print(width)
    print(l3)
