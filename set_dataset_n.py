import os
import cv2
import numpy as np
import pandas as pd


# 小型自然场景实验数据集生成
def get_natural_dataset():
    hh162j = pd.read_csv('hh162/hh162j.csv')
    well_data = hh162j.iloc[500:756, 1:]
    # 256*250的 mask转成256*256
    mask_data = well_data.mask(well_data != -9999.0, 255)
    mask_data = mask_data.mask(mask_data == -9999.0, 0).to_numpy()
    mask = np.hstack((mask_data, np.zeros([256, 6]))).astype(np.uint8)

    for i in range(10):
        img = cv2.imread('place365/{}.jpg'.format(i + 1))
        img = cv2.resize(img, (256, 256))

        try:
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            masked_img = cv2.bitwise_and(gray_img, mask)

        except cv2.error as e:
            print('err')

        path = 'dataset/自然场景实验数据集/' + str(i + 1)
        if not os.path.exists(path):
            os.makedirs(path)

        # 注意：CV2默认读取RGB图像，如果读取到的是灰度图，会默认将图层复制三遍。
        cv2.imwrite(path + 'gray_img.jpg', gray_img)
        cv2.imwrite(path + 'masked_img.jpg', masked_img)
        cv2.imwrite(path + 'mask.jpg', mask)


def get_well_dataset():
    hh162j = pd.read_csv('hh162/hh162j.csv')

    for i in range(1024):  # 1024 根据hh162行数算出来的
        path = 'dataset/井像静态图数据集/' + str(i + 1)
        if not os.path.exists(path):
            os.makedirs(path)

        # 256*250的 well
        well_data = hh162j.iloc[i * 256:(i + 1) * 256, 1:]  # 多出来的[maxRows-256:maxRows,1:]
        well = well_data.mask(well_data == -9999.0, 0).to_numpy().astype(np.uint8)
        print(path)
        cv2.imwrite(path + '/masked_img.jpg', well)  # 保存路径 'dataset/井像静态图数据集/{}/masked.jpg'.format(i)

        # 256*250的 mask转成256*256
        mask_data = well_data.mask(well_data != -9999.0, 255)
        mask_data = mask_data.mask(mask_data == -9999.0, 0).to_numpy()
        mask = np.hstack((mask_data, np.zeros([256, 6]))).astype(np.uint8)
        cv2.imwrite(path + '/mask.jpg', mask)  # 保存路径 'dataset/井像静态图数据集/{}/mask.jpg'.format(i)


if __name__ == '__main__':
    # get_natural_dataset()
    get_well_dataset()
