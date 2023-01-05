import torch
import torch.nn as nn
from torch.nn import functional as F


# 基本卷积块
class Conv(nn.Module):
    def __init__(self, C_in, C_out):
        super(Conv, self).__init__()
        self.layer = nn.Sequential(

            nn.Conv2d(C_in, C_out, 3, 1, 1),
            nn.BatchNorm2d(C_out),
            # 防止过拟合
            nn.Dropout(0.3),
            nn.LeakyReLU(),

            nn.Conv2d(C_out, C_out, 3, 1, 1),
            nn.BatchNorm2d(C_out),
            # 防止过拟合
            nn.Dropout(0.4),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.layer(x)


# 下采样模块
class DownSampling(nn.Module):
    def __init__(self, C):
        super(DownSampling, self).__init__()
        self.Down = nn.Sequential(
            # 使用卷积进行2倍的下采样，通道数不变
            nn.Conv2d(C, C, 3, 2, 1),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.Down(x)


# 上采样模块
class UpSampling(nn.Module):

    def __init__(self, C):
        super(UpSampling, self).__init__()
        # 特征图大小扩大2倍，通道数减半
        self.Up = nn.Conv2d(C, C // 2, 1, 1)

    def forward(self, x, r):
        # 使用邻近插值进行下采样
        up = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.Up(up)
        # 拼接，当前上采样的，和之前下采样过程中的
        return torch.cat((x, r), 1)


# 主干网络
class UNet(nn.Module):

    def __init__(self):
        super(UNet, self).__init__()

        # 4次下采样
        #         self.C1 = Conv(3, 64)
        self.C1 = Conv(2, 64)
        self.D1 = DownSampling(64)
        self.C2 = Conv(64, 128)
        self.D2 = DownSampling(128)
        self.C3 = Conv(128, 256)
        self.D3 = DownSampling(256)
        self.C4 = Conv(256, 512)
        self.D4 = DownSampling(512)
        self.C5 = Conv(512, 1024)

        # 4次上采样
        self.U1 = UpSampling(1024)
        self.C6 = Conv(1024, 512)
        self.U2 = UpSampling(512)
        self.C7 = Conv(512, 256)
        self.U3 = UpSampling(256)
        self.C8 = Conv(256, 128)
        self.U4 = UpSampling(128)
        self.C9 = Conv(128, 64)

        self.Th = torch.nn.Sigmoid()
        self.pred = torch.nn.Conv2d(64, 1, 3, 1, 1)

    #         self.pred = torch.nn.Conv2d(64, 3, 3, 1, 1)

    def forward(self, x):
        # 下采样部分
        R1 = self.C1(x)
        R2 = self.C2(self.D1(R1))
        R3 = self.C3(self.D2(R2))
        R4 = self.C4(self.D3(R3))
        Y1 = self.C5(self.D4(R4))

        # 上采样部分
        # 上采样的时候需要拼接起来
        O1 = self.C6(self.U1(Y1, R4))
        O2 = self.C7(self.U2(O1, R3))
        O3 = self.C8(self.U3(O2, R2))
        O4 = self.C9(self.U4(O3, R1))

        # 输出预测，这里大小跟输入是一致的
        # 可以把下采样时的中间抠出来再进行拼接，这样修改后输出就会更小
        return self.Th(self.pred(O4))


class Discriminator(nn.Module):
    """
    判别器
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            # Conv1 H/W:256->128
            nn.Conv2d(1, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),


            # Conv2 H/W:128->64
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),


            # Conv3 H/W:64->32
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),


            # Conv4 H/W:32->16
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),

            # Conv5 H/W:16->8
            nn.Conv2d(256, 512, 3, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),

            # Conv6 H/W 8->4
            nn.Conv2d(512, 1, 3, 2, 1),
            nn.Sigmoid()  # 4×4
        )

    def forward(self, img):
        return self.net(img)


class Generator(nn.Module):
    """
    生成器
    """
    def __init__(self):
        super(Generator, self).__init__()
        self.net = UNet()  # 使用 UNet 作为生成模型

    def forward(self, x):
        return self.net(x)  # 输出 Tensor size :[1,1,256,256]


if __name__ == '__main__':
    import numpy as np
    from torch import Tensor
    import torchvision.transforms as transforms

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

    meshgrid = get_meshgrid()

    dis = Discriminator()
    gen = Generator()

    fake_img = gen(meshgrid)
    print(fake_img.size())
    print(dis(fake_img))
