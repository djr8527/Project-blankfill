import torch.nn as nn
from models.UNet import UNet


class Generator(nn.Module):
    """
    生成器
    """
    def __init__(self):
        super(Generator, self).__init__()
        self.net = UNet()  # 使用 UNet 作为生成模型

    def forward(self, x):
        return self.net(x)  # 输出 Tensor size :[1,1,256,256]

    def _initialize_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)


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

    def _initialize_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
