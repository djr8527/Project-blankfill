from modules import *


class UNet(nn.Module):

    def __init__(self):
        super(UNet, self).__init__()
        self.t = 16
        _filters = [self.t, 2*self.t, 4*self.t, 8*self.t, 16*self.t]
        # 4次下采样
        self.conv_1 = Conv(2, _filters[0])
        self.downSample_1 = DownSampling(_filters[0])

        self.conv_2 = Conv(_filters[0], _filters[1])
        self.downSample_2 = DownSampling(_filters[1])

        self.conv_3 = Conv(_filters[1], _filters[2])
        self.downSample_3 = DownSampling(_filters[2])

        self.conv_4 = Conv(_filters[2], _filters[3])
        self.downSample_4 = DownSampling(_filters[3])

        self.A5 = Conv(_filters[3], _filters[4])

        # 4次上采样
        self.upSample_1 = UpSampling(_filters[4])
        self.upSample_Conv_1 = Conv(_filters[4], _filters[3])

        self.upSample_2 = UpSampling(_filters[3])
        self.upSample_Conv_2 = Conv(_filters[3], _filters[2])

        self.upSample_3 = UpSampling(_filters[2])
        self.upSample_Conv_3 = Conv(_filters[2], _filters[1])

        self.upSample_4 = UpSampling(_filters[1])
        self.upSample_Conv_4 = Conv(_filters[1], _filters[0])

        self.Th = torch.nn.Sigmoid()
        self.pred = torch.nn.Conv2d(self.t, 1, 3, 1, 1)

    def forward(self, x):
        # 下采样部分
        R1 = self.conv_1(x)
        R2 = self.conv_2(self.downSample_1(R1))
        R3 = self.conv_3(self.downSample_2(R2))
        R4 = self.conv_4(self.downSample_3(R3))

        Y1 = self.A5(self.downSample_4(R4))

        # 上采样部分,拼接
        O1 = self.upSample_Conv_1(self.upSample_1(Y1, R4))
        O2 = self.upSample_Conv_2(self.upSample_2(O1, R3))
        O3 = self.upSample_Conv_3(self.upSample_3(O2, R2))
        O4 = self.upSample_Conv_4(self.upSample_4(O3, R1))

        # 输出预测，这里大小跟输入是一致的
        # 可以把下采样时的中间抠出来再进行拼接，这样修改后输出就会更小
        return self.Th(self.pred(O4))
