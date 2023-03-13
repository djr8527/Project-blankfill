from models.Encoder_Decoder import Encoder_Decoder
from models.UNet import UNet
from models.UNet_SE import UNet_SE
from models.UNet_ResSE import UNet_ResSE
from models.UNet_CBAM import UNet_CBAM
from models.UNet_CBAM_SE import UNet_CBAM_SE
from models.UNet_CBAM_ResSE import UNet_CBAM_ResSE
from models.GAN import Discriminator,Generator

# 通过模型名称创建模型
def create_Encoder_Decoder():
    return Encoder_Decoder()

def create_UNet():
    return UNet()

def create_UNet_SE():
    return UNet_SE()

def create_UNet_ResSE():
    return UNet_ResSE()

def create_UNet_CBAM():
    return UNet_CBAM()

def create_UNet_CBAM_SE():
    return UNet_CBAM_SE()

def create_UNet_CBAM_ResSE():
    return UNet_CBAM_ResSE()

def create_GAN():
    return Generator(), Discriminator()


func_dict = {
'Encoder_Decoder': create_Encoder_Decoder,
'UNet': create_UNet,
'UNet_SE': create_UNet_SE,
'UNet_ResSE': create_UNet_ResSE,
'UNet_CBAM': create_UNet_CBAM,
'UNet_CBAM_SE': create_UNet_CBAM_SE,
'UNet_CBAM_ResSE': create_UNet_CBAM_ResSE,
'GAN':create_GAN,
}


def get_model_by_name(name):
    return func_dict[name]()


if __name__ == '__main__':
    gen, dis = get_model_by_name('GAN')
    print(gen)
    print(dis)

