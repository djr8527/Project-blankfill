import sys
from tqdm import tqdm
from loss import L1Loss
from torch.optim import Adam
from torchvision.utils import save_image
from utils.utils import *
from utils.well_process import *


def train(args, model):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print("Training on:", device)

    # 定义输入meshgrid/掩膜mask/待修复图像img0
    meshgrid = torch.randn([1, 2, args.size, args.width+args.l3]).to(device)  # 随机噪声输入
    mask = get_mask(args).to(device)    # 掩膜mask
    img0 = get_masked(args).to(device)  # 带空白带的原始井像图

    # 将模型加载到指定设备
    model = model.to(device)

    # 模型参数量计算
    print('model' + get_parameter_number(model))

    # 定义损失以及优化器
    criterion = L1Loss()
    optim = Adam(model.parameters(), lr=args.lr)

    model.train()

    epoch_loader = tqdm(range(args.epoch_num), file=sys.stdout)

    for epoch in epoch_loader:
        optim.zero_grad()

        img1 = model(meshgrid)      # 空白带补全后的图像

        loss = criterion(img1, img0, mask)
        loss.backward()
        optim.step()

        if (epoch == 0) or ((epoch + 1) % 100 == 0):
            save_image(img1[0], os.path.join(args.res_dir_i, 'epoch-{}.jpg'.format(epoch)))
