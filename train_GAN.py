import os.path
from loss import L1Loss
import torch.nn as nn
from tqdm import tqdm
import sys
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from utils.utils import *
from utils.natural_process import *
from utils.eval_indicator import SSIM, AvgPixel, PSNR


def train(args, Generator, Discriminator):
    setup_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    experiment_dir, checkpoints_dir, tensorboard_dir = args.experiment_dir, args.checkpoints_dir, args.tensorboard_dir

    tb_writer = SummaryWriter(log_dir=tensorboard_dir, filename_suffix='--GAN')

    # 服务器转发到本地查看需要更改端口号
    print('Start Tensorboard with "tensorboard --logdir={}", view at http://localhost:6006/')
    print("Training on:", device)

    # 定义输入meshgrid/掩膜mask/待修复图像img0
    meshgrid = get_meshgrid().to(device)            # 两通道的meshgrid输入
    mask = get_mask(args.data_dir_i).to(device)     # 掩膜
    img0 = get_masked(args.data_dir_i).to(device)   # 真实待修复图像
    origin = get_gray(args.data_dir_i).to(device)   # 真实图像

    # 创建生成器与判别器
    gen = Generator.to(device)
    dis = Discriminator.to(device)

    # 模型参数量计算
    print('Generator' + get_parameter_number(gen))
    print('Discriminator' + get_parameter_number(dis))

    # 定义损失以及优化器
    criterion = nn.MSELoss()
    pixel_criterion = L1Loss()
    g_optimizer = Adam(gen.parameters(), lr=0.01)
    d_optimizer = Adam(dis.parameters(), lr=0.01)

    # 定义评价指标ssim以及 平均像素误差AvgPixel
    eval_ssim = SSIM(device=device)
    eval_pixel = AvgPixel()
    eval_psnr = PSNR()

    epoch_loader = tqdm(range(args.epoch_num), file=sys.stdout)
    for epoch in epoch_loader:
        # 打开训练模式
        gen.train()
        dis.train()

        # 生成的图像img1
        gen_img = gen(meshgrid)
        with torch.no_grad():
            img1 = gen_img * mask  # 用mask之后的img1进行判别

        # =======================================
        # 判别器训练，损失的构建和优化，分为以下两部分
        # 1.真实的图片判断为真
        # 2.虚假的图片判断为假
        # =======================================
        d_optimizer.zero_grad()

        real_output = dis(img0)  # 期望对真实图像img0的判别结果为1
        d_real_loss = criterion(real_output, torch.ones_like(real_output))

        fake_output = dis(img1)  # 期望对生成图像img1的判别结果为0
        d_fake_loss = criterion(fake_output, torch.zeros_like(fake_output))

        d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()
        d_optimizer.step()

        # =======================================
        # 生成器器训练，损失的构建和优化
        # =======================================
        g_optimizer.zero_grad()

        fake_output = dis(img1.detach())  # 期望对生成图像img1的判别结果为1
        Loss_GAN = criterion(fake_output, torch.ones_like(fake_output))
        Loss_Pixel = pixel_criterion(gen_img, img0, mask)
        g_loss = 0.1 * Loss_GAN + 0.9 * Loss_Pixel
        g_loss.backward()
        g_optimizer.step()

        # print('epoch-{} generator-loss:{} discriminator-loss:{}'
        #       .format(epoch, g_loss.item(), d_loss.item()))

        # tensorboard记录
        if epoch % args.val_epoch_step == 0:
            with torch.no_grad():
                ssim = eval_ssim(gen_img, origin)
                pixel = 255 * torch.sum(torch.abs(gen_img - origin)) / (origin.shape[2] * origin.shape[3])
            tags = ['ssim', 'pixel', 'generator-loss', 'discriminator-loss']
            tb_writer.add_scalar(tags[0], ssim, epoch)
            tb_writer.add_scalar(tags[1], pixel, epoch)
            tb_writer.add_scalar(tags[2], g_loss.item(), epoch)
            tb_writer.add_scalar(tags[3], d_loss.item(), epoch)

        if (epoch == 0) or ((epoch + 1) % 100 == 0):
            save_path = os.path.join(args.res_dir_i, 'epoch-{}.jpg'.format(epoch))
            save_image(gen_img[0], save_path)
            with torch.no_grad():
                ssim = eval_ssim(gen_img, origin)
                pixel = eval_pixel(gen_img, origin)
                psnr = eval_psnr(gen_img, origin)
                print('epoch-{}\t\t ssim:{}\t\t pixel:{}\t\t psnr:{}'.format(epoch, ssim.item(), pixel, psnr))

        # 计算最后一张补全后的图像img1 与 带空白带的图像img0 融合后的图像mergeImg 与 原图origin 的ssim与平均像素灰度误差
        if epoch == args.epoch_num-1:
            merge(args)
            mergeImg = get_merge(args.res_dir_i).to(device)
            with torch.no_grad():
                ssim = eval_ssim(mergeImg, origin)
                pixel = eval_pixel(mergeImg, origin)
                psnr = eval_psnr(mergeImg, origin)
                print("===================================================================")
                print('eval-merge-{}\t\t ssim:{}\t\t pixel:{}\t\t psnr:{}'.format(epoch, ssim.item(), pixel, psnr))

    return ssim.item(), pixel.cpu(), psnr.cpu()
