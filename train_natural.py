import sys
from loss import L1Loss
from tqdm import tqdm
from torch.optim import Adam
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
from utils.utils import *
from utils.natural_process import *
from utils.eval_indicator import SSIM, AvgPixel, PSNR


def train(args, model):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    experiment_dir, checkpoints_dir, tensorboard_dir = args.experiment_dir, args.checkpoints_dir, args.tensorboard_dir

    tb_writer = SummaryWriter(log_dir=tensorboard_dir)

    # 服务器转发到本地查看需要更改端口号
    print('Start Tensorboard with "tensorboard --logdir={}", view at http://localhost:6006/')
    print("Training on:", device)

    # 定义输入meshgrid/掩膜mask/待修复图像img0
    meshgrid = get_meshgrid().to(device)  # 两通道的meshgrid输入
    mask = get_mask(args.data_dir_i).to(device)  # 掩膜
    img0 = get_masked(args.data_dir_i).to(device)  # 真实待修复图像
    origin = get_gray(args.data_dir_i).to(device)  # 真实图像

    # 加载模型
    model = model.to(device)

    # 模型参数量计算
    print('model' + get_parameter_number(model))

    # 定义损失以及优化器
    criterion = L1Loss()
    optim = Adam(model.parameters(), lr=args.lr)

    # 定义评价指标ssim以及 平均像素误差AvgPixel
    eval_ssim = SSIM(device=device)
    eval_pixel = AvgPixel()
    eval_psnr = PSNR()

    model.train()
    epoch_loader = tqdm(range(args.epoch_num), file=sys.stdout)

    for epoch in epoch_loader:
        optim.zero_grad()

        img1 = model(meshgrid)

        loss = criterion(img1, img0, mask)
        loss.backward()
        optim.step()

        # tensorboard记录
        if epoch % args.val_epoch_step == 0:
            with torch.no_grad():
                ssim = eval_ssim(img1, origin)
                pixel = 255*torch.sum(torch.abs(img1-origin))/(origin.shape[2]*origin.shape[3])
            tags = ['ssim', 'pixel']
            tb_writer.add_scalar(tags[0], ssim, epoch)
            tb_writer.add_scalar(tags[1], pixel, epoch)

        if (epoch == 0) or ((epoch + 1) % 100 == 0):
            save_path = os.path.join(args.res_dir_i, 'epoch-{}.jpg'.format(epoch))
            save_image(img1[0], save_path)
            with torch.no_grad():
                ssim = eval_ssim(img1, origin)
                pixel = eval_pixel(img1, origin)
                psnr = eval_psnr(img1, origin)
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
