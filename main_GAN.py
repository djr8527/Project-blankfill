import numpy as np

from utils.natural_process import *
from config.config1 import get_args
from train_GAN import train
from utils.utils import init_settings, setup_seed
from utils.create_model import get_model_by_name
import time


if __name__ == '__main__':
    args = get_args()
    setup_seed(args.seed)
    args.model_name = 'GAN'
    args.experiment_dir, args.checkpoints_dir, args.tensorboard_dir = init_settings(args)
    gen, dis = get_model_by_name(args.model_name)

    ssim, l1, psnr = [], [], []

    start = time.time()

    # n次试验取平均
    for _ in range(args.n):
        e1, e2, e3 = [], [], []
        # 训练10张自然图像
        for dir in os.listdir(args.data_dir):
            args.data_dir_i = os.path.join(args.data_dir, dir)
            args.res_dir_i = os.path.join(args.res_dir, dir)

            if not os.path.exists(args.res_dir_i):
                os.makedirs(args.res_dir_i)

            print(args.data_dir_i)
            res = train(args, gen, dis)
            e1.append(res[0])
            e2.append(res[1])
            e3.append(res[2])
            print('===============================================================================')
        ssim.append(np.mean(e1))
        l1.append(np.mean(e2))
        psnr.append(np.mean(e3))

    end = time.time()
    mean_run_time = int((end - start) / args.n)

    with open(os.path.join(args.experiment_dir, 'log.txt'), mode='w') as log_object:
        log_object.write(args.model_name + '\tevaluate\t ssim:{}\t pixel:{}\t psnr:{}'.format(np.mean(ssim), np.mean(l1), np.mean(psnr)))
        log_object.write('\nmean-run-time:' + time.strftime("%H:%M:%S", time.gmtime(mean_run_time)))
    print('evaluate\t\t ssim:{}\t\t pixel:{}\t\t psnr:{}'.format(np.mean(ssim), np.mean(l1), np.mean(psnr)))
