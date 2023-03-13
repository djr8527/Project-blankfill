import argparse


def get_args():
    """
    训练自然图像填充的参数
    :return:
    """
    parser = argparse.ArgumentParser(description='Training')

    parser.add_argument('--seed', default=999, type=int, help='随机种子')
    parser.add_argument('--device', default='cuda:0', type=str, help='设备号')
    parser.add_argument('--lr', default=0.01, type=float, help='模型学习率')
    parser.add_argument('--model_name', default='UNet', type=str, help='模型名称')
    parser.add_argument('--epoch_num', default=2000, type=int, help='训练迭代次数')
    parser.add_argument('--n', default=1, type=int, help='实验次数')
    parser.add_argument('--val_epoch_step', default=10, type=int, help='模型评估步数')

    parser.add_argument('--data_dir', default='dataset/place365', type=str, help='自然图像数据集路径')
    parser.add_argument('--res_dir', default='result/place365', type=str, help='自然图像补全结果保存路径')
    parser.add_argument('--data_dir_i', default='dataset/place365/1', type=str, help='第i张自然图像的保存路径')
    parser.add_argument('--res_dir_i', default='result/place365/1', type=str, help='第i张自然图像的补全结果保存路径')
    parser.add_argument('--type', default=1, type=int, help='表明为自然图像')
    parser.add_argument('--log_dir', default='./log/place365', type=str, help='日志根路径')
    parser.add_argument('--experiment_dir', default='./log/place365', type=str, help='实验根路径')
    parser.add_argument('--checkpoints_dir', default='./log/place365', type=str, help='检查点路径')
    parser.add_argument('--tensorboard_dir', default='./log/place365', type=str, help='tensorboard日志路径')

    args = parser.parse_args()

    return args
