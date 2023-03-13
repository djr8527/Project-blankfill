import argparse
from utils.building_dataset import building_dataset


def get_args():
    """
    训练井像填充空白带的参数
    :return:
    """
    parser = argparse.ArgumentParser(description='Training')

    parser.add_argument('--seed', default=999, type=int, help='随机种子')
    parser.add_argument('--device', default='cuda:0', type=str, help='设备号')
    parser.add_argument('--model_name', default='UNet', type=str, help='模型名称')
    parser.add_argument('--lr', default=0.01, type=float, help='模型学习率')
    parser.add_argument('--epoch_num', default=2000, type=int, help='训练迭代次数')
    parser.add_argument('--val_epoch_step', default=10, type=int, help='模型评估步数')
    parser.add_argument('--las_dir', default='las/ma3/ma3-stat.las', type=str, help='井像las文件存储路径')
    parser.add_argument('--data_dir', default='dataset/well/stat/ma3', type=str, help='所有井段数据集保存路径')
    parser.add_argument('--res_dir', default='result/well/stat/ma3', type=str, help='所有补全结果保存路径')
    parser.add_argument('--data_dir_i', default='dataset/well/stat/ma3/1', type=str, help='第i个井段数据集保存路径')
    parser.add_argument('--res_dir_i', default='result/well/stat/ma3/1', type=str, help='第i个井段补全结果保存路径')
    parser.add_argument('--size', default=256, type=int, help='井段长度为n个256行像素块')
    parser.add_argument('--n', default=20, type=int, help='井段长度为256行像素块的块数')
    parser.add_argument('--width', default=250, type=int, help='井段宽度')

    parser.add_argument('--l1', default=0, type=int, help='最后一段完整的256行像素块的个数')
    parser.add_argument('--l2', default=256, type=int, help='最后一段最后不完整的256行像素块需要补齐的像素行数')
    parser.add_argument('--l3', default=6, type=int, help='宽度与32倍数的像素值')

    parser.add_argument('--log_dir', default='./log/ma3', type=str, help='日志路径')
    parser.add_argument('--type', default=0, type=int, help='表明为井像训练')

    args = parser.parse_args()

    # 构建井像图数据集
    width, l3 = building_dataset(args.las_dir, args.data_dir, args.n)
    args.width = width
    args.l3 = l3

    return args
