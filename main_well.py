import time
from utils.well_process import *
from config.config0 import get_args
from train_well import train
from utils.utils import init_settings, setup_seed
from utils.create_model import get_model_by_name


if __name__ == '__main__':
    args = get_args()
    setup_seed(args.seed)
    args.model_name = 'UNet'
    log_dir = init_settings(args)

    model = get_model_by_name(args.model_name)
    run_time = []

    # 将井像图分割成一定长度(256*n)的井段，分别进行补全,最后合并
    for dir in os.listdir(args.data_dir):
        print(dir)
        args.data_dir_i = os.path.join(args.data_dir, dir)
        args.res_dir_i = os.path.join(args.res_dir, dir)
        if not os.path.exists(args.res_dir_i):
            os.makedirs(args.res_dir_i)

        start = time.time()

        if dir != 'last':
            args.size = 256 * args.n
            args.l2 = 256

            # 训练 生成补全空白带后的图像
            train(args, model)

            # 将生成的空白带部分 与 带空白带的原图 合并
            merge(args)

        else:
            df = pd.read_csv(os.path.join(args.data_dir_i, 'last.csv'))
            args.l1 = df['0'][0]  # 最后一段完整的256行像素块的个数
            args.l2 = df['0'][1]  # 最后一段最后不完整的256行像素块需要补齐的像素行数
            args.size = 256*(args.l1+1)

            train(args, model)
            merge(args)

        end = time.time()

        run_time.append(end - start)

    # 记录补全图像所花费的时长
    with open(os.path.join(log_dir, 'log.txt'), mode='w') as log_object:
        log_object.write(time.strftime("%H:%M:%S", time.gmtime(int(sum(run_time)))))

    # 将井段合并保存为las文件
    save_las(args)
