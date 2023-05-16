import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser(description='TTSR')

# log setting
parser.add_argument('--save_dir', type=str, default='save_dir',
                    help='Directory to save log, arguments, models and images')  # log保存目录
parser.add_argument('--reset', type=str2bool, default=False,
                    help='Delete save_dir to create a new one')  # 是否删除保存目录以创建新的目录
parser.add_argument('--log_file_name', type=str, default='TTSR.log',
                    help='Log file name')  # log文件名称
parser.add_argument('--logger_name', type=str, default='TTSR',
                    help='Logger name')  # logger名称

### device setting
parser.add_argument('--cpu', type=str2bool, default=False,
                    help='Use CPU to run code')  # 是否使用 CPU 运行代码
parser.add_argument('--num_gpu', type=int, default=1,
                    help='The number of GPU used in training')  # 使用的 GPU 数量

### dataset setting
parser.add_argument('--dataset', type=str, default='RRSSRD',
                    help='Which dataset to train and test')  # 数据集名称
parser.add_argument('--dataset_dir', type=str, default='./RRSSRD/',
                    help='Directory of dataset')  # 数据集目录

### dataloader setting
parser.add_argument('--num_workers', type=int, default=9,
                    help='The number of workers when loading data')  # 数据加载器的工作进程数量

# model setting
parser.add_argument('--num_res_blocks', type=str, default='16+16+8+4',
                    help='The number of residual blocks in each stage')   # 每个阶段的残差块数量
parser.add_argument('--n_feats', type=int, default=64,
                    help='The number of channels in network')   # 网络中的通道数
parser.add_argument('--res_scale', type=float, default=1.,
                    help='Residual scale')  # 残差块系数

# loss setting
parser.add_argument('--GAN_type', type=str, default='WGAN_GP',
                    help='The type of GAN used in training')  # 设置生成对抗网络的类型，初始值为WGAN_GP
parser.add_argument('--GAN_k', type=int, default=2,
                    help='Training discriminator k times when training generator once')  # 训练生成器时每训练一次判别器，就要训练GAN_k次
parser.add_argument('--tpl_use_S', type=str2bool, default=False,
                    help='Whether to multiply soft-attention map in transferal perceptual loss')  # 是否在转移感知损失中乘以软注意力图
parser.add_argument('--tpl_type', type=str, default='l2',
                    help='Which loss type to calculate gram matrix difference in transferal perceptual loss [l1 / l2]')   # 转移感知损失中计算Gram矩阵差异的损失类型，可以为'l1'或'l2'
parser.add_argument('--rec_w', type=float, default=1.,
                    help='The weight of reconstruction loss')  # 重建损失的权重
parser.add_argument('--per_w', type=float, default=0.006,
                    help='The weight of perceptual loss')   # 感知损失的权重
parser.add_argument('--tpl_w', type=float, default=0.006,
                    help='The weight of transferal perceptual loss')  # 转移感知损失的权重
parser.add_argument('--adv_w', type=float, default=0.001,
                    help='The weight of adversarial loss')  # 对抗损失的权重

# optimizer setting
parser.add_argument('--beta1', type=float, default=0.9,
                    help='The beta1 in Adam optimizer')  # Adam优化器的beta1参数
parser.add_argument('--beta2', type=float, default=0.999,
                    help='The beta2 in Adam optimizer')  # Adam优化器的beta2参数
parser.add_argument('--eps', type=float, default=1e-8,
                    help='The eps in Adam optimizer')  # Adam优化器的epsilon参数
parser.add_argument('--lr_rate', type=float, default=1e-4,
                    help='Learning rate')  # 初始学习率
parser.add_argument('--lr_rate_dis', type=float, default=1e-4,
                    help='Learning rate of discriminator')  # 判别器的初始学习率
parser.add_argument('--lr_rate_lte', type=float, default=1e-5,
                    help='Learning rate of LTE')  # 反卷积网络的初始学习率
parser.add_argument('--decay', type=float, default=999999,
                    help='Learning rate decay type')  # 学习率衰减类型
parser.add_argument('--gamma', type=float, default=0.5,
                    help='Learning rate decay factor for step decay')   # 学习率衰减因子

# training setting
parser.add_argument('--batch_size', type=int, default=8,
                    help='Training batch size')  # 原来是9    # 训练时的批量大小
parser.add_argument('--train_crop_size', type=int, default=40,
                    help='Training data crop size')  # 训练数据的裁剪大小，默认为40
parser.add_argument('--num_init_epochs', type=int, default=2,
                    help='The number of init epochs which are trained with only reconstruction loss')  #  原来是2。初始训练周期，只用重构损失进行训练
parser.add_argument('--num_epochs', type=int, default=50,   # 原来是1
                    help='The number of training epochs')  # 总的训练周期
parser.add_argument('--print_every', type=int, default=50,
                    help='Print period')   # 每隔多少个周期打印一次训练信息，默认为1
parser.add_argument('--save_every', type=int, default=5,
                    help='Save period')  # 每隔多少个周期保存一次模型，默认为999999
parser.add_argument('--val_every', type=int, default=5,
                    help='Validation period')  # 每隔多少个周期进行一次验证，默认为999999

# evaluate / test / finetune setting
parser.add_argument('--eval', type=str2bool, default=False,
                    help='Evaluation mode')   # 是否进入评估模式，默认为False
parser.add_argument('--eval_save_results', type=str2bool, default=False,
                    help='Save each image during evaluation')   # 是否保存评估结果图像，默认为False
parser.add_argument('--model_path', type=str, default='./gaijin/model/model_00050.pt',
                    help='The path of model to evaluation')  # 评估模型的路径，默认为'./TTSR.pt'
parser.add_argument('--test', type=str2bool, default=True,
                    help='Test mode')  # 是否进入测试模式，默认为False
parser.add_argument('--lr_path', type=str, default='./test/demo/lr/L18_112592_217064_s018.jpg',
                    help='The path of input lr image when testing')  # 测试模式下输入的低分辨率图像路径
parser.add_argument('--ref_path', type=str, default='./test/demo/ref/L18_112592_217064_s018.jpg',
                    help='The path of ref image when testing')   # 测试模式下参考图像的路径

args = parser.parse_args()
