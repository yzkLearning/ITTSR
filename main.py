from option import args
from utils import mkExpDir
from dataset import dataloader
from model import TTSR
from loss1.loss import get_loss_dict
from trainer import Trainer

import os

import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':
    # make save_dir，创建保存目录，以存储模型和日志文件
    _logger = mkExpDir(args)

    # dataloader of training set and testing set
    # 加载数据集（创建训练集和测试集的数据加载器，用于将图像数据加载到模型中进行训练和测试。如果是测试模式，则不需要创建数据加载器。）
    _dataloader = dataloader.get_dataloader(args) if (not args.test) else None

    # device and model，指定设备
    device = torch.device('cpu' if args.cpu else 'cuda')
    # 加载模型
    _model = TTSR.TTSR(args).to(device)
    # （检查计算设备是否为CPU或GPU。
	#  创建 TTSR 模型，并将其移动到所选的计算设备上。
	#  如果计算设备为GPU，且设置使用多个GPU，则使用DataParallel将模型封装为多GPU模型。
    if ((not args.cpu) and (args.num_gpu > 1)):
        _model = nn.DataParallel(_model, list(range(args.num_gpu)))

    # 加载损失函数，用于计算训练期间的损失
    _loss_all = get_loss_dict(args, _logger)

    # （创建一个Trainer对象，用于管理训练过程并进行测试和评估）
    t = Trainer(args, _logger, _dataloader, _model, _loss_all)

    # test / eval / train
    if (args.test):
        t.load(model_path=args.model_path)
        t.test()
    elif (args.eval):
        t.load(model_path=args.model_path)
        t.evaluate()
    else:
        # 初始化训练
        for epoch in range(1, args.num_init_epochs+1):
            t.train(current_epoch=epoch, is_init=True)
        # 常规训练
        for epoch in range(1, args.num_epochs+1):
            t.train(current_epoch=epoch, is_init=False)
            # 定期评估模型性能
            if (epoch % args.val_every == 0):
                t.evaluate(current_epoch=epoch)  # 每训练 val_every 轮后，调用 t.evaluate() 方法进行模型的验证。current_epoch 参数表示当前训练的轮数。
