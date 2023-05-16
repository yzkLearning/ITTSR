from utils import calc_psnr_and_ssim
from model import Vgg19

import os
import numpy as np
from imageio import imread, imsave
from PIL import Image

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as utils


class Trainer():
    def __init__(self, args, logger, dataloader, model, loss_all):
        self.args = args
        self.logger = logger
        self.dataloader = dataloader
        self.model = model
        self.loss_all = loss_all
        self.device = torch.device('cpu') if args.cpu else torch.device('cuda')
        # 初始化 Vgg19 模型
        self.vgg19 = Vgg19.Vgg19(requires_grad=False).to(self.device)
        if ((not self.args.cpu) and (self.args.num_gpu > 1)):
            self.vgg19 = nn.DataParallel(self.vgg19, list(range(self.args.num_gpu)))
        # 设置优化器的参数
        self.params = [
            {"params": filter(lambda p: p.requires_grad, self.model.MainNet.parameters() if 
             args.num_gpu==1 else self.model.module.MainNet.parameters()),
             "lr": args.lr_rate
            },
            {"params": filter(lambda p: p.requires_grad, self.model.LTE.parameters() if 
             args.num_gpu==1 else self.model.module.LTE.parameters()), 
             "lr": args.lr_rate_lte
            }
        ]
        # 初始化优化器
        self.optimizer = optim.Adam(self.params, betas=(args.beta1, args.beta2), eps=args.eps)
        # 初始化学习率调整策略
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=self.args.decay, gamma=self.args.gamma)
        # 初始化最大的 PSNR 和 SSIM 以及对应的 epoch
        self.max_psnr = 0.
        self.max_psnr_epoch = 0
        self.max_ssim = 0.
        self.max_ssim_epoch = 0

    def load(self, model_path=None):
        # 加载模型，传入参数 model_path，如果存在就加载模型
        if (model_path):
            self.logger.info('load_model_path: ' + model_path)
            # 加载模型参数，并更新到 self.model 中
            # model_state_dict_save = {k.replace('module.',''):v for k,v in torch.load(model_path).items()}
            model_state_dict_save = {k:v for k,v in torch.load(model_path, map_location=self.device).items()}
            model_state_dict = self.model.state_dict()
            model_state_dict.update(model_state_dict_save)
            self.model.load_state_dict(model_state_dict)

    def prepare(self, sample_batched):
        # 将数据批次中的所有数据移动到相应的设备中
        for key in sample_batched.keys():
            sample_batched[key] = sample_batched[key].to(self.device)
        return sample_batched

    def train(self, current_epoch=0, is_init=False):
        # 将模型设为训练模式
        self.model.train()
        if (not is_init):
            self.scheduler.step()  # 如果不是初始的epoch，则更新scheduler
        self.logger.info('Current epoch learning rate: %e' %(self.optimizer.param_groups[0]['lr']))  # 打印当前epoch的学习率

        for i_batch, sample_batched in enumerate(self.dataloader['train']):  # 遍历训练数据集
            self.optimizer.zero_grad()  # 梯度清零

            sample_batched = self.prepare(sample_batched)  # 准备数据
            lr = sample_batched['LR']  # 低分辨率图像
            lr_sr = sample_batched['LR_sr']  # 放大后的低分辨率图像
            hr = sample_batched['HR']  # 高分辨率图像
            ref = sample_batched['Ref']  # 引导图像
            ref_sr = sample_batched['Ref_sr']  # 放大后的引导图像
            sr, S, T_lv3, T_lv2, T_lv1 = self.model(lr=lr, lrsr=lr_sr, ref=ref, refsr=ref_sr)  # 通过模型计算出预测结果和相关变量

            ### calc loss
            is_print = ((i_batch + 1) % self.args.print_every == 0)  # 标记是否需要打印

            rec_loss = self.args.rec_w * self.loss_all['rec_loss'](sr, hr)  # 重建损失
            loss = rec_loss
            if (is_print):  # 打印重建损失
                self.logger.info( ('init ' if is_init else '') + 'epoch: ' + str(current_epoch) + 
                    '\t batch: ' + str(i_batch+1) )
                self.logger.info( 'rec_loss: %.10f' %(rec_loss.item()) )

            if (not is_init):  # 如果不是初始的epoch
                if ('per_loss' in self.loss_all):  # 计算感知损失
                    sr_relu5_1 = self.vgg19((sr + 1.) / 2.)  # 预测结果通过VGG网络得到的relu5_1特征
                    with torch.no_grad():
                        hr_relu5_1 = self.vgg19((hr.detach() + 1.) / 2.)  # 真实结果通过VGG网络得到的relu5_1特征
                    per_loss = self.args.per_w * self.loss_all['per_loss'](sr_relu5_1, hr_relu5_1)  # 计算感知损失
                    loss += per_loss  # 总损失加上感知损失
                    if (is_print):   # 打印感知损失
                        self.logger.info( 'per_loss: %.10f' %(per_loss.item()) )
                if ('tpl_loss' in self.loss_all):  # 计算模板损失
                    sr_lv1, sr_lv2, sr_lv3 = self.model(sr=sr)  # 预测结果通过模型得到的3个尺度的特征
                    tpl_loss = self.args.tpl_w * self.loss_all['tpl_loss'](sr_lv3, sr_lv2, sr_lv1, 
                        S, T_lv3, T_lv2, T_lv1)  # 计算texture perceptual loss
                    loss += tpl_loss
                    if (is_print):   # 如果需要打印信息
                        self.logger.info( 'tpl_loss: %.10f' %(tpl_loss.item()) )  # 打印纹理感知损失的值
                if ('adv_loss' in self.loss_all):
                    adv_loss = self.args.adv_w * self.loss_all['adv_loss'](sr, hr)  # 计算adversarial loss
                    loss += adv_loss
                    if (is_print):
                        self.logger.info( 'adv_loss: %.10f' %(adv_loss.item()) )

            loss.backward()   # 反向传播，计算梯度
            self.optimizer.step()  # 更新参数

        if ((not is_init) and current_epoch % self.args.save_every == 0):  # 如果模型已经初始化，且当前epoch是保存模型的倍数
            self.logger.info('saving the model...')
            tmp = self.model.state_dict()   # 得到模型的状态字典
            model_state_dict = {key.replace('module.',''): tmp[key] for key in tmp if 
                (('SearchNet' not in key) and ('_copy' not in key))}  # 除去不需要保存的模块和复制模块，将模型状态字典的键中的"module."去除
            model_name = self.args.save_dir.strip('/')+'/model/model_'+str(current_epoch).zfill(5)+'.pt'
            torch.save(model_state_dict, model_name)  # 模型保存的路径和名称

    def evaluate(self, current_epoch=0):
        self.logger.info('Epoch ' + str(current_epoch) + ' evaluation process...')

        if (self.args.dataset == 'RRSSRD'):
            self.model.eval()
            with torch.no_grad():
                psnr, ssim, cnt = 0., 0., 0
                for i_batch, sample_batched in enumerate(self.dataloader['test']['1']):
                    cnt += 1
                    sample_batched = self.prepare(sample_batched)
                    lr = sample_batched['LR']
                    lr_sr = sample_batched['LR_sr']
                    hr = sample_batched['HR']
                    ref = sample_batched['Ref']
                    ref_sr = sample_batched['Ref_sr']

                    sr, _, _, _, _ = self.model(lr=lr, lrsr=lr_sr, ref=ref, refsr=ref_sr)
                    if (self.args.eval_save_results):
                        sr_save = (sr+1.) * 127.5
                        sr_save = np.transpose(sr_save.squeeze().round().cpu().numpy(), (1, 2, 0)).astype(np.uint8)
                        imsave(os.path.join(self.args.save_dir, 'save_results', str(i_batch).zfill(5)+'.jpg'), sr_save)
                    
                    ### calculate psnr and ssim
                    _psnr, _ssim = calc_psnr_and_ssim(sr.detach(), hr.detach())

                    psnr += _psnr
                    ssim += _ssim

                psnr_ave = psnr / cnt
                ssim_ave = ssim / cnt
                self.logger.info('Ref  PSNR (now): %.3f \t SSIM (now): %.4f' %(psnr_ave, ssim_ave))
                if (psnr_ave > self.max_psnr):
                    self.max_psnr = psnr_ave
                    self.max_psnr_epoch = current_epoch
                if (ssim_ave > self.max_ssim):
                    self.max_ssim = ssim_ave
                    self.max_ssim_epoch = current_epoch
                self.logger.info('Ref  PSNR (max): %.3f (%d) \t SSIM (max): %.4f (%d)' 
                    %(self.max_psnr, self.max_psnr_epoch, self.max_ssim, self.max_ssim_epoch))

        self.logger.info('Evaluation over.')

    def test(self):
        self.logger.info('Test process...')
        self.logger.info('lr path:     %s' %(self.args.lr_path))
        self.logger.info('ref path:    %s' %(self.args.ref_path))

        ### LR and LR_sr
        LR = imread(self.args.lr_path)
        h1, w1 = LR.shape[:2]
        LR_sr = np.array(Image.fromarray(LR).resize((w1*4, h1*4), Image.BICUBIC))
        
        ### Ref and Ref_sr
        Ref = imread(self.args.ref_path)
        h2, w2 = Ref.shape[:2]
        h2, w2 = h2//4*4, w2//4*4
        Ref = Ref[:h2, :w2, :]
        Ref_sr = np.array(Image.fromarray(Ref).resize((w2//4, h2//4), Image.BICUBIC))
        Ref_sr = np.array(Image.fromarray(Ref_sr).resize((w2, h2), Image.BICUBIC))

        ### change type
        LR = LR.astype(np.float32)
        LR_sr = LR_sr.astype(np.float32)
        Ref = Ref.astype(np.float32)
        Ref_sr = Ref_sr.astype(np.float32)

        ### rgb range to [-1, 1]
        LR = LR / 127.5 - 1.
        LR_sr = LR_sr / 127.5 - 1.
        Ref = Ref / 127.5 - 1.
        Ref_sr = Ref_sr / 127.5 - 1.

        ### to tensor
        LR_t = torch.from_numpy(LR.transpose((2,0,1))).unsqueeze(0).float().to(self.device)
        LR_sr_t = torch.from_numpy(LR_sr.transpose((2,0,1))).unsqueeze(0).float().to(self.device)
        Ref_t = torch.from_numpy(Ref.transpose((2,0,1))).unsqueeze(0).float().to(self.device)
        Ref_sr_t = torch.from_numpy(Ref_sr.transpose((2,0,1))).unsqueeze(0).float().to(self.device)

        self.model.eval()
        with torch.no_grad():
            sr, _, _, _, _ = self.model(lr=LR_t, lrsr=LR_sr_t, ref=Ref_t, refsr=Ref_sr_t)
            sr_save = (sr+1.) * 127.5
            sr_save = np.transpose(sr_save.squeeze().round().cpu().numpy(), (1, 2, 0)).astype(np.uint8)
            save_path = os.path.join(self.args.save_dir, 'save_results', os.path.basename(self.args.lr_path))
            imsave(save_path, sr_save)
            self.logger.info('output path: %s' %(save_path))

        self.logger.info('Test over.')