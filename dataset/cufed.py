import os
from imageio.v2 import imread
from PIL import Image
import numpy as np
import glob
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")


class RandomRotate(object):
    def __call__(self, sample):
        k1 = np.random.randint(0, 4)  # 随机生成0-4之间的整数k1
        sample['LR'] = np.rot90(sample['LR'], k1).copy()  # 对LR属性进行k1次90度旋转，并替换原来的LR属性
        sample['HR'] = np.rot90(sample['HR'], k1).copy()  # 对HR属性进行k1次90度旋转，并替换原来的HR属性
        sample['LR_sr'] = np.rot90(sample['LR_sr'], k1).copy()  # 对LR_sr属性进行k1次90度旋转，并替换原来的LR_sr属性
        k2 = np.random.randint(0, 4)
        sample['Ref'] = np.rot90(sample['Ref'], k2).copy()
        sample['Ref_sr'] = np.rot90(sample['Ref_sr'], k2).copy()
        return sample  # 返回旋转后的样本


class RandomFlip(object):
    def __call__(self, sample):
        # 随机进行左右翻转
        if (np.random.randint(0, 2) == 1):
            sample['LR'] = np.fliplr(sample['LR']).copy()
            sample['HR'] = np.fliplr(sample['HR']).copy()
            sample['LR_sr'] = np.fliplr(sample['LR_sr']).copy()
        if (np.random.randint(0, 2) == 1):
            sample['Ref'] = np.fliplr(sample['Ref']).copy()
            sample['Ref_sr'] = np.fliplr(sample['Ref_sr']).copy()
        # 随机进行上下翻转
        if (np.random.randint(0, 2) == 1):
            sample['LR'] = np.flipud(sample['LR']).copy()
            sample['HR'] = np.flipud(sample['HR']).copy()
            sample['LR_sr'] = np.flipud(sample['LR_sr']).copy()
        if (np.random.randint(0, 2) == 1):
            sample['Ref'] = np.flipud(sample['Ref']).copy()
            sample['Ref_sr'] = np.flipud(sample['Ref_sr']).copy()
        return sample


class ToTensor(object):
    def __call__(self, sample):
        # 从sample中获取LR, LR_sr, HR, Ref, Ref_sr属性
        LR, LR_sr, HR, Ref, Ref_sr = sample['LR'], sample['LR_sr'], sample['HR'], sample['Ref'], sample['Ref_sr']
        # 将LR属性转换为(通道数，高度，宽度)的形状
        LR = LR.transpose((2, 0, 1))
        LR_sr = LR_sr.transpose((2, 0, 1))
        HR = HR.transpose((2, 0, 1))
        Ref = Ref.transpose((2, 0, 1))
        Ref_sr = Ref_sr.transpose((2, 0, 1))
        # 将转换后的属性转换为torch.Tensor类型，并返回
        return {'LR': torch.from_numpy(LR).float(),
                'LR_sr': torch.from_numpy(LR_sr).float(),
                'HR': torch.from_numpy(HR).float(),
                'Ref': torch.from_numpy(Ref).float(),
                'Ref_sr': torch.from_numpy(Ref_sr).float()}


class TrainSet(Dataset):
    def __init__(self, args, transform=transforms.Compose([RandomFlip(), RandomRotate(), ToTensor()])):
        # 初始化数据集
        # args是一个命名空间，包含超分辨率模型训练的所有参数
        # transform是一个数据变换的组合
        # 初始化输入图像列表
        self.input_list = sorted([os.path.join(args.dataset_dir, './train/input', name) for name in
                                  os.listdir(os.path.join(args.dataset_dir, './train/input'))])
        # 初始化参考图像列表
        self.ref_list = sorted([os.path.join(args.dataset_dir, './train/ref', name) for name in
                                os.listdir(os.path.join(args.dataset_dir, './train/ref'))])
        # 将数据变换组合存储到类的变量中
        self.transform = transform

    def __len__(self):
        # 返回数据集中图像的数量
        return len(self.input_list)

    def __getitem__(self, idx):
        ### HR
        # print(len(self.ref_list));
        # print(index);
        # 读取高分辨率图像
        HR = imread(self.input_list[idx])
        h, w = HR.shape[:2] # 去彩色图像的长、宽
        # HR = HR[:h//4*4, :w//4*4, :]

        # LR and LR_sr
        # 缩小高分辨率图像四倍，得到低分辨率图像LR和LR_sr
        LR = np.array(Image.fromarray(HR).resize((w // 4, h // 4), Image.BICUBIC))
        LR_sr = np.array(Image.fromarray(LR).resize((w, h), Image.BICUBIC))

        ### Ref and Ref_sr
        # 读取参考图像，并得到参考图像的低分辨率版本Ref_sr
        Ref_sub = imread(self.ref_list[idx])
        h2, w2 = Ref_sub.shape[:2]
        Ref_sr_sub = np.array(Image.fromarray(Ref_sub).resize((w2 // 4, h2 // 4), Image.BICUBIC))
        Ref_sr_sub = np.array(Image.fromarray(Ref_sr_sub).resize((w2, h2), Image.BICUBIC))

        ### complete ref and ref_sr to the same size, to use batch_size > 1
        # 将参考图像和其低分辨率版本都放入160x160的图像中，以支持批处理
        Ref = np.zeros((160, 160, 3))
        Ref_sr = np.zeros((160, 160, 3))
        # Ref = np.zeros((480, 480, 3))
        # Ref_sr = np.zeros((480, 480, 3))

        Ref[:h2, :w2, :] = Ref_sub
        Ref_sr[:h2, :w2, :] = Ref_sr_sub

        ### change type
        # 将所有图像转换为float32类型
        LR = LR.astype(np.float32)
        LR_sr = LR_sr.astype(np.float32)
        HR = HR.astype(np.float32)
        Ref = Ref.astype(np.float32)
        Ref_sr = Ref_sr.astype(np.float32)

        ### rgb range to [-1, 1]
        # 将所有图像的像素值归一化到[-1，1]范围内
        LR = LR / 127.5 - 1.
        LR_sr = LR_sr / 127.5 - 1.
        HR = HR / 127.5 - 1.
        Ref = Ref / 127.5 - 1.
        Ref_sr = Ref_sr / 127.5 - 1.

        # 将这些图像样本以字典形式存储在sample变量中
        sample = {'LR': LR,
                  'LR_sr': LR_sr,
                  'HR': HR,
                  'Ref': Ref,
                  'Ref_sr': Ref_sr}

        if self.transform:
            sample = self.transform(sample)
        return sample


class TestSet(Dataset):
    def __init__(self, args, ref_level='1', transform=transforms.Compose([ToTensor()])):
        # self.input_list = sorted(glob.glob(os.path.join(args.dataset_dir, './test/RRSSRD', '*_0.png')))
        # self.ref_list = sorted(glob.glob(os.path.join(args.dataset_dir, './test/RRSSRD',
        #                                               '*_' + ref_level + '.png')))
        self.input_list = sorted(glob.glob(os.path.join(args.dataset_dir, './test/input', '*.jpg')))
        self.ref_list = sorted(glob.glob(os.path.join(args.dataset_dir, './test/ref', '*.jpg')))

        self.transform = transform

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        ### HR
        HR = imread(self.input_list[idx])
        h, w = HR.shape[:2]
        h, w = h // 4 * 4, w // 4 * 4
        HR = HR[:h, :w, :]  ### crop to the multiple of 4

        ### LR and LR_sr
        LR = np.array(Image.fromarray(HR).resize((w // 4, h // 4), Image.BICUBIC))
        LR_sr = np.array(Image.fromarray(LR).resize((w, h), Image.BICUBIC))

        ### Ref and Ref_sr
        Ref = imread(self.ref_list[idx])
        h2, w2 = Ref.shape[:2]
        h2, w2 = h2 // 4 * 4, w2 // 4 * 4
        Ref = Ref[:h2, :w2, :]
        Ref_sr = np.array(Image.fromarray(Ref).resize((w2 // 4, h2 // 4), Image.BICUBIC))
        Ref_sr = np.array(Image.fromarray(Ref_sr).resize((w2, h2), Image.BICUBIC))
        # 首先，它将 Ref 图像缩小4倍，生成一个更小的 Ref_sr 图像。这里使用了双三次插值方法 (Image.BICUBIC)，它是一种高质量的插值方法，可以得到比较平滑的插值效果。
        # 然后，代码再将 Ref_sr 图像放大4倍，得到一个与 Ref 图像相同大小的 Ref_sr 图像，也是使用双三次插值方法进行处理。
        # 这样做的目的是将 Ref 和 Ref_sr 图像都变为同样的尺寸，以便后续进行批处理。

        ### change type
        LR = LR.astype(np.float32)
        LR_sr = LR_sr.astype(np.float32)
        HR = HR.astype(np.float32)
        Ref = Ref.astype(np.float32)
        Ref_sr = Ref_sr.astype(np.float32)

        ### rgb range to [-1, 1]
        LR = LR / 127.5 - 1.
        LR_sr = LR_sr / 127.5 - 1.
        HR = HR / 127.5 - 1.
        Ref = Ref / 127.5 - 1.
        Ref_sr = Ref_sr / 127.5 - 1.

        sample = {'LR': LR,
                  'LR_sr': LR_sr,
                  'HR': HR,
                  'Ref': Ref,
                  'Ref_sr': Ref_sr}

        if self.transform:
            sample = self.transform(sample)
        return sample
