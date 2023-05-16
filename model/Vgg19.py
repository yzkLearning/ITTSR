import torch
from torchvision import models
from utils import MeanShift


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False, rgb_range=1):
        super(Vgg19, self).__init__()
        # 从预训练模型中加载VGG19的特征提取部分
        vgg_pretrained_features = models.vgg19(pretrained=True).features

        # 截取VGG19特定层的部分用于特征提取
        # 截取模型的前30层，用于特征提取；如果requires_grad为False，则将该层的参数requires_grad设置为False；
        self.slice1 = torch.nn.Sequential()
        for x in range(30):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        # 如果不需要对该层进行梯度反向传播，则将参数requires_grad设置为False
        if not requires_grad:
            for param in self.slice1.parameters():
                param.requires_grad = False

        # RGB图像预处理，将像素值从[0,1]标准化到[-1,1]范围内
        vgg_mean = (0.485, 0.456, 0.406)
        vgg_std = (0.229 * rgb_range, 0.224 * rgb_range, 0.225 * rgb_range)
        self.sub_mean = MeanShift(rgb_range, vgg_mean, vgg_std)

    # 前向传播函数
    def forward(self, X):
        # 对输入进行RGB图像预处理
        h = self.sub_mean(X)
        # 使用VGG19提取特征
        h_relu5_1 = self.slice1(h)
        return h_relu5_1


if __name__ == '__main__':
    vgg19 = Vgg19(requires_grad=False)