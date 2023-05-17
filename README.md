# ITTSR
TTSR的改进算法。

## 内容
- [介绍](#介绍)
  - [贡献](#贡献)
  - [主要结果](#主要结果)
- [实验环境与依赖](#实验环境与依赖)
- [模型](#模型)
- [数据集](#数据集)

## 介绍
为RefSR任务提出了一种称为ITTSR的方法。与SISR相比，RefSR具有额外的高分辨参考图像，其纹理可用于帮助超分辨重建出效果更好的图像。

### 贡献
在TTSR（CVPR 2020）的基础上，对其主网络的激活函数ReLU、重建损失函数进行改进，使模型训练稳定。


### 主要结果
<img src="https://github.com/yzkLearning/ITTSR/blob/master/IMG/results.png" width=80%>

## 实验环境与依赖
* python 3.8 (需要使用 [Anaconda](https://www.anaconda.com/))
* python packages: `pip install opencv-python imageio`
* pytorch = 1.12.0+cu113
* torchvision = 0.13.0+cu113

## 数据集
1. 数据集可在[百度网盘](https://pan.baidu.com/s/1OVCdRxTfStuFdeDLcallOg)(提取码：d6qo)中下载
2. 数据集文件夹结构:
- RRSSRD
    - train
        - input
        - ref
    - test
        - HR
        - input
        - ref


