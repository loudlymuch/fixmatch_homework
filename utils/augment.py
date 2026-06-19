import torch
from torchvision import transforms

# CIFAR-10 数据集的均值和标准差（在整个训练集上统计的固定数值）
# 用于归一化，将像素值映射到标准正态分布附近，帮助模型更快收敛
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2471, 0.2435, 0.2616)


def get_weak_transform():
    """获取弱数据增强变换（FixMatch 中的 "weak augmentation"）。

    弱增强用于：
    1. 有标签数据的标准训练增强
    2. 无标签数据生成伪标签时的增强（保证预测稳定）

    包含操作：
    - RandomHorizontalFlip：随机水平翻转
    - RandomCrop：随机裁剪（先填充 4 像素再裁回 32×32）
    - ToTensor + Normalize：转为 Tensor 并归一化

    Returns:
        torchvision.transforms.Compose 组合变换
    """
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32, padding=4, padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
        ]
    )


def get_strong_transform():
    """获取强数据增强变换（FixMatch 中的 "strong augmentation"）。

    强增强用于无标签数据的强增强版本，与弱增强版本的伪标签一起
    计算一致性正则化损失。这是 FixMatch 区别于 MixMatch 等方法的
    核心设计之一。

    包含操作：
    - RandomHorizontalFlip：随机水平翻转
    - RandomCrop：随机裁剪
    - RandAugment：随机选择 2 种增强操作，强度为 10（核心强增强）
    - ToTensor：转为 Tensor
    - RandomErasing：随机擦除（模拟 Cutout 正则化）
    - Normalize：归一化

    Returns:
        torchvision.transforms.Compose 组合变换
    """
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32, padding=4, padding_mode='reflect'),
            # 核心：RandAugment。n=2 代表随机选2种变换，m=10 代表变换强度（0-30）
            transforms.RandAugment(num_ops=2, magnitude=10),
            transforms.ToTensor(),
            # 随机擦除（模拟 Cutout），参数：执行概率 50%，擦除区域面积比例，擦除区域长宽比
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0),
            transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
        ]
    )


def get_test_transform():
    """获取测试集预处理变换。

    测试时不做任何数据增强，仅进行基础的 ToTensor + Normalize，
    保证评估的一致性。

    Returns:
        torchvision.transforms.Compose 组合变换
    """
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
        ]
    )
