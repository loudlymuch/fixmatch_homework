import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader

from dataset import split_labeled_unlabeled, CIFAR10Labeled, CIFAR10Unlabeled
from utils.augment import get_weak_transform, get_strong_transform, get_test_transform
from models.wrn import build_wrn28_2
from config import Config40 as args  # 可选择 Config40, Config250, Config4000

import numpy as np
import random
import os


def prepare_datasets(args):
    """准备有标签/无标签/测试数据集及对应的 DataLoader。

    FixMatch 的数据划分策略：
    - 从训练集中按类别均匀采样少量样本作为有标签数据
    - 所有训练数据（含已标注的）同时作为无标签数据使用（但不使用其标签）
    - 测试集保持独立，仅用于评估

    Args:
        args: 配置对象，需包含 data_dir, num_labeled, batch_size, mu 等字段

    Returns:
        (labeled_loader, unlabeled_loader, test_loader) 三元组
    """
    # 下载并加载原始 CIFAR-10 训练集
    print("正在加载 CIFAR-10 数据集...")
    base_dataset = datasets.CIFAR10(args.data_dir, train=True, download=True)
    print(f"数据集加载完成，共 {len(base_dataset)} 张训练图像")

    # 计算每个类别的标注样本数（均匀分配，例如 40 张标签 → 每类 4 张）
    n_labeled_per_class = args.num_labeled // 10
    lb_idx, ulb_idx = split_labeled_unlabeled(base_dataset.targets, n_labeled_per_class)

    # 有标签训练集：仅使用弱增强（RandomHorizontalFlip + RandomCrop）
    train_labeled_dataset = CIFAR10Labeled(
        root=args.data_dir,
        indices=lb_idx,
        train=True,
        transform=get_weak_transform()
    )

    # 无标签训练集：同时返回弱增强和强增强版本
    # 弱增强用于生成伪标签，强增强用于一致性正则化
    train_unlabeled_dataset = CIFAR10Unlabeled(
        root=args.data_dir,
        indices=ulb_idx,
        train=True,
        transform_weak=get_weak_transform(),
        transform_strong=get_strong_transform()
    )

    # 测试集：仅做基础预处理（ToTensor + Normalize），不做数据增强
    test_dataset = datasets.CIFAR10(
        root=args.data_dir,
        train=False,
        transform=get_test_transform(),
        download=True
    )

    # 有标签 DataLoader：drop_last=False 允许不完整的 batch
    # 原因：少量标签场景（如 40 张）下样本数可能不足一个 batch，drop_last=True 会导致 DataLoader 为空
    labeled_loader = DataLoader(
        train_labeled_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False
    )

    # 无标签 DataLoader：batch_size = 有标签的 mu 倍
    # FixMatch 使用大量无标签数据来提升半监督效果
    unlabeled_loader = DataLoader(
        train_unlabeled_dataset,
        batch_size=args.batch_size * args.mu,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )

    # 测试 DataLoader：不打乱顺序
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    return labeled_loader, unlabeled_loader, test_loader


def train(labeled_loader, unlabeled_loader, model, optimizer, scheduler, args):
    """执行一个 epoch 的 FixMatch 训练。

    FixMatch 核心算法流程：
    1. 对有标签数据计算标准交叉熵损失 L_s
    2. 对无标签数据的弱增强版本预测伪标签
    3. 仅保留置信度 >= threshold 的高置信度伪标签（mask 过滤）
    4. 对强增强版本计算与伪标签的交叉熵损失 L_u
    5. 总损失 = L_s + lambda_u * L_u

    Args:
        labeled_loader: 有标签数据加载器
        unlabeled_loader: 无标签数据加载器
        model: WideResNet 模型
        optimizer: SGD 优化器
        scheduler: 余弦退火学习率调度器（每个 step 调用一次）
        args: 配置对象

    Returns:
        (avg_loss_s, avg_loss_u): 本 epoch 的平均有监督损失和无监督损失
    """
    model.train()

    # 使用 iter 手动控制数据迭代，实现跨 epoch 的无限循环采样
    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)

    # 累计损失，用于计算 epoch 平均值
    total_loss_s = 0
    total_loss_u = 0

    for i in range(args.steps_per_epoch):
        # ---- 获取有标签 batch ----
        try:
            inputs_x, targets_x = next(labeled_iter)
        except StopIteration:
            # 数据用完则重新初始化迭代器，实现循环采样
            labeled_iter = iter(labeled_loader)
            inputs_x, targets_x = next(labeled_iter)

        # ---- 获取无标签 batch ----
        try:
            inputs_u_w, inputs_u_s = next(unlabeled_iter)
        except StopIteration:
            unlabeled_iter = iter(unlabeled_loader)
            inputs_u_w, inputs_u_s = next(unlabeled_iter)

        # ---- 移动到 GPU ----
        inputs_x, targets_x = inputs_x.to(args.device), targets_x.to(args.device)
        inputs_u_w, inputs_u_s = inputs_u_w.to(args.device), inputs_u_s.to(args.device)

        # ---- 有监督损失：标准交叉熵 ----
        logits_x = model(inputs_x)
        loss_s = F.cross_entropy(logits_x, targets_x, reduction='mean')

        # ---- 无监督损失：伪标签 + 一致性正则化 ----
        with torch.no_grad():
            # 对弱增强版本预测，生成伪标签
            logits_u_w = model(inputs_u_w)
            max_probs, pseudo_labels = torch.max(torch.softmax(logits_u_w, dim=1), dim=1)
            # 核心机制：仅保留置信度超过阈值的伪标签
            mask = max_probs.ge(args.threshold).float()

        # 对强增强版本计算与伪标签的交叉熵
        logits_u_s = model(inputs_u_s)
        loss_u_all = F.cross_entropy(logits_u_s, pseudo_labels, reduction='none')
        loss_u = (loss_u_all * mask).mean()  # mask 过滤掉低置信度样本

        # ---- 总损失加权求和 ----
        loss = loss_s + args.lambda_u * loss_u

        # ---- 反向传播与参数更新 ----
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ---- 学习率调度：每个 step 更新一次（余弦退火） ----
        scheduler.step()

        # 累计损失
        total_loss_s += loss_s.item()
        total_loss_u += loss_u.item()

    # 计算 epoch 平均损失
    avg_loss_s = total_loss_s / args.steps_per_epoch
    avg_loss_u = total_loss_u / args.steps_per_epoch

    return avg_loss_s, avg_loss_u


def validate(test_loader, model, args):
    """在测试集上评估模型准确率。

    Args:
        test_loader: 测试数据加载器
        model: 模型
        args: 配置对象（用于获取 device）

    Returns:
        accuracy: 测试准确率（百分比，0-100）
    """
    model.eval()  # 切换到评估模式（关闭 Dropout 和 BN 的运行时统计）
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = 100 * correct / total
    return accuracy


def main():
    """FixMatch 训练主函数。

    训练流程：
    1. 设置随机种子，确保实验可复现
    2. 准备数据集与 DataLoader（有标签 / 无标签 / 测试）
    3. 构建 WideResNet-28-2 模型
    4. 配置 SGD 优化器 + 余弦退火学习率调度
    5. 逐 epoch 训练，每个 epoch 后评估并保存最佳模型
    """
    # ---- 设置随机种子，确保实验可复现 ----
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    device = args.device
    print(f"正在使用设备: {device}")

    # ---- 准备数据集与 DataLoader ----
    labeled_loader, unlabeled_loader, test_loader = prepare_datasets(args)

    # ---- 构建 WideResNet-28-2 模型 ----
    model = build_wrn28_2(num_classes=10).to(device)

    # ---- 优化器：SGD + Nesterov 动量（FixMatch 论文标准配置） ----
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        nesterov=True,
        weight_decay=args.weight_decay
    )

    # ---- 学习率调度器：余弦退火（Cosine Annealing） ----
    # T_max 设为总训练步数，每个 step 调用一次 scheduler.step()
    total_steps = args.epochs * args.steps_per_epoch
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # ---- 训练循环 ----
    print("开始训练...")
    best_acc = 0.0
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs} 训练中...", flush=True)
        # 训练一个 epoch
        loss_s, loss_u = train(labeled_loader, unlabeled_loader, model, optimizer, scheduler, args)

        # 在测试集上评估
        acc = validate(test_loader, model, args)

        # 保存最佳模型（仅当准确率提升时）
        if acc > best_acc:
            best_acc = acc
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            torch.save(model.state_dict(), os.path.join(args.save_dir, args.save_name))
            print(f"新最佳模型保存，准确率: {best_acc:.2f}%")

        print(f"Epoch [{epoch+1}/{args.epochs}] - Loss_S: {loss_s:.4f}, Loss_U: {loss_u:.4f}, "
              f"Test Acc: {acc:.2f}%, Best Acc: {best_acc:.2f}%")

    print(f"训练完成，最佳测试准确率: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
