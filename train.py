import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler


from dataset import split_labeled_unlabeled, CIFAR10Labeled, CIFAR10Unlabeled
from utils.augment import get_weak_transform, get_strong_transform, get_test_transform
from models.wrn import build_wrn28_2
from config import Config40 as args  # 可选择 Config40, Config250, Config4000

import numpy as np
import random
import os
import csv


scaler = GradScaler()


class EMAModel:
    """指数移动平均（EMA）模型，用于生成更稳定的模型参数。

    EMA 按如下规则在每次 step 后更新影子参数：
        shadow_param = decay * shadow_param + (1 - decay) * param

    EMA 模型通常比原始模型泛化更好，适合用于最终评估。
    参考：Mean Teacher (Tarvainen & Valpola, 2017), FixMatch (Sohn et al., 2020)

    Args:
        model: 原始模型
        decay: EMA 衰减率（越大更新越慢，越接近历史平均），典型值 0.999
    """

    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}       # 存储影子参数
        self.step_count = 0
        self._initialized = False

    def _init_shadow(self):
        """首次调用时，将影子参数初始化为模型参数的深拷贝。"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()
        self._initialized = True

    def update(self):
        """在每次 optimizer.step() 之后调用，更新影子参数。"""
        if not self._initialized:
            self._init_shadow()
        self.step_count += 1
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    # shadow = decay * shadow + (1 - decay) * param
                    self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply_shadow(self):
        """将影子参数应用到模型（用于评估），返回原始参数备份以便恢复。"""
        if not self._initialized:
            self._init_shadow()
        backup = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
        return backup

    def restore(self, backup):
        """用备份恢复模型原始参数。"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(backup[name])


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
        num_workers=4,
        drop_last=False,
        pin_memory=True,
        persistent_workers=True, # 保持进程不被销毁，减少 epoch 切换开销
        prefetch_factor=2        # 每个 worker 预抓取 2 个 batch
    )

    # 无标签 DataLoader：batch_size = 有标签的 mu 倍
    # FixMatch 使用大量无标签数据来提升半监督效果
    unlabeled_loader = DataLoader(
        train_unlabeled_dataset,
        batch_size=args.batch_size * args.mu,
        shuffle=True,
        num_workers=4,
        drop_last=True,
        pin_memory=True,
        persistent_workers=True, # 保持进程不被销毁，减少 epoch 切换开销
        prefetch_factor=2        # 每个 worker 预抓取 2 个 batch
    )

    # 测试 DataLoader：不打乱顺序
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True, # 保持进程不被销毁，减少 epoch 切换开销
        prefetch_factor=2        # 每个 worker 预抓取 2 个 batch
    )
    return labeled_loader, unlabeled_loader, test_loader


def create_warmup_scheduler(optimizer, args):
    """创建 warmup + 余弦退火的学习率调度器。

    Warmup 阶段：学习率从 0 线性增长到 args.lr（共 warmup_steps 步）
    余弦退火阶段：学习率从 args.lr 按余弦曲线衰减到 0

    Args:
        optimizer: 优化器
        args: 配置对象（需包含 warmup_steps, epochs, steps_per_epoch）

    Returns:
        scheduler: LambdaLR 调度器，每个 step 调用一次
    """
    total_steps = args.epochs * args.steps_per_epoch
    warmup_steps = args.warmup_steps

    if warmup_steps <= 0:
        # 无 warmup，直接使用余弦退火
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # 线性 warmup：从 0 增长到 1
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # 余弦退火：从 1 衰减到 0
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + np.cos(np.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(labeled_loader, unlabeled_loader, model, optimizer, scheduler, args, ema_model=None):
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
        scheduler: 学习率调度器（每个 step 调用一次）
        args: 配置对象
        ema_model: EMA 模型（可选），传入后每个 step 自动更新影子参数

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

        # if i % 50 == 0:
        #     print(f"{i}：正在训练")

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

        optimizer.zero_grad()

        with autocast(): # 开启自动混合精度
            # 1. 有监督部分
            logits_x = model(inputs_x)
            loss_s = F.cross_entropy(logits_x, targets_x)
            
            # 2. 无监督部分 (伪标签生成和强增强损失)
            with torch.no_grad():
                logits_u_w = model(inputs_u_w)
                max_probs, pseudo_labels = torch.max(torch.softmax(logits_u_w, dim=1), dim=1)
                mask = max_probs.ge(args.threshold).float()

            logits_u_s = model(inputs_u_s)
            loss_u = (F.cross_entropy(logits_u_s, pseudo_labels, reduction='none') * mask).mean()
            
            loss = loss_s + args.lambda_u * loss_u

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # ---- EMA 模型更新：在参数更新后进行指数移动平均 ----
        if ema_model is not None:
            ema_model.update()

        # ---- 学习率调度：每个 step 更新一次 ----
        scheduler.step()

        # 累计损失
        total_loss_s += loss_s.item()
        total_loss_u += loss_u.item()

    # 计算 epoch 平均损失
    avg_loss_s = total_loss_s / args.steps_per_epoch
    avg_loss_u = total_loss_u / args.steps_per_epoch

    return avg_loss_s, avg_loss_u


def validate(test_loader, model, args, ema_model=None):
    """在测试集上评估模型准确率。

    Args:
        test_loader: 测试数据加载器
        model: 原始模型（当 ema_model 为 None 时使用）
        args: 配置对象（用于获取 device）
        ema_model: EMA 模型（可选），传入后使用 EMA 影子参数进行评估

    Returns:
        accuracy: 测试准确率（百分比，0-100）
    """
    model.eval()  # 切换到评估模式（关闭 Dropout 和 BN 的运行时统计）

    # 如果提供了 EMA 模型，将影子参数加载到模型中
    backup = None
    if ema_model is not None:
        backup = ema_model.apply_shadow()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    # 恢复原始参数
    if backup is not None:
        ema_model.restore(backup)

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

    # ---- 学习率调度器：Warmup + 余弦退火 ----
    # warmup_steps 步内线性增长到 lr，之后按余弦曲线衰减
    scheduler = create_warmup_scheduler(optimizer, args)

    # ---- EMA 模型：指数移动平均，用于更稳定的评估 ----
    ema_model = None
    if args.ema_decay > 0:
        ema_model = EMAModel(model, args.ema_decay)
        print(f"已启用 EMA 模型，decay = {args.ema_decay}")

    # ---- 训练循环 ----
    # 创建日志目录并打开 CSV 文件记录训练指标
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    log_name = args.save_name.replace('.pth', '.csv')
    log_path = os.path.join(args.log_dir, log_name)

    # 根据是否启用 EMA 决定 CSV 列
    if ema_model is not None:
        fieldnames = ['epoch', 'loss_s', 'loss_u', 'test_acc', 'ema_acc', 'best_acc', 'lr']
    else:
        fieldnames = ['epoch', 'loss_s', 'loss_u', 'test_acc', 'best_acc', 'lr']

    csv_file = open(log_path, 'w', newline='')
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    print(f"训练日志将保存至: {log_path}")

    print("开始训练...")
    best_acc = 0.0
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs} 训练中...", flush=True)
        # 训练一个 epoch
        loss_s, loss_u = train(labeled_loader, unlabeled_loader, model, optimizer, scheduler, args,
                               ema_model=ema_model)

        # 在测试集上评估（原始模型）
        acc = validate(test_loader, model, args)

        # 在测试集上评估（EMA 模型）
        ema_acc = validate(test_loader, model, args, ema_model=ema_model) if ema_model is not None else acc

        # 保存最佳模型（基于 EMA 准确率，若禁用 EMA 则基于原始准确率）
        save_acc = ema_acc if ema_model is not None else acc
        if save_acc > best_acc:
            best_acc = save_acc
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            # 保存时使用 EMA 影子参数
            backup = ema_model.apply_shadow() if ema_model is not None else None
            torch.save(model.state_dict(), os.path.join(args.save_dir, args.save_name))
            if backup is not None:
                ema_model.restore(backup)
            print(f"新最佳模型保存，准确率: {best_acc:.2f}%")

        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']

        # 写入 CSV 日志
        row = {
            'epoch': epoch + 1,
            'loss_s': f'{loss_s:.4f}',
            'loss_u': f'{loss_u:.4f}',
            'test_acc': f'{acc:.2f}',
            'best_acc': f'{best_acc:.2f}',
            'lr': f'{current_lr:.6f}',
        }
        if ema_model is not None:
            row['ema_acc'] = f'{ema_acc:.2f}'
        writer.writerow(row)
        csv_file.flush()  # 每个 epoch 后刷新到磁盘，防止中断丢失数据

        if ema_model is not None:
            print(f"Epoch [{epoch+1}/{args.epochs}] - Loss_S: {loss_s:.4f}, Loss_U: {loss_u:.4f}, "
                  f"Raw Acc: {acc:.2f}%, EMA Acc: {ema_acc:.2f}%, Best Acc: {best_acc:.2f}%")
        else:
            print(f"Epoch [{epoch+1}/{args.epochs}] - Loss_S: {loss_s:.4f}, Loss_U: {loss_u:.4f}, "
                  f"Test Acc: {acc:.2f}%, Best Acc: {best_acc:.2f}%")

    csv_file.close()
    print(f"训练完成，最佳测试准确率: {best_acc:.2f}%")
    print(f"训练日志已保存至: {log_path}")


if __name__ == "__main__":
    main()
