import torch


class BaseConfig:
    """FixMatch 训练的基础配置类。

    定义所有公共超参数，特定标注数量的配置继承此类并覆写
    num_labeled 和 save_name。
    """

    # ---- 数据与设备 ----
    data_dir = './data'                                           # CIFAR-10 数据集下载/存储路径
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动选择 GPU/CPU

    save_dir = './saved_models'                                   # 模型保存目录
    log_dir = './logs'                                            # 训练日志（CSV）保存目录

    # ---- 训练超参数 ----
    batch_size = 64                                               # 有标签数据的批次大小
    mu = 7                                                        # 无标签数据倍数（无标签 batch = batch_size * mu）
    threshold = 0.95                                              # 伪标签置信度阈值（FixMatch 核心参数）
    lambda_u = 1.0                                                # 无监督损失权重

    # ---- 优化器参数 ----
    lr = 0.03                                                     # 初始学习率（SGD）
    momentum = 0.9                                                # SGD 动量
    weight_decay = 5e-4                                           # 权重衰减（L2 正则化系数）

    # ---- 训练长度 ----
    epochs = 50                                                   # 训练总轮数
    steps_per_epoch = 1024                                        # 每 epoch 的训练步数
    # 总训练步数 = 50 * 1024 = 51200
    warmup_steps = 0                                              # 学习率 warmup 步数（设为 0 则跳过 warmup）
    ema_decay = 0.999                                             # EMA 模型衰减率（设为 0 则禁用 EMA）


class Config40(BaseConfig):
    """40 张标签的 FixMatch 配置。

    每类仅 4 张有标签样本，是极度稀缺标注场景。
    CIFAR-10 共 50000 张训练图像，仅 40 张有标签。
    """
    num_labeled = 40
    save_name = 'fixmatch_40.pth'


class Config250(BaseConfig):
    """250 张标签的 FixMatch 配置。

    每类 25 张有标签样本，是中等稀缺标注场景。
    """
    num_labeled = 250
    save_name = 'fixmatch_250.pth'


class Config4000(BaseConfig):
    """4000 张标签的 FixMatch 配置。

    每类 400 张有标签样本，是较充足的标注场景。
    """
    num_labeled = 4000
    save_name = 'fixmatch_4000.pth'
