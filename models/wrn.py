import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """WideResNet 的基本残差块。

    采用 BN → LeakyReLU → Conv 的预激活顺序（pre-activation），
    与原始 WideResNet 论文保持一致。
    """
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0):
        super(BasicBlock, self).__init__()

        # 第一个卷积单元：BN → LeakyReLU → Conv3x3
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.LeakyReLU(0.1, inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

        # 第二个卷积单元：BN → LeakyReLU → Conv3x3（步长固定为 1）
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.drop_rate = drop_rate
        self.equalInOut = (in_planes == out_planes)

        # 当输入输出通道数不同或步长不为 1 时，需要 1×1 卷积对齐维度
        self.shortcut = nn.Sequential()
        if not self.equalInOut or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)
            )

    def forward(self, x):
        # 预激活 + 第一个卷积
        out = self.relu1(self.bn1(x))

        # 如果通道数不匹配，跳跃连接也需要变换（这里复用 out 作为 shortcut 输入）
        if not self.equalInOut:
            x = out

        # 第二个卷积
        out = self.relu2(self.bn2(self.conv1(out)))
        # Dropout 正则化（仅在 drop_rate > 0 时生效）
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)

        out = self.conv2(out)
        # 残差连接：维度匹配时直接相加，否则经过 1×1 卷积对齐
        out = torch.add(x if self.equalInOut else self.shortcut(x), out)
        return out


class WideResNet(nn.Module):
    """FixMatch 使用的主干网络：WideResNet-28-2。

    深度 depth=28，宽度因子 widen_factor=2。
    标准结构：一个初始卷积 + 三个 layer 组 + BN + 全局平均池化 + 全连接分类层。
    """
    def __init__(self, depth=28, widen_factor=2, num_classes=10, drop_rate=0.0):
        super(WideResNet, self).__init__()
        # 根据深度计算每层的残差块数量：n = (depth - 4) / 6
        n = (depth - 4) // 6
        k = widen_factor

        # 各阶段通道数：[16, 16*k, 32*k, 64*k]
        n_channels = [16, 16 * k, 32 * k, 64 * k]

        # 初始卷积层（3×3，步长 1，不改变空间尺寸）
        self.conv1 = nn.Conv2d(3, n_channels[0], kernel_size=3, stride=1, padding=1, bias=False)

        # 三个残差层组，空间尺寸逐步减半
        self.layer1 = self._make_layer(n_channels[0], n_channels[1], n, stride=1, drop_rate=drop_rate)
        self.layer2 = self._make_layer(n_channels[1], n_channels[2], n, stride=2, drop_rate=drop_rate)
        self.layer3 = self._make_layer(n_channels[2], n_channels[3], n, stride=2, drop_rate=drop_rate)

        # 最终 BN + 激活
        self.bn1 = nn.BatchNorm2d(n_channels[3])
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        # 全连接分类层
        self.fc = nn.Linear(n_channels[3], num_classes)

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_planes, out_planes, num_blocks, stride, drop_rate=0.0):
        """构建一个残差层组。

        Args:
            in_planes: 输入通道数
            out_planes: 输出通道数
            num_blocks: 残差块数量
            stride: 第一个残差块的步长（控制下采样）
            drop_rate: Dropout 概率
        """
        layers = []
        for i in range(num_blocks):
            # 第一个残差块使用指定的 in_planes 和 stride
            # 后续残差块的输入通道数 = out_planes，步长 = 1
            layers.append(BasicBlock(
                in_planes if i == 0 else out_planes,
                out_planes,
                stride if i == 0 else 1,
                drop_rate=drop_rate
            ))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.relu(self.bn1(out))
        # 全局平均池化（8×8 → 1×1）
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        return self.fc(out)


def build_wrn28_2(num_classes=10, drop_rate=0.0):
    """构建 WideResNet-28-2 模型的工厂函数。

    Args:
        num_classes: 分类类别数（CIFAR-10 为 10）
        drop_rate: Dropout 概率（FixMatch 论文中通常为 0.0）

    Returns:
        WideResNet 实例
    """
    return WideResNet(depth=28, widen_factor=2, num_classes=num_classes, drop_rate=drop_rate)
