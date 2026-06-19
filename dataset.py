import numpy as np
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import Dataset


def split_labeled_unlabeled(targets, n_labeled_per_class, num_classes=10, seed=42):
    """按类别均匀划分有标签和无标签样本索引。

    FixMatch 的标准数据划分方式：
    - 从每个类别中随机采样 n_labeled_per_class 个样本作为有标签数据
    - 所有训练样本（包括已标注的）都作为无标签数据使用
      注意：有标签样本也会出现在无标签集中，这是 FixMatch 的设计，
      模型在有标签样本上也可以利用无监督信号

    Args:
        targets: 所有样本的标签列表
        n_labeled_per_class: 每个类别的有标签样本数量
        num_classes: 类别总数（CIFAR-10 为 10）
        seed: 随机种子，确保划分可复现

    Returns:
        labeled_idx: 有标签样本的索引数组
        unlabeled_idx: 无标签样本的索引数组（包含全部训练样本）
    """
    np.random.seed(seed)
    targets = np.array(targets)
    labeled_idx = []
    for i in range(num_classes):
        idx = np.where(targets == i)[0]
        idx = np.random.choice(idx, n_labeled_per_class, replace=False)
        labeled_idx.extend(idx)

    labeled_idx = np.array(labeled_idx)
    # FixMatch 中无标签集包含所有训练数据（含已标注的，但不使用其标签）
    unlabeled_idx = np.array(range(len(targets)))

    return labeled_idx, unlabeled_idx


class CIFAR10Unlabeled(datasets.CIFAR10):
    """CIFAR-10 无标签数据集。

    继承自 torchvision 的 CIFAR10，重写 __getitem__ 使其同时返回
    弱增强版本和强增强版本的图像（用于 FixMatch 的一致性正则化）。

    注意：此数据集不返回标签，因为无标签数据在训练中不使用真实标签。
    """

    def __init__(self, root, indices, train=True, transform_weak=None, transform_strong=None, download=False):
        super().__init__(root, train=train, download=download)
        # 仅保留指定索引的数据
        self.data = self.data[indices]
        self.targets = np.array(self.targets)[indices]
        self.transform_weak = transform_weak
        self.transform_strong = transform_strong

    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(img)

        # 弱增强：用于生成伪标签（如 RandomHorizontalFlip + RandomCrop）
        if self.transform_weak is not None:
            img_weak = self.transform_weak(img)
        else:
            # 即使没有增强变换，也至少转为 Tensor 以保持类型一致
            img_weak = transforms.ToTensor()(img)

        # 强增强：用于一致性正则化（如 RandAugment + RandomErasing）
        if self.transform_strong is not None:
            img_strong = self.transform_strong(img)
        else:
            img_strong = transforms.ToTensor()(img)

        return img_weak, img_strong


class CIFAR10Labeled(datasets.CIFAR10):
    """CIFAR-10 有标签数据集。

    继承自 torchvision 的 CIFAR10，重写 __getitem__ 使其返回
    经过弱增强的图像及其对应的真实标签。
    """

    def __init__(self, root, indices, train=True, transform=None, download=False):
        super().__init__(root, train=train, download=download)
        # 仅保留指定索引的数据
        self.data = self.data[indices]
        self.targets = np.array(self.targets)[indices]
        self.transform = transform

    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        else:
            # 即使没有增强变换，也至少转为 Tensor 以保持类型一致
            img = transforms.ToTensor()(img)
        target = self.targets[index]
        return img, target
