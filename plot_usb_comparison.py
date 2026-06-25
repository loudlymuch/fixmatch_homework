"""生成自有实现 vs USB 库的对比图表"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['mathtext.default'] = 'regular'

# 读取自有实现日志
my40  = pd.read_csv('data/fixmatch_40.csv')
my250 = pd.read_csv('data/fixmatch_250.csv')
my4000 = pd.read_csv('data/fixmatch_4000.csv')

# 读取 USB 实现日志
usb40  = pd.read_csv('data/usb_fixmatch_40.csv')
usb250 = pd.read_csv('data/usb_fixmatch_250.csv')
usb4000 = pd.read_csv('data/usb_fixmatch_4000.csv')

# 颜色方案: 自有=暖色, USB=冷色
COLORS = {
    'my40':  '#E74C3C', 'usb40':  '#C0392B',
    'my250': '#2ECC71', 'usb250': '#27AE60',
    'my4000':'#3498DB', 'usb4000':'#2980B9',
}

def smooth(series, window=15):
    return series.rolling(window=window, center=True, min_periods=1).mean()

# ==================== 图 1: 准确率对比 (三行布局) ====================
fig1, axes = plt.subplots(3, 1, figsize=(14, 14))

configs = [
    (40,  my40,  usb40,  axes[0], '40 labels (4/class)'),
    (250, my250, usb250, axes[1], '250 labels (25/class)'),
    (4000,my4000,usb4000,axes[2], '4000 labels (400/class)'),
]

for nl, my, usb, ax, title in configs:
    # 自有实现用 epoch 作为 x 轴
    ax.plot(my['epoch'], smooth(my['test_acc']),
            color=COLORS[f'my{nl}'], linewidth=2.5, label='Ours (Test Acc)')
    ax.plot(my['epoch'], smooth(my['ema_acc']),
            color=COLORS[f'my{nl}'], linewidth=2.5, linestyle='--', alpha=0.7, label='Ours (EMA Acc)')

    # USB 用 iter 作为 x 轴，对齐为 epoch 等价 (每 epoch 1024 步)
    usb_epoch = usb['iter'] / 1024
    ax.plot(usb_epoch, smooth(usb['test_acc']),
            color=COLORS[f'usb{nl}'], linewidth=2.5, label='USB (Test Acc)')
    ax.plot(usb_epoch, smooth(usb['ema_acc']),
            color=COLORS[f'usb{nl}'], linewidth=2.5, linestyle='--', alpha=0.7, label='USB (EMA Acc)')

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch (or Iter/1024 for USB)', fontsize=11)
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1050)

fig1.suptitle('FixMatch on CIFAR-10: Our Implementation vs USB Library', fontsize=16, fontweight='bold')
fig1.tight_layout()
fig1.savefig('data/usb_comparison_accuracy.png', dpi=150)
print("图1已保存: data/usb_comparison_accuracy.png")

# ==================== 图 2: 损失函数对比 (三行两列) ====================
fig2, axes2 = plt.subplots(3, 2, figsize=(16, 14))

for row, (nl, my, usb, _, title) in enumerate(configs):
    my_epoch = my['epoch']
    usb_epoch = usb['iter'] / 1024

    # L_s
    ax = axes2[row, 0]
    ax.plot(my_epoch, smooth(my['loss_s']), color=COLORS[f'my{nl}'], linewidth=2, label='Ours $L_s$')
    ax.plot(usb_epoch, smooth(usb['loss_s']), color=COLORS[f'usb{nl}'], linewidth=2, label='USB $L_s$')
    ax.set_title(f'{title} — Supervised Loss', fontsize=12)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    ax.set_xlim(0, 1050)

    # L_u
    ax = axes2[row, 1]
    ax.plot(my_epoch, smooth(my['loss_u']), color=COLORS[f'my{nl}'], linewidth=2, label='Ours $L_u$')
    ax.plot(usb_epoch, smooth(usb['loss_u']), color=COLORS[f'usb{nl}'], linewidth=2, label='USB $L_u$')
    ax.set_title(f'{title} — Unsupervised Loss', fontsize=12)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1050)

fig2.suptitle('Loss Comparison: Our Implementation vs USB Library', fontsize=16, fontweight='bold')
fig2.tight_layout()
fig2.savefig('data/usb_comparison_loss.png', dpi=150)
print("图2已保存: data/usb_comparison_loss.png")

# ==================== 图 3: 柱状图 - 最佳准确率对比 ====================
fig3, ax3 = plt.subplots(figsize=(10, 7))

x = np.arange(3)
width = 0.3

my_best  = [my40['best_acc'].max(),  my250['best_acc'].max(),  my4000['best_acc'].max()]
usb_best = [usb40['best_acc'].max(), usb250['best_acc'].max(), usb4000['best_acc'].max()]

bars1 = ax3.bar(x - width/2, my_best,  width, label='Our Implementation',
                color=['#E74C3C', '#2ECC71', '#3498DB'], edgecolor='white', linewidth=1.5)
bars2 = ax3.bar(x + width/2, usb_best, width, label='USB Library',
                color=['#C0392B', '#27AE60', '#2980B9'], edgecolor='white', linewidth=1.5,
                hatch='///', alpha=0.85)

for bar, val in zip(bars1, my_best):
    ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.8,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
for bar, val in zip(bars2, usb_best):
    ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.8,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

# 差距标注
for i, (m, u) in enumerate(zip(my_best, usb_best)):
    gap = u - m
    ax3.annotate(f'Δ={gap:.1f}%', xy=(i, (m+u)/2),
                 fontsize=10, ha='center', va='center',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.9))

ax3.set_xticks(x)
ax3.set_xticklabels(['40 labels\n(4/class)', '250 labels\n(25/class)', '4000 labels\n(400/class)'],
                    fontsize=12)
ax3.set_ylabel('Best Accuracy (%)', fontsize=13)
ax3.set_title('FixMatch on CIFAR-10: Our Implementation vs USB Library (EMA Best)', fontsize=14)
ax3.legend(fontsize=12)
ax3.set_ylim(0, 105)
ax3.grid(True, alpha=0.3, axis='y')
fig3.tight_layout()
fig3.savefig('data/usb_best_comparison_bar.png', dpi=150)
print("图3已保存: data/usb_best_comparison_bar.png")

# ==================== 图 4: 汇总对比大图 ====================
fig4 = plt.figure(figsize=(18, 12))

# 左上: Test Accuracy 所有曲线
ax = fig4.add_subplot(2, 3, 1)
for nl, my, usb, _, title in configs:
    ax.plot(my['epoch'], smooth(my['test_acc']), color=COLORS[f'my{nl}'], linewidth=2, label=f'Ours-{nl}')
    ax.plot(usb['iter']/1024, smooth(usb['test_acc']), color=COLORS[f'usb{nl}'], linewidth=2,
            linestyle='--', label=f'USB-{nl}')
ax.set_title('Test Accuracy (All)', fontsize=12)
ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy (%)')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_xlim(0, 1050)

# 中上: EMA Accuracy 所有曲线
ax = fig4.add_subplot(2, 3, 2)
for nl, my, usb, _, title in configs:
    ax.plot(my['epoch'], smooth(my['ema_acc']), color=COLORS[f'my{nl}'], linewidth=2, label=f'Ours-{nl}')
    ax.plot(usb['iter']/1024, smooth(usb['ema_acc']), color=COLORS[f'usb{nl}'], linewidth=2,
            linestyle='--', label=f'USB-{nl}')
ax.set_title('EMA Accuracy (All)', fontsize=12)
ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy (%)')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_xlim(0, 1050)

# 右上: 最佳准确率对比柱状图
ax = fig4.add_subplot(2, 3, 3)
x = np.arange(3)
ax.bar(x - width/2, my_best, width, label='Ours', color=['#E74C3C','#2ECC71','#3498DB'], edgecolor='white')
ax.bar(x + width/2, usb_best, width, label='USB', color=['#C0392B','#27AE60','#2980B9'],
       edgecolor='white', hatch='///')
for i, (m, u) in enumerate(zip(my_best, usb_best)):
    ax.annotate(f'Δ={u-m:.1f}%', xy=(i, min(m,u)-3), fontsize=9, ha='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
ax.set_xticks(x)
ax.set_xticklabels(['40 labels', '250 labels', '4000 labels'])
ax.set_title('Best Accuracy Comparison', fontsize=12)
ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='y')

# 左下: L_s 对比 (对数)
ax = fig4.add_subplot(2, 3, 4)
for nl, my, usb, _, title in configs:
    ax.plot(my['epoch'], smooth(my['loss_s']), color=COLORS[f'my{nl}'], linewidth=1.5, alpha=0.8, label=f'Ours-{nl}')
    ax.plot(usb['iter']/1024, smooth(usb['loss_s']), color=COLORS[f'usb{nl}'], linewidth=1.5,
            linestyle='--', alpha=0.8, label=f'USB-{nl}')
ax.set_title('Supervised Loss $L_s$', fontsize=12)
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
ax.set_yscale('log'); ax.set_xlim(0, 1050)

# 中下: L_u 对比
ax = fig4.add_subplot(2, 3, 5)
for nl, my, usb, _, title in configs:
    ax.plot(my['epoch'], smooth(my['loss_u']), color=COLORS[f'my{nl}'], linewidth=1.5, alpha=0.8, label=f'Ours-{nl}')
    ax.plot(usb['iter']/1024, smooth(usb['loss_u']), color=COLORS[f'usb{nl}'], linewidth=1.5,
            linestyle='--', alpha=0.8, label=f'USB-{nl}')
ax.set_title('Unsupervised Loss $L_u$', fontsize=12)
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_xlim(0, 1050)

# 右下: 差距随标注量变化
ax = fig4.add_subplot(2, 3, 6)
label_counts = [40, 250, 4000]
gaps = [u - m for m, u in zip(my_best, usb_best)]
ax.plot(label_counts, gaps, 'o-', color='#8E44AD', linewidth=2.5, markersize=12, markerfacecolor='#8E44AD')
for lc, g in zip(label_counts, gaps):
    ax.annotate(f'{g:.1f}%', xy=(lc, g), xytext=(lc+30, g+0.5),
                fontsize=10, ha='center',
                arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))
ax.set_xlabel('Number of Labeled Samples', fontsize=11)
ax.set_ylabel('Accuracy Gap (USB - Ours)', fontsize=11)
ax.set_title('Performance Gap vs Labeled Data', fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_xscale('log')
ax.set_xticks(label_counts)
ax.set_xticklabels([str(l) for l in label_counts])
ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)

fig4.suptitle('FixMatch on CIFAR-10: Comprehensive Comparison — Ours vs USB Library',
              fontsize=15, fontweight='bold')
fig4.tight_layout()
fig4.savefig('data/usb_full_comparison.png', dpi=150)
print("图4已保存: data/usb_full_comparison.png")

# ==================== 终端输出汇总 ====================
print("\n" + "="*70)
print("自有实现 vs USB 库 对比汇总")
print("="*70)
for nl, my, usb, _, title in configs:
    print(f"\n--- {title} ---")
    print(f"  {'指标':<25} {'自有实现':>12} {'USB库':>12} {'差距':>10}")
    print(f"  {'-'*55}")
    print(f"  {'最佳 Test Acc':<25} {my['test_acc'].max():>10.2f}% {usb['test_acc'].max():>10.2f}% {usb['test_acc'].max()-my['test_acc'].max():>8.2f}%")
    print(f"  {'最佳 EMA Acc':<25} {my['ema_acc'].max():>10.2f}% {usb['ema_acc'].max():>10.2f}% {usb['ema_acc'].max()-my['ema_acc'].max():>8.2f}%")
    print(f"  {'最佳保存 Acc':<25} {my['best_acc'].max():>10.2f}% {usb['best_acc'].max():>10.2f}% {usb['best_acc'].max()-my['best_acc'].max():>8.2f}%")
    print(f"  {'最终 L_s':<25} {my['loss_s'].iloc[-1]:>10.4f}  {usb['loss_s'].iloc[-1]:>10.4f}")
    print(f"  {'最终 L_u':<25} {my['loss_u'].iloc[-1]:>10.4f}  {usb['loss_u'].iloc[-1]:>10.4f}")
    print(f"  {'训练步数':<25} {len(my)*1024:>10,}  {usb['iter'].max():>10,}")

plt.close('all')
print("\n所有对比图表生成完毕！")
