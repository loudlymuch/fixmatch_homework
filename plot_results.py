"""生成 FixMatch 实验结果的对比图表。"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['mathtext.default'] = 'regular'

# 读取三个实验的 CSV 日志
df40  = pd.read_csv('data/fixmatch_40.csv')
df250 = pd.read_csv('data/fixmatch_250.csv')
df4000 = pd.read_csv('data/fixmatch_4000.csv')

colors = {'40 labels': '#E74C3C', '250 labels': '#2ECC71', '4000 labels': '#3498DB'}

# ==================== 图 1: 测试准确率 vs Epoch ====================
fig1, ax1 = plt.subplots(figsize=(12, 7))
ax1.plot(df40['epoch'],  df40['test_acc'],  color=colors['40 labels'],   alpha=0.3, linewidth=0.8)
ax1.plot(df250['epoch'], df250['test_acc'], color=colors['250 labels'],  alpha=0.3, linewidth=0.8)
ax1.plot(df4000['epoch'],df4000['test_acc'],color=colors['4000 labels'], alpha=0.3, linewidth=0.8)

# 平滑曲线 (window=10)
for df, label, c in [(df40, '40 labels', colors['40 labels']),
                      (df250, '250 labels', colors['250 labels']),
                      (df4000, '4000 labels', colors['4000 labels'])]:
    smoothed = df['test_acc'].rolling(window=15, center=True, min_periods=1).mean()
    ax1.plot(df['epoch'], smoothed, color=c, linewidth=2.5, label=label)

ax1.set_xlabel('Epoch', fontsize=13)
ax1.set_ylabel('Test Accuracy (%)', fontsize=13)
ax1.set_title('FixMatch Test Accuracy on CIFAR-10 (smoothed, window=15)', fontsize=14)
ax1.legend(fontsize=12, loc='lower right')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, max(df250['epoch'].max(), df4000['epoch'].max()))
fig1.tight_layout()
fig1.savefig('data/accuracy_comparison.png', dpi=150)
print("图1已保存: data/accuracy_comparison.png")

# ==================== 图 2: EMA 准确率对比 ====================
fig2, ax2 = plt.subplots(figsize=(12, 7))
for df, label, c in [(df40, '40 labels', colors['40 labels']),
                      (df250, '250 labels', colors['250 labels']),
                      (df4000, '4000 labels', colors['4000 labels'])]:
    smoothed = df['ema_acc'].rolling(window=15, center=True, min_periods=1).mean()
    ax2.plot(df['epoch'], smoothed, color=c, linewidth=2.5, label=label)

ax2.set_xlabel('Epoch', fontsize=13)
ax2.set_ylabel('EMA Accuracy (%)', fontsize=13)
ax2.set_title('FixMatch EMA Model Accuracy on CIFAR-10 (smoothed, window=15)', fontsize=14)
ax2.legend(fontsize=12, loc='lower right')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, max(df250['epoch'].max(), df4000['epoch'].max()))
fig2.tight_layout()
fig2.savefig('data/ema_accuracy_comparison.png', dpi=150)
print("图2已保存: data/ema_accuracy_comparison.png")

# ==================== 图 3: 损失函数对比 ====================
fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(16, 6))

# 有监督损失
for df, label, c in [(df40, '40 labels', colors['40 labels']),
                      (df250, '250 labels', colors['250 labels']),
                      (df4000, '4000 labels', colors['4000 labels'])]:
    sm = df['loss_s'].rolling(window=15, center=True, min_periods=1).mean()
    ax3a.plot(df['epoch'], sm, color=c, linewidth=2, label=label)
ax3a.set_xlabel('Epoch', fontsize=12)
ax3a.set_ylabel('Supervised Loss $L_s$', fontsize=12)
ax3a.set_title('Supervised Loss $L_s$ (smoothed)', fontsize=13)
ax3a.legend(fontsize=11)
ax3a.grid(True, alpha=0.3)
ax3a.set_yscale('log')

# 无监督损失
for df, label, c in [(df40, '40 labels', colors['40 labels']),
                      (df250, '250 labels', colors['250 labels']),
                      (df4000, '4000 labels', colors['4000 labels'])]:
    sm = df['loss_u'].rolling(window=15, center=True, min_periods=1).mean()
    ax3b.plot(df['epoch'], sm, color=c, linewidth=2, label=label)
ax3b.set_xlabel('Epoch', fontsize=12)
ax3b.set_ylabel('Unsupervised Loss $L_u$', fontsize=12)
ax3b.set_title('Unsupervised Loss $L_u$ (smoothed)', fontsize=13)
ax3b.legend(fontsize=11)
ax3b.grid(True, alpha=0.3)

fig3.tight_layout()
fig3.savefig('data/loss_comparison.png', dpi=150)
print("图3已保存: data/loss_comparison.png")

# ==================== 图 4: 最佳准确率柱状图 ====================
fig4, ax4 = plt.subplots(figsize=(8, 6))
configs = ['40 labels\n(4/class)', '250 labels\n(25/class)', '4000 labels\n(400/class)']
best_test  = [df40['test_acc'].max(),  df250['test_acc'].max(),  df4000['test_acc'].max()]
best_ema   = [df40['ema_acc'].max(),   df250['ema_acc'].max(),   df4000['ema_acc'].max()]
best_best  = [df40['best_acc'].max(),  df250['best_acc'].max(),  df4000['best_acc'].max()]

x = np.arange(len(configs))
w = 0.25
bars1 = ax4.bar(x - w, best_test,  w, label='Max Test Acc',      color='#E74C3C', edgecolor='white')
bars2 = ax4.bar(x,     best_ema,   w, label='Max EMA Acc',       color='#2ECC71', edgecolor='white')
bars3 = ax4.bar(x + w, best_best,  w, label='Best Saved (EMA)',  color='#3498DB', edgecolor='white')

for bar in bars1:
    ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
             f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
for bar in bars2:
    ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
             f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
for bar in bars3:
    ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
             f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax4.set_xticks(x)
ax4.set_xticklabels(configs, fontsize=12)
ax4.set_ylabel('Accuracy (%)', fontsize=13)
ax4.set_title('FixMatch Best Accuracy Comparison on CIFAR-10', fontsize=14)
ax4.legend(fontsize=11, loc='lower right')
ax4.set_ylim(0, 100)
ax4.grid(True, alpha=0.3, axis='y')
fig4.tight_layout()
fig4.savefig('data/best_accuracy_bar.png', dpi=150)
print("图4已保存: data/best_accuracy_bar.png")

# ==================== 图 5: 学习率衰减曲线 ====================
fig5, ax5 = plt.subplots(figsize=(10, 5))
ax5.plot(df40['epoch'],  df40['lr'],  color=colors['40 labels'],   linewidth=2, label='40 labels')
ax5.plot(df250['epoch'], df250['lr'], color=colors['250 labels'],  linewidth=2, label='250 labels')
ax5.plot(df4000['epoch'],df4000['lr'],color=colors['4000 labels'], linewidth=2, label='4000 labels')
ax5.set_xlabel('Epoch', fontsize=13)
ax5.set_ylabel('Learning Rate', fontsize=13)
ax5.set_title('Learning Rate Schedule (Cosine Annealing)', fontsize=14)
ax5.legend(fontsize=12)
ax5.grid(True, alpha=0.3)
fig5.tight_layout()
fig5.savefig('data/lr_schedule.png', dpi=150)
print("图5已保存: data/lr_schedule.png")

# ==================== 图 6: 合成大图 (2x2) ====================
fig6, axes = plt.subplots(2, 2, figsize=(16, 12))

# 6a: Test Accuracy
ax = axes[0, 0]
for df, label, c in [(df40, '40 labels', colors['40 labels']),
                      (df250, '250 labels', colors['250 labels']),
                      (df4000, '4000 labels', colors['4000 labels'])]:
    sm = df['test_acc'].rolling(window=15, center=True, min_periods=1).mean()
    ax.plot(df['epoch'], sm, color=c, linewidth=2, label=label)
ax.set_title('Test Accuracy (smoothed)', fontsize=13)
ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy (%)')
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

# 6b: EMA Accuracy
ax = axes[0, 1]
for df, label, c in [(df40, '40 labels', colors['40 labels']),
                      (df250, '250 labels', colors['250 labels']),
                      (df4000, '4000 labels', colors['4000 labels'])]:
    sm = df['ema_acc'].rolling(window=15, center=True, min_periods=1).mean()
    ax.plot(df['epoch'], sm, color=c, linewidth=2, label=label)
ax.set_title('EMA Accuracy (smoothed)', fontsize=13)
ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy (%)')
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

# 6c: Supervised Loss
ax = axes[1, 0]
for df, label, c in [(df40, '40 labels', colors['40 labels']),
                      (df250, '250 labels', colors['250 labels']),
                      (df4000, '4000 labels', colors['4000 labels'])]:
    sm = df['loss_s'].rolling(window=15, center=True, min_periods=1).mean()
    ax.plot(df['epoch'], sm, color=c, linewidth=2, label=label)
ax.set_title('Supervised Loss $L_s$ (smoothed, log scale)', fontsize=13)
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
ax.set_yscale('log')

# 6d: Unsupervised Loss
ax = axes[1, 1]
for df, label, c in [(df40, '40 labels', colors['40 labels']),
                      (df250, '250 labels', colors['250 labels']),
                      (df4000, '4000 labels', colors['4000 labels'])]:
    sm = df['loss_u'].rolling(window=15, center=True, min_periods=1).mean()
    ax.plot(df['epoch'], sm, color=c, linewidth=2, label=label)
ax.set_title('Unsupervised Loss $L_u$ (smoothed)', fontsize=13)
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

fig6.suptitle('FixMatch Training Dynamics on CIFAR-10 — Full Comparison', fontsize=15, fontweight='bold')
fig6.tight_layout()
fig6.savefig('data/full_comparison.png', dpi=150)
print("图6已保存: data/full_comparison.png")

# ==================== 关键数值汇总 ====================
print("\n" + "="*70)
print("关键实验指标汇总")
print("="*70)
for name, df in [('FixMatch-40', df40), ('FixMatch-250', df250), ('FixMatch-4000', df4000)]:
    print(f"\n{name}:")
    print(f"  总 Epoch 数:       {len(df)}")
    print(f"  最高 Test Acc:     {df['test_acc'].max():.2f}% (epoch {df['test_acc'].idxmax()+1})")
    print(f"  最高 EMA Acc:      {df['ema_acc'].max():.2f}% (epoch {df['ema_acc'].idxmax()+1})")
    print(f"  最佳保存 Acc:      {df['best_acc'].max():.2f}%")
    print(f"  最终 Test Acc:     {df['test_acc'].iloc[-1]:.2f}%")
    print(f"  最终 EMA Acc:      {df['ema_acc'].iloc[-1]:.2f}%")
    print(f"  初始 L_s:          {df['loss_s'].iloc[0]:.4f}")
    print(f"  最终 L_s:          {df['loss_s'].iloc[-1]:.6f}")
    print(f"  初始 L_u:          {df['loss_u'].iloc[0]:.4f}")
    print(f"  最终 L_u:          {df['loss_u'].iloc[-1]:.4f}")
    # 关键里程碑
    for ep_name, ep_num in [('Epoch 10', 10), ('Epoch 50', 50), ('Epoch 100', 100)]:
        idx = min(ep_num-1, len(df)-1)
        print(f"  {ep_name} Test Acc:   {df['test_acc'].iloc[idx]:.2f}%")

plt.close('all')
print("\n所有图表生成完毕！")
