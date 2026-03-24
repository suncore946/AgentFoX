import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# 1. 准备数据
# 横坐标：JPEG压缩质量因子
x_labels = ['100', '90', '80', '70']

# 各个方法的ACC数据
data = {
    'DRCT':         [0.5340, 0.5359, 0.5665, 0.5535],
    'RINE':         [0.5404, 0.4415, 0.3723, 0.3674],
    'SPAI':         [0.5423, 0.5523, 0.5468, 0.5394],
    'PatchShuffle': [0.5025, 0.0426, 0.0194, 0.0129],
    'Our':          [0.7635, 0.7192, 0.7205, 0.7207]
}

# 2. 设置绘图风格 (可选，会让图表更好看)
plt.style.use('seaborn-v0_8-whitegrid') # 如果你的matplotlib版本较老，用 'seaborn-whitegrid'

# 创建画布
plt.figure(figsize=(8, 6))

# 3. 定义样式配置 (标记点形状, 颜色, 线型)
# 常用marker: 'o'圆, 's'方块, '^'三角, 'D'菱形, 'x'叉号
styles = {
    'DRCT':         {'marker': 's', 'color': '#1f77b4', 'linestyle': '--', 'label': 'DRCT'},
    'RINE':         {'marker': '^', 'color': '#ff7f0e', 'linestyle': '--', 'label': 'RINE'},
    'SPAI':         {'marker': 'D', 'color': '#2ca02c', 'linestyle': '--', 'label': 'SPAI'},
    'PatchShuffle': {'marker': 'x', 'color': '#9467bd', 'linestyle': '-.', 'label': 'PatchShuffle'},
    'Our':          {'marker': 'o', 'color': '#d62728', 'linestyle': '-',  'label': 'Our (AgentFoX)', 'linewidth': 3}
}

# 4. 循环绘制线条
for name, y_values in data.items():
    style = styles[name]
    plt.plot(x_labels, y_values, 
             marker=style['marker'], 
             color=style['color'], 
             linestyle=style['linestyle'],
             label=style['label'],
             linewidth=style.get('linewidth', 2),
             markersize=8)

# 5. 添加图表细节
# plt.title('Robustness against JPEG Compression', fontsize=14, fontweight='bold')
plt.xlabel('JPEG Quality Factor', fontsize=12)
plt.ylabel('F1 Score', fontsize=12)

# 设置Y轴范围 (根据数据范围调整，留一点余地)
plt.ylim(0, 1.0) 

# 添加网格
plt.grid(True, linestyle='--', alpha=0.7)

# 添加图例 (位置自动寻找最佳)
plt.legend(frameon=True, fontsize=10, loc='best')

# 6. 显示或保存
plt.tight_layout()

# 如果需要保存为高清图片用于论文，取消下面这行的注释
# plt.savefig('jpeg_robustness.png', dpi=300, bbox_inches='tight')

plt.show()
