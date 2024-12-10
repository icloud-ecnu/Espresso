import matplotlib.pyplot as plt
from pipeline_model import cal_pipeline_time

# 更新后的画图函数
def plot_pipeline_improved(op, startTime, time):
    # print("SSS")
    # 初始化图像和坐标轴
    fig, ax = plt.subplots(figsize=(16, 3))
    plt.subplots_adjust(left=0.05, right=0.95)  # 根据需要调整这些值
    # 每个stage的高度
    stage_height = 0.5
    # 每个block的高度
    block_height = 0.5
    
    # 遍历每个stage和每个操作
    for stage_index, (stage_op, stage_startTime, stage_time) in enumerate(zip(op, startTime, time)):
        for op_index, (op_type, start, end) in enumerate(zip(stage_op, stage_startTime, stage_time)):
            # 设置颜色
            color = 'blue' if op_type == 'F' else 'green'
            # 画矩形，增加黑色边框
            ax.add_patch(plt.Rectangle((start, stage_index * stage_height), end - start, block_height, 
                                    edgecolor='black', linewidth=1, facecolor=color))

    # 移除坐标轴的标签
    ax.set_xticks([])
    ax.set_yticks([i * stage_height + block_height / 2 for i in range(len(op))])
    ax.set_yticklabels(['Stage {}'.format(i) for i in range(len(op))])
    ax.tick_params(axis='y', which='both', left=False)  # 移除y轴刻度线

    # 增加网格线
    ax.grid(True, axis='x', color='gray', linestyle='--', linewidth=0.5)

    # 设置坐标轴的界限
    max_pos = 0
    for x in time:
        max_pos = max(max_pos,max(x))
    ax.set_xlim(0, max_pos +1)
    ax.set_ylim(0, len(op) * stage_height)
    
    # 调整格子大小以按比例缩小，每个格子都有黑色边框
    for rect in ax.patches:
        # rect.set_height(rect.get_height() * 0.8)  # 缩小高度
        rect.set_y(rect.get_y() + (block_height * 0.1))  # 调整y位置以居中

    # 移除坐标轴的边框
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.savefig('pipeline.png', dpi=300)
    plt.show()


fwd=[0.691302,0.695241,0.702519,0.771321,0.831282]
bwd=[2.160882,2.161524,2.176281,2.383767,2.32209]
n = len(fwd)
ratio = 1
for i in range(n):
    fwd[i] = fwd[i] * ratio
    bwd[i] = bwd[i] * ratio
start,end,op = cal_pipeline_time(fwd,bwd,4,need=True)
plot_pipeline_improved(op,start,end)