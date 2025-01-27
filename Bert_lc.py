import matplotlib.pyplot as plt
import numpy as np
import os
import re
from matplotlib.ticker import MaxNLocator

# 配置参数（可根据需要调整）
CONFIG = {
    "input_log": "training_log.txt",          # 输入日志文件路径
    "output_dir": "d:\\fake_news\\result",    # 输出目录
    "output_name": "BERT_Training_Analysis",  # 输出文件名(不含扩展名)
    
    # 可视化参数
    "figure_size": (18, 9),                   # 图像尺寸(英寸)
    "dpi": 150,                               # 输出分辨率
    "color": {
        "raw_loss": "#1f77b480",              # 原始损失颜色(带透明度)
        "trend_line": "#d62728",              # 趋势线颜色
        "epoch_line": "#2ca02c",              # Epoch分界线颜色
        "highlight": ["#fff59d", "#c8e6c9"]   # 阶段高亮颜色
    },
    
    # 数据处理参数
    "downsample_rate": 50,                    # 降采样比率(1表示不降采样)
    "smooth_window": 500,                     # 滑动平均窗口大小
    "phase_boundaries": [0, 5000, 40000]      # 阶段分界步数
}

def parse_training_log(log_path):
    """解析训练日志文件"""
    step_data = []
    val_acc = []
    
    # 编译正则表达式
    step_pattern = re.compile(
        r"Epoch (\d+) \| Step (\d+)/\d+ \| Loss: (\d+\.\d+)"
    )
    epoch_pattern = re.compile(
        r"Epoch (\d+)/\d+ \| Train Loss: \d+\.\d+ \| Val Acc: (\d+\.\d+)"
    )

    try:
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                # 解析训练步骤
                if step_match := step_pattern.search(line):
                    epoch = int(step_match.group(1))
                    step = int(step_match.group(2))
                    loss = float(step_match.group(3))
                    # 计算全局步数
                    global_step = (epoch-1)*11481 + step
                    step_data.append((global_step, loss))
                
                # 解析验证准确率
                if epoch_match := epoch_pattern.search(line):
                    val_acc.append(float(epoch_match.group(2)))
                    
    except FileNotFoundError:
        print(f"错误：找不到日志文件 {log_path}")
        exit(1)
    except Exception as e:
        print(f"解析日志时发生错误：{str(e)}")
        exit(1)

    # 转换为numpy数组
    steps = np.array([x[0] for x in step_data])
    losses = np.array([x[1] for x in step_data])
    return steps, losses, np.array(val_acc)

def create_visualization(steps, losses, val_acc, config):
    """创建训练过程可视化图表"""
    # 准备输出目录
    os.makedirs(config["output_dir"], exist_ok=True)
    
    # 创建画布
    plt.figure(figsize=config["figure_size"], dpi=config["dpi"])
    ax = plt.gca()
    
    # 降采样数据
    sample_mask = slice(None, None, config["downsample_rate"])
    sampled_steps = steps[sample_mask]
    sampled_losses = losses[sample_mask]
    
    # 绘制原始损失（半透明）
    ax.plot(
        sampled_steps,
        sampled_losses,
        color=config["color"]["raw_loss"],
        alpha=0.3,
        label="Step Loss"
    )
    
    # 计算滑动平均
    window = config["smooth_window"]
    trend_loss = np.convolve(losses, np.ones(window)/window, "valid")
    trend_steps = steps[:len(trend_loss)]
    
    # 绘制趋势线
    ax.plot(
        trend_steps,
        trend_loss,
        color=config["color"]["trend_line"],
        linewidth=2,
        label=f"{window}-Step Moving Average"
    )
    
    # 标注阶段区域
    phases = config["phase_boundaries"] + [steps[-1]]
    for i in range(len(phases)-1):
        ax.axvspan(
            phases[i],
            phases[i+1],
            facecolor=config["color"]["highlight"][i%2],
            alpha=0.1
        )
    
    # 标注Epoch信息
    epoch_steps = [11481 * i for i in range(1, 6)]
    for idx, (step, acc) in enumerate(zip(epoch_steps, val_acc)):
        ax.axvline(
            step,
            color=config["color"]["epoch_line"],
            linestyle="--",
            alpha=0.7
        )
        ax.text(
            step * 1.02,
            ax.get_ylim()[1] * 0.85 - idx*0.1,
            f"Epoch {idx+1}\nAcc: {acc:.1%}",
            ha="left",
            va="top",
            fontsize=10,
            bbox=dict(
                boxstyle="round",
                facecolor="white",
                alpha=0.9,
                edgecolor=config["color"]["epoch_line"]
            )
        )
    
    # 设置坐标轴
    ax.set_xlabel("Training Steps", fontsize=12, labelpad=10)
    ax.set_ylabel("Loss", fontsize=12, labelpad=10)
    ax.set_title(
        "BERT Training Process Analysis\n"
        f"(Showing {len(sampled_steps):,} of {len(steps):,} steps)",
        fontsize=14,
        pad=20
    )
    ax.set_ylim(0, min(2.0, losses.max()*1.1))
    ax.xaxis.set_major_locator(MaxNLocator(10))
    
    # 添加次坐标轴显示epoch
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(epoch_steps)
    ax2.set_xticklabels([f"Epoch {i+1}" for i in range(5)])
    ax2.tick_params(axis="x", colors=config["color"]["epoch_line"])
    ax2.spines["top"].set_color(config["color"]["epoch_line"])
    
    # 添加图例
    ax.legend(
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="gray"
    )
    
    # 保存图像
    output_path = os.path.join(
        config["output_dir"],
        f"{config['output_name']}.png"
    )
    plt.savefig(
        output_path,
        bbox_inches="tight",
        facecolor="white",
        dpi=config["dpi"]
    )
    plt.close()
    
    print(f"可视化图表已保存至：{output_path}")
    return output_path

if __name__ == "__main__":
    # 解析日志数据
    steps, losses, val_acc = parse_training_log(CONFIG["input_log"])
    
    # 生成可视化图表
    output_file = create_visualization(steps, losses, val_acc, CONFIG)
    
    # 显示完成信息
    print(f"数据统计：")
    print(f"- 总训练步数：{len(steps):,}")
    print(f"- 最小损失值：{losses.min():.4f}")
    print(f"- 最大损失值：{losses.max():.4f}")
    print(f"- 最终验证准确率：{val_acc[-1]:.1%}")