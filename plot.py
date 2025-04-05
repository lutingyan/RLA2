import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_from_csv(csv_path, rolling_window=100):
    # 读取CSV文件
    df = pd.read_csv(csv_path)

    # 获取所有不同的学习率
    learning_rates = df['learning_rate'].unique()

    # 设置颜色映射（可根据学习率个数扩展）
    colors = ['red', 'green', 'blue']

    plt.figure(figsize=(12, 6))

    for i, lr in enumerate(learning_rates):
        # 筛选当前学习率的数据
        lr_data = df[df['learning_rate'] == lr]
        # 计算滚动平均和标准差
        reward_rolling = lr_data['average_reward'].rolling(rolling_window, min_periods=1).mean()
        std_rolling = lr_data['std_reward'].rolling(rolling_window, min_periods=1).mean()

        # 绘制曲线（以 episode 为横坐标）
        plt.plot(lr_data['episode'], reward_rolling,
                 linewidth=2, label=f"LR={lr} (Smoothed)", color=colors[i])

        # 绘制标准差区域
        plt.fill_between(lr_data['episode'],
                         reward_rolling - std_rolling,
                         reward_rolling + std_rolling,
                         color=colors[i], alpha=0.1, label=f"LR={lr} (Std Dev)")

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("REINFORCE Performance by Learning Rate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 使用示例
plot_from_csv('../results/reinforce_resultslr.csv', rolling_window=100)
