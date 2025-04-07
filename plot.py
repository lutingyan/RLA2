import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_from_npy(npy_path, rolling_window=100):
    # 读取.npy文件
    all_returns_per_step = np.load(npy_path)

    # 计算每个步骤的平均值和标准差
    average_per_step = np.nanmean(all_returns_per_step, axis=0)
    std_per_step = np.nanstd(all_returns_per_step, axis=0)

    # 设置绘图
    plt.figure(figsize=(12, 6))

    # 计算滚动平均
    reward_rolling = np.convolve(average_per_step, np.ones(rolling_window)/rolling_window, mode='valid')
    std_rolling = np.convolve(std_per_step, np.ones(rolling_window)/rolling_window, mode='valid')

    # 绘制曲线
    plt.plot(np.arange(len(reward_rolling)), reward_rolling, linewidth=2, label="Smoothed Average Reward", color='blue')

    # 绘制标准差区域
    plt.fill_between(np.arange(len(reward_rolling)),
                     reward_rolling - std_rolling,
                     reward_rolling + std_rolling,
                     color='blue', alpha=0.1, label="Std Dev")

    # 设置标签
    plt.xlabel("Steps")
    plt.ylabel("Reward")
    plt.title("REINFORCE Performance by Step (From .npy)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_from_csv(csv_path, rolling_window=100):
    # 读取CSV文件
    df = pd.read_csv(csv_path)

    # 获取所有不同的学习率
    gammas = df['gamma'].unique()

    # 设置颜色映射（可根据学习率个数扩展）
    colors = ['red', 'green', 'blue']

    plt.figure(figsize=(12, 6))

    for i, lr in enumerate(gammas):
        # 筛选当前学习率的数据
        lr_data = df[df['gamma'] == lr]
        # 计算滚动平均和标准差
        reward_rolling = lr_data['avg_reward'].rolling(rolling_window, min_periods=1).mean()
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

# plot_from_csv('results/reinforce_gamma_results.csv', rolling_window=50)
# plot_from_npy('results/TN_ER_softupdate_decay.npy', rolling_window=50)
