import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 设置绘图样式
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def analyze_target_variable(file_path: str):
    """
    分析目标变量t的分布，以诊断潜在的离群值问题。
    
    动机:
    回归模型的MSE和MAE指标异常巨大，R²和MAPE却表现良好，
    这强烈暗示目标变量t中可能存在极端离群值。
    通过可视化和统计分析，我们可以：
    1. 确认是否存在离群值。
    2. 了解数据的整体分布和数值范围。
    3. 为后续的数据处理或模型选择提供决策依据。
    """
    try:
        # 1. 加载数据
        print(f"正在加载数据文件: {file_path}")
        df = pd.read_excel(file_path, header=None)
        # 假设目标变量在第一列
        t_values = df.iloc[:, 0]
        print("数据加载完成。")
        print("-" * 50)

        # 2. 打印描述性统计信息
        print("目标变量 't' 的描述性统计信息:")
        print(t_values.describe())
        print("-" * 50)

        # 3. 打印排序后的头部和尾部值，以快速发现异常
        print("值最小的5个样本:")
        print(t_values.sort_values().head())
        print("\n值最大的5个样本:")
        print(t_values.sort_values(ascending=False).head())
        print("-" * 50)

        # 4. 创建可视化图表
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle("目标变量 't' 的分布分析", fontsize=16)

        # 绘制直方图 (Histogram)
        # 直方图可以显示数据的频率分布
        sns.histplot(t_values, kde=True, ax=axes[0], bins=30)
        axes[0].set_title("直方图 (Histogram)")
        axes[0].set_xlabel("t 值")
        axes[0].set_ylabel("频数")

        # 绘制箱形图 (Box Plot)
        # 箱形图能非常清晰地标示出离群值
        sns.boxplot(y=t_values, ax=axes[1])
        axes[1].set_title("箱形图 (Box Plot)")
        axes[1].set_ylabel("t 值")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # 保存图表
        save_path = 'target_variable_analysis.png'
        plt.savefig(save_path, dpi=200)
        print(f"分析图表已保存至: {save_path}")
        
        plt.show()

    except FileNotFoundError:
        print(f"错误: 文件未找到 at '{file_path}'")
    except Exception as e:
        print(f"分析过程中出现错误: {e}")

if __name__ == '__main__':
    # 目标变量t存储在tt.xlsx文件中
    file_to_analyze = 'datas/tt.xlsx'
    analyze_target_variable(file_to_analyze)
