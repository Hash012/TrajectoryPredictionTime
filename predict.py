import torch
import numpy as np
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from model import VAE, Regressor

# ==================== 配置 ====================
# 使用GPU（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 获取当前脚本文件所在目录的绝对路径
# 动机: 使用动态路径计算，以确保脚本在不同环境下都能正确找到相关文件。
#       这样可以避免因项目移动或在不同机器上运行时产生的路径问题。
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 定义各个目录的路径
MODEL_DIR = os.path.join(CURRENT_DIR, 'trained_models')
DATA_DIR = os.path.join(CURRENT_DIR, 'datas')
RESULTS_DIR = os.path.join(CURRENT_DIR, 'prediction_results') # 新建一个专门存放结果的目录

# 确保结果目录存在
os.makedirs(RESULTS_DIR, exist_ok=True)

# ==================== 核心功能函数 ====================

def load_all_models_and_scalers(x_dim, z_dim):
    """
    加载所有必需的模型和标准化器。
    
    Args:
        x_dim (int): VAE的输入维度。
        z_dim (int): VAE的潜在空间维度。
        
    Returns:
        tuple: (vae, regressor, scaler_X_real, scaler_X_imag, scaler_y)
    """
    print("1. 正在加载所有模型和标准化器...")
    
    # --- 加载VAE ---
    vae = VAE(x_dim, z_dim)
    vae_checkpoint = torch.load(os.path.join(MODEL_DIR, 'vae_model.pth'), map_location=device, weights_only=True)
    vae.load_state_dict(vae_checkpoint['vae_state_dict'])
    vae.to(device)
    vae.eval() # 设置为评估模式
    
    # --- 加载回归器 ---
    regressor = Regressor(z_dim)
    regressor_checkpoint = torch.load(os.path.join(MODEL_DIR, 'regressor_model.pth'), map_location=device, weights_only=True)
    regressor.load_state_dict(regressor_checkpoint['regressor'])
    regressor.to(device)
    regressor.eval() # 设置为评估模式
    
    # --- 加载标准化器 ---
    scaler_X_real = joblib.load(os.path.join(MODEL_DIR, 'scaler_X_real.save'))
    scaler_X_imag = joblib.load(os.path.join(MODEL_DIR, 'scaler_X_imag.save'))
    scaler_y = joblib.load(os.path.join(MODEL_DIR, 'scaler_y.save'))
    
    print("   ...加载完成！")
    return vae, regressor, scaler_X_real, scaler_X_imag, scaler_y

def predict_time_from_trajectory(vae, regressor, scaler_X_real, scaler_X_imag, scaler_y, x_traj, y_traj):
    """
    根据单条轨迹(x, y)预测时间t（无打印输出）。
    
    Args:
        vae (nn.Module): 预训练的VAE模型。
        regressor (nn.Module): 预训练的回归器模型。
        scaler_X_real: 实部标准化器。
        scaler_X_imag: 虚部标准化器。
        scaler_y: 目标值标准化器。
        x_traj (np.ndarray): x坐标轨迹 (1D array)。
        y_traj (np.ndarray): y坐标轨迹 (1D array)。
        
    Returns:
        float: 预测的时间值。
    """
    with torch.no_grad():
        # --- 步骤 1: 将输入数据转换为复数 ---
        trajectory_complex = x_traj + 1j * y_traj
        
        # --- 步骤 2: 使用加载的scaler进行数据标准化 ---
        traj_real_std = scaler_X_real.transform(trajectory_complex.real.reshape(1, -1))
        traj_imag_std = scaler_X_imag.transform(trajectory_complex.imag.reshape(1, -1))
        
        # 重新组合
        trajectory_complex_std = traj_real_std + 1j * traj_imag_std
        
        # --- 步骤 3: 转换为PyTorch Tensor并移动到设备 ---
        trajectory_tensor = torch.from_numpy(trajectory_complex_std).to(dtype=torch.complex64, device=device)
        
        # --- 步骤 4: 使用VAE提取特征 (mu) ---
        _, mu, _, _ = vae(trajectory_tensor)
        
        # --- 步骤 5: 使用回归器进行预测 ---
        predicted_value_scaled = regressor(mu)
        
        # --- 步骤 6: 反向变换以得到原始尺度的时间 ---
        predicted_value_log = scaler_y.inverse_transform(predicted_value_scaled.cpu().numpy())
        predicted_time_final = np.expm1(predicted_value_log)

    return predicted_time_final[0][0]

# ==================== 主执行函数 ====================

def main():
    """
    主函数，执行加载、遍历测试集、预测并生成结果报告。
    """
    # --- 加载测试数据 ---
    print("正在加载所有测试数据...")
    try:
        x_df = pd.read_excel(os.path.join(DATA_DIR, 'xx.xlsx'), header=None)
        y_df = pd.read_excel(os.path.join(DATA_DIR, 'yy.xlsx'), header=None)
        t_df = pd.read_excel(os.path.join(DATA_DIR, 'tt.xlsx'), header=None)
        print(f"   ...加载了 {len(x_df)} 条测试样本。")
    except FileNotFoundError as e:
        print(f"\n错误: 测试数据文件未找到 - {e}")
        print(f"请确保 'xx.xlsx', 'yy.xlsx', 'tt.xlsx' 文件在 '{DATA_DIR}' 目录下。")
        return

    # 获取维度信息（从第一个样本）
    x_dim = x_df.iloc[0].values.shape[0]
    Z_DIM_from_training = 150 
    z_dim = min(Z_DIM_from_training, max(50, x_dim // 8))

    # --- 加载模型和标准化器 ---
    try:
        vae, regressor, scaler_X_real, scaler_X_imag, scaler_y = load_all_models_and_scalers(x_dim, z_dim)
    except FileNotFoundError as e:
        print(f"\n错误: 模型或标准化器文件未找到 - {e}")
        print(f"请确保所有必需的 '.pth' 和 '.save' 文件都在 '{MODEL_DIR}' 目录下。")
        return

    # --- 遍历测试集并进行预测 ---
    results = []
    print("\n2. 开始遍历测试集进行预测...")
    for i in range(len(x_df)):
        sample_x_traj = x_df.iloc[i].values
        sample_y_traj = y_df.iloc[i].values
        true_time = t_df.iloc[i].values[0]
        
        predicted_time = predict_time_from_trajectory(
            vae, regressor, 
            scaler_X_real, scaler_X_imag, scaler_y, 
            sample_x_traj, sample_y_traj
        )
        
        error = true_time - predicted_time
        
        results.append({
            '样本索引': i,
            '真实时间': true_time,
            '预测时间': predicted_time,
            '绝对误差': abs(error)
        })
        
        print(f"   - 样本 {i}: 真实值={true_time:.4f}, 预测值={predicted_time:.4f}, 误差={error:.4f}")

    # --- 创建并显示结果表格 ---
    results_df = pd.DataFrame(results)
    
    # --- 保存结果到CSV ---
    results_csv_path = os.path.join(RESULTS_DIR, 'prediction_results.csv')
    results_df.to_csv(results_csv_path, index=False, float_format='%.4f')
    
    print("\n================== 预测结果汇总 ==================")
    # 为了在终端更美观地显示，限制pandas的显示宽度
    pd.set_option('display.width', 120)
    print(results_df)
    print("==================================================")
    
    # --- 计算并显示总体指标 ---
    mean_abs_error = results_df['绝对误差'].mean()
    print("\n================== 性能评估 ======================")
    print(f"预测样本总数: {len(results_df)}")
    print(f"平均绝对误差 (MAE): {mean_abs_error:.4f}")
    print("==================================================")
    
    # --- 将结果表格保存为图片 ---
    fig, ax = plt.subplots(figsize=(8, len(results_df) * 0.3 + 1)) # 根据行数动态调整高度
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=results_df.values, colLabels=results_df.columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    plt.title("预测结果详情", fontsize=16, y=1.05) # 调整标题位置
    
    results_img_path = os.path.join(RESULTS_DIR, 'prediction_results.png')
    plt.savefig(results_img_path, bbox_inches='tight', dpi=200)
    
    print(f"\n详细预测结果已保存至CSV文件: {results_csv_path}")
    print(f"预测结果表格图片已保存至: {results_img_path}")


if __name__ == '__main__':
    main()
