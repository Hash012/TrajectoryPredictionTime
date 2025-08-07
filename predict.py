import torch
import numpy as np
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from model import VAE, Regressor
from dataload import get_dataloader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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
        predicted_time_final = scaler_y.inverse_transform(predicted_value_scaled.cpu().numpy())

    return predicted_time_final[0][0]

def predict_on_dataset(vae, regressor, scaler_y, dataloader, dataset_name):
    """
    对整个数据集进行预测并返回结果
    
    Args:
        vae: VAE模型
        regressor: 回归器模型  
        scaler_y: 目标值标准化器
        dataloader: 数据加载器
        dataset_name: 数据集名称（用于显示）
        
    Returns:
        dict: 包含预测结果和评估指标的字典
    """
    vae.eval()
    regressor.eval()
    
    predictions = []
    true_values = []
    
    print(f"\n开始对{dataset_name}进行预测...")
    
    with torch.no_grad():
        for batch_idx, (x_complex, y_true_normalized) in enumerate(dataloader):
            # 转换数据类型并移动到设备
            if isinstance(x_complex, np.ndarray):
                x_complex = torch.from_numpy(x_complex).to(dtype=torch.complex64, device=device)
            else:
                x_complex = x_complex.to(dtype=torch.complex64, device=device)
                
            if isinstance(y_true_normalized, np.ndarray):
                y_true_normalized = torch.from_numpy(y_true_normalized).to(dtype=torch.float32, device=device)
            else:
                y_true_normalized = y_true_normalized.to(dtype=torch.float32, device=device)
            
            # 使用VAE提取特征
            _, mu, _, _ = vae(x_complex)
            
            # 使用回归器进行预测
            pred_normalized = regressor(mu)
            
            # 反标准化预测值和真实值
            pred_original = scaler_y.inverse_transform(pred_normalized.cpu().numpy())
            true_original = scaler_y.inverse_transform(y_true_normalized.cpu().numpy())
            
            predictions.extend(pred_original.flatten())
            true_values.extend(true_original.flatten())
    
    # 计算评估指标
    predictions = np.array(predictions)
    true_values = np.array(true_values)
    
    mse = mean_squared_error(true_values, predictions)
    mae = mean_absolute_error(true_values, predictions)
    r2 = r2_score(true_values, predictions)
    
    # 计算MAPE
    def mean_absolute_percentage_error(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        mask = y_true != 0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    mape = mean_absolute_percentage_error(true_values, predictions)
    
    print(f"{dataset_name}预测完成:")
    print(f"  样本数: {len(predictions)}")
    print(f"  MSE: {mse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R²: {r2:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    
    return {
        'predictions': predictions,
        'true_values': true_values,
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'mape': mape,
        'sample_count': len(predictions)
    }

# ==================== 主执行函数 ====================

def create_detailed_result_table(results, dataset_name, max_display_rows=20):
    """
    创建详细的结果表格图片
    
    Args:
        results: 预测结果字典
        dataset_name: 数据集名称
        max_display_rows: 最多显示的行数
    """
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 准备表格数据
    true_values = results['true_values']
    predictions = results['predictions']
    errors = np.abs(np.array(predictions) - np.array(true_values))
    relative_errors = np.abs((np.array(predictions) - np.array(true_values)) / np.array(true_values)) * 100
    
    # 创建DataFrame
    df = pd.DataFrame({
        '样本序号': range(1, len(true_values) + 1),
        '真实值': true_values,
        '预测值': predictions,
        '绝对误差': errors,
        '相对误差(%)': relative_errors
    })
    
    # 如果样本太多，只显示前N行
    display_df = df.head(max_display_rows) if len(df) > max_display_rows else df
    
    # 格式化数值
    display_data = []
    for _, row in display_df.iterrows():
        display_data.append([
            int(row['样本序号']),
            f"{row['真实值']:.4f}",
            f"{row['预测值']:.4f}",
            f"{row['绝对误差']:.4f}",
            f"{row['相对误差(%)']:.2f}%"
        ])
    
    # 创建图形
    fig_height = min(max(len(display_data) * 0.4 + 3, 8), 20)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.axis('tight')
    ax.axis('off')
    
    # 创建表格
    table = ax.table(
        cellText=display_data,
        colLabels=['样本序号', '真实值', '预测值', '绝对误差', '相对误差'],
        loc='center',
        cellLoc='center'
    )
    
    # 设置表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # 设置表头样式
    for i in range(len(display_df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 交替行颜色
    for i in range(1, len(display_data) + 1):
        for j in range(len(display_df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
            else:
                table[(i, j)].set_facecolor('white')
    
    # 添加标题和统计信息
    title_text = f'{dataset_name}预测结果详细表格'
    if len(df) > max_display_rows:
        title_text += f'\n（显示前{max_display_rows}行，共{len(df)}行）'
    
    plt.title(title_text, fontsize=14, fontweight='bold', pad=20)
    
    # 在表格下方添加统计信息
    stats_text = f"""
统计指标：
总样本数: {results['sample_count']}    MSE: {results['mse']:.6f}    MAE: {results['mae']:.6f}    R²: {results['r2']:.6f}    MAPE: {results['mape']:.2f}%
平均绝对误差: {np.mean(errors):.6f}    最大绝对误差: {np.max(errors):.6f}    最小绝对误差: {np.min(errors):.6f}
    """
    
    fig.text(0.5, 0.02, stats_text, ha='center', va='bottom', fontsize=10,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.3))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # 为统计信息留出空间
    
    # 保存图片
    table_path = os.path.join(RESULTS_DIR, f'{dataset_name}_详细结果表格.png')
    plt.savefig(table_path, dpi=200, bbox_inches='tight')
    plt.show()
    
    print(f"\n{dataset_name}详细结果表格已保存至: {table_path}")
    return table_path

def visualize_prediction_results(train_results, test_results):
    """
    可视化训练集和测试集的预测结果对比
    """
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 1. 创建训练集详细表格
    create_detailed_result_table(train_results, "训练集", max_display_rows=25)
    
    # 2. 创建测试集详细表格
    create_detailed_result_table(test_results, "测试集", max_display_rows=25)
    
    # 3. 创建对比散点图
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 训练集散点图
    axes[0].scatter(train_results['true_values'], train_results['predictions'], 
                   alpha=0.6, color='blue', s=30, edgecolors='darkblue', linewidth=0.5)
    min_val = min(min(train_results['true_values']), min(train_results['predictions']))
    max_val = max(max(train_results['true_values']), max(train_results['predictions']))
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    axes[0].set_xlabel('真实值', fontsize=12)
    axes[0].set_ylabel('预测值', fontsize=12)
    axes[0].set_title(f'训练集预测精度\nR² = {train_results["r2"]:.4f}, MAE = {train_results["mae"]:.4f}', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # 测试集散点图
    axes[1].scatter(test_results['true_values'], test_results['predictions'], 
                   alpha=0.6, color='green', s=30, edgecolors='darkgreen', linewidth=0.5)
    min_val = min(min(test_results['true_values']), min(test_results['predictions']))
    max_val = max(max(test_results['true_values']), max(test_results['predictions']))
    axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    axes[1].set_xlabel('真实值', fontsize=12)
    axes[1].set_ylabel('预测值', fontsize=12)
    axes[1].set_title(f'测试集预测精度\nR² = {test_results["r2"]:.4f}, MAE = {test_results["mae"]:.4f}', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存散点图
    scatter_path = os.path.join(RESULTS_DIR, 'prediction_scatter_comparison.png')
    plt.savefig(scatter_path, dpi=200, bbox_inches='tight')
    plt.show()
    
    print(f"\n预测散点对比图已保存至: {scatter_path}")

def main():
    """
    主函数，分别对训练集和测试集进行预测并生成对比结果报告。
    """
    print("开始加载数据和模型...")
    
    # 使用与训练相同的参数加载数据
    BATCH_SIZE = 16
    train_loader, test_loader, scaler_X_real, scaler_X_imag, scaler_y = get_dataloader(
        DATA_DIR, batch_size=BATCH_SIZE, shuffle=False  # 预测时不打乱数据
    )
    
    # 获取维度信息
    sample_X, _ = next(iter(train_loader))
    x_dim = sample_X.shape[1]
    Z_DIM_from_training = 64 
    z_dim = min(Z_DIM_from_training, max(32, x_dim // 8))
    
    print(f"数据加载完成:")
    print(f"  输入维度: {x_dim}")
    print(f"  潜在空间维度: {z_dim}")
    
    # 加载模型和标准化器
    try:
        vae, regressor, scaler_X_real_loaded, scaler_X_imag_loaded, scaler_y_loaded = load_all_models_and_scalers(x_dim, z_dim)
    except FileNotFoundError as e:
        print(f"\n错误: 模型或标准化器文件未找到 - {e}")
        print(f"请确保所有必需的 '.pth' 和 '.save' 文件都在 '{MODEL_DIR}' 目录下。")
        return
    
    # 使用从文件加载的标准化器，这样确保与训练时一致
    print("\n使用训练时保存的标准化器进行预测...")
    
    # 分别对训练集和测试集进行预测
    train_results = predict_on_dataset(vae, regressor, scaler_y_loaded, train_loader, "训练集")
    test_results = predict_on_dataset(vae, regressor, scaler_y_loaded, test_loader, "测试集")
    
    # 分析目标值范围
    train_true = np.array(train_results['true_values'])
    test_true = np.array(test_results['true_values'])
    all_true = np.concatenate([train_true, test_true])
    
    print("\n" + "="*70)
    print("目标值统计分析")
    print("="*70)
    print(f"目标值范围: [{all_true.min():.2f}, {all_true.max():.2f}]")
    print(f"目标值均值: {all_true.mean():.2f}")
    print(f"目标值标准差: {all_true.std():.2f}")
    print(f"目标值中位数: {np.median(all_true):.2f}")
    
    # 计算相对误差
    train_relative_error = np.abs(train_results['predictions'] - train_true) / np.abs(train_true) * 100
    test_relative_error = np.abs(test_results['predictions'] - test_true) / np.abs(test_true) * 100
    
    print("\n误差分布分析:")
    print(f"训练集最大绝对误差: {np.max(np.abs(train_results['predictions'] - train_true)):.2f}")
    print(f"测试集最大绝对误差: {np.max(np.abs(test_results['predictions'] - test_true)):.2f}")
    print(f"训练集90%分位数误差: {np.percentile(np.abs(train_results['predictions'] - train_true), 90):.2f}")
    print(f"测试集90%分位数误差: {np.percentile(np.abs(test_results['predictions'] - test_true), 90):.2f}")
    
    # 保存详细结果到CSV
    train_df = pd.DataFrame({
        '真实值': train_results['true_values'],
        '预测值': train_results['predictions'],
        '绝对误差': np.abs(np.array(train_results['predictions']) - np.array(train_results['true_values'])),
        '相对误差(%)': train_relative_error
    })
    
    test_df = pd.DataFrame({
        '真实值': test_results['true_values'],
        '预测值': test_results['predictions'],
        '绝对误差': np.abs(np.array(test_results['predictions']) - np.array(test_results['true_values'])),
        '相对误差(%)': test_relative_error
    })
    
    # 保存结果到CSV文件
    train_csv_path = os.path.join(RESULTS_DIR, 'train_prediction_results.csv')
    test_csv_path = os.path.join(RESULTS_DIR, 'test_prediction_results.csv')
    
    train_df.to_csv(train_csv_path, index=False, float_format='%.6f')
    test_df.to_csv(test_csv_path, index=False, float_format='%.6f')
    
    print(f"\n训练集预测结果已保存至: {train_csv_path}")
    print(f"测试集预测结果已保存至: {test_csv_path}")
    
    # 输出总结
    print("\n" + "="*70)
    print("预测结果总结对比")
    print("="*70)
    print(f"{'指标':<15} {'训练集':<20} {'测试集':<20}")
    print("-" * 70)
    print(f"{'样本数':<15} {train_results['sample_count']:<20} {test_results['sample_count']:<20}")
    print(f"{'MSE':<15} {train_results['mse']:<20.6f} {test_results['mse']:<20.6f}")
    print(f"{'MAE':<15} {train_results['mae']:<20.6f} {test_results['mae']:<20.6f}")
    print(f"{'R²':<15} {train_results['r2']:<20.6f} {test_results['r2']:<20.6f}")
    print(f"{'MAPE (%)':<15} {train_results['mape']:<20.2f} {test_results['mape']:<20.2f}")
    print("="*70)
    
    # 生成可视化图表
    visualize_prediction_results(train_results, test_results)
    
    # 保存总结结果到JSON
    summary_results = {
        'train_results': {
            'mse': float(train_results['mse']),
            'mae': float(train_results['mae']),
            'r2': float(train_results['r2']),
            'mape': float(train_results['mape']),
            'sample_count': int(train_results['sample_count'])
        },
        'test_results': {
            'mse': float(test_results['mse']),
            'mae': float(test_results['mae']),
            'r2': float(test_results['r2']),
            'mape': float(test_results['mape']),
            'sample_count': int(test_results['sample_count'])
        }
    }
    
    summary_path = os.path.join(RESULTS_DIR, 'prediction_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        import json
        json.dump(summary_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n预测总结已保存至: {summary_path}")
    print("\n预测流程完成！")


if __name__ == '__main__':
    main()
