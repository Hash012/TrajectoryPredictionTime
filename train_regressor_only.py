# 仅训练回归器的脚本
import torch
import os
import torch.nn as nn
import torch.optim as optim
from dataload import get_dataloader
from model import VAE, Regressor
import joblib
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import traceback

# GPU设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 获取当前脚本文件所在目录的绝对路径
# 动机: 使用动态路径计算，以确保脚本在不同环境下都能正确找到相关文件。
#       这样可以避免因项目移动或在不同机器上运行时产生的路径问题。
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 参数
DATA_DIR = os.path.join(CURRENT_DIR, 'datas')
MODELS_DIR = os.path.join(CURRENT_DIR, 'trained_models')
RESULTS_DIR = os.path.join(CURRENT_DIR, 'training_results')
BATCH_SIZE = 128
REGRESSOR_EPOCHS = 1000
Z_DIM = 64

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=0.1)
        torch.nn.init.constant_(m.bias, 0.01)

def load_vae_model(x_dim, z_dim):
    """加载预训练的VAE模型"""
    print("正在加载VAE模型...")
    z_dim = min(Z_DIM, max(32, x_dim // 8))
    vae = VAE(x_dim, z_dim)
    checkpoint = torch.load(os.path.join(MODELS_DIR, 'vae_model.pth'), map_location='cpu', weights_only=True)
    vae.load_state_dict(checkpoint['vae_state_dict'])
    vae = vae.to(device)
    vae.eval()
    print("VAE模型加载完成！")
    return vae

def train_regressor_epoch(regressor, loader, optimizer, epoch, vae):
    """训练回归器一个epoch"""
    regressor.train()
    vae.eval()
    
    total_loss = 0
    valid_batches = 0
    
    for batch_idx, (x, t) in enumerate(loader):
        try:
            if isinstance(x, np.ndarray):
                x_complex = torch.from_numpy(x).to(dtype=torch.complex64, device=device)
            else:
                x_complex = x.to(dtype=torch.complex64, device=device)
            
            if isinstance(t, np.ndarray):
                t = torch.from_numpy(t).to(dtype=torch.float32, device=device)
            else:
                t = t.to(dtype=torch.float32, device=device)
            # 直接用标准化后的t
            with torch.no_grad():
                _, mu, _, _ = vae(x_complex)
            pred_time = regressor(mu)
            loss = nn.HuberLoss()(pred_time, t.view_as(pred_time))
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
                
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(regressor.parameters(), max_norm=5.0)
            optimizer.step()
            
            total_loss += loss.item() * x.size(0)
            valid_batches += x.size(0)
            
        except Exception as e:
            print(f"!! 训练时出错, batch {batch_idx}: {e}")
            traceback.print_exc()
            continue
    
    if valid_batches > 0:
        avg_loss = total_loss / valid_batches
    else:
        avg_loss = 0
    
    return avg_loss, valid_batches

def mean_absolute_percentage_error(y_true, y_pred): 
    """计算MAPE"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # 避免除以零
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def test_regressor_model(regressor, test_loader, vae, scaler_y):
    """
    测试回归器模型性能
    
    参数:
        regressor: 训练好的回归器模型
        test_loader: 测试数据加载器
        vae: 预训练的VAE模型
        scaler_y: 输出标准化器
        
    返回:
        dict: 包含测试结果的字典
    """
    print("\n" + "="*50)
    print("回归器模型测试")
    print("="*50)
    
    regressor.eval()
    vae.eval()
    
    valid_batches = 0
    
    predictions = []
    true_values = []
    
    with torch.no_grad():
        for batch_idx, (x, t) in enumerate(test_loader):
            try:
                if isinstance(x, np.ndarray):
                    x_complex = torch.from_numpy(x).to(dtype=torch.complex64, device=device)
                else:
                    x_complex = x.to(dtype=torch.complex64, device=device)
                
                if isinstance(t, np.ndarray):
                    t = torch.from_numpy(t).to(dtype=torch.float32, device=device)
                else:
                    t = t.to(dtype=torch.float32, device=device)
                
                _, mu, _, _ = vae(x_complex)
                pred_time = regressor(mu)
                
                pred_time_original = scaler_y.inverse_transform(pred_time.cpu().numpy().reshape(-1, 1))
                true_time_original = scaler_y.inverse_transform(t.cpu().numpy().reshape(-1, 1))
                
                valid_batches += x.size(0)
                
                predictions.extend(pred_time_original.flatten())
                true_values.extend(true_time_original.flatten())
                
            except Exception as e:
                print(f"!! 测试时出错, batch {batch_idx}: {e}")
                traceback.print_exc()
                continue
    
    if valid_batches > 0:
        mse = mean_squared_error(true_values, predictions)
        mae = mean_absolute_error(true_values, predictions)
        r2 = r2_score(true_values, predictions)
        mape = mean_absolute_percentage_error(true_values, predictions)
    else:
        mse = mae = r2 = mape = 0

    print(f"测试集MSE: {mse:.6f}")
    print(f"测试集MAE: {mae:.6f}")
    print(f"测试集R²: {r2:.6f}")
    print(f"测试集MAPE: {mape:.2f}%")
    print(f"测试样本数: {valid_batches}")
    
    return {
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'mape': mape,
        'samples': valid_batches,
        'predictions': predictions,
        'true_values': true_values
    }

def visualize_regressor_results(regressor_history, test_results):
    """
    可视化回归器训练结果
    """
    print("\n生成可视化图表...")
    
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 训练损失曲线
    ax1.plot(regressor_history['epoch'], regressor_history['loss'], 'b-', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('回归器训练损失')
    ax1.grid(True, alpha=0.3)
    
    # 2. 预测值 vs 真实值
    if len(test_results['predictions']) > 0:
        ax2.scatter(test_results['true_values'], test_results['predictions'], alpha=0.6, color='blue', s=20)
        min_val = min(min(test_results['true_values']), min(test_results['predictions']))
        max_val = max(max(test_results['true_values']), max(test_results['predictions']))
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        ax2.set_xlabel('True Values')
        ax2.set_ylabel('Predictions')
        ax2.set_title(f'预测精度 (R²={test_results["r2"]:.3f})')
        ax2.grid(True, alpha=0.3)
    
    # 3. 残差图
    if len(test_results['predictions']) > 0:
        residuals = np.array(test_results['predictions']) - np.array(test_results['true_values'])
        ax3.scatter(test_results['predictions'], residuals, alpha=0.6, color='green', s=20)
        ax3.axhline(y=0, color='r', linestyle='--')
        ax3.set_xlabel('Predictions')
        ax3.set_ylabel('Residuals')
        ax3.set_title('残差分析')
        ax3.grid(True, alpha=0.3)
    
    # 4. 测试结果总结
    metrics_text = f"""
回归器测试结果:
MSE: {test_results['mse']:.2e}
MAE: {test_results['mae']:.2f}
R²: {test_results['r2']:.4f}
MAPE: {test_results['mape']:.2f}%
测试样本: {test_results['samples']}
    """
    ax4.text(0.1, 0.5, metrics_text, transform=ax4.transAxes, fontsize=12,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.set_title('测试结果')
    ax4.axis('off')
    
    plt.tight_layout()
    # 定义图片保存路径
    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_path = os.path.join(RESULTS_DIR, 'regressor_training_results.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()
    
    print(f"可视化完成！图片已保存为: {save_path}")

def save_regressor_model(regressor, scaler_y, regressor_history, test_results):
    """保存回归器模型和测试结果"""
    print("\n保存回归器模型和结果...")
    
    # 定义文件保存路径
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, 'regressor_model.pth')
    scaler_y_path = os.path.join(MODELS_DIR, 'scaler_y.save')
    history_path = os.path.join(RESULTS_DIR, 'regressor_training_history.json')
    test_results_path = os.path.join(RESULTS_DIR, 'regressor_test_results.json')

    # 保存模型
    torch.save({'regressor': regressor.state_dict()}, model_path)
    joblib.dump(scaler_y, scaler_y_path)
    
    # 保存训练历史
    with open(history_path, 'w') as f:
        json.dump(regressor_history, f, indent=2)
    
    # 保存测试结果
    test_results_save = {
        'mse': test_results['mse'],
        'mae': test_results['mae'],
        'r2': test_results['r2'],
        'samples': test_results['samples']
    }
    with open(test_results_path, 'w') as f:
        json.dump(test_results_save, f, indent=2)
    
    print("回归器模型和结果已保存！")

if __name__ == '__main__':
    print("开始训练回归器...")
    
    # 加载数据
    train_loader, test_loader, scaler_X_real, scaler_X_imag, scaler_y = get_dataloader(DATA_DIR, batch_size=BATCH_SIZE)
    sample_X, _ = next(iter(train_loader))
    x_dim = sample_X.shape[1]
    
    # 加载预训练的VAE
    z_dim = min(Z_DIM, max(32, x_dim // 8))
    vae = load_vae_model(x_dim, z_dim)
    
    regressor = Regressor(z_dim)
    regressor = regressor.to(device)
    regressor.apply(init_weights)
    
    optimizer = optim.AdamW(regressor.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-5)
    regressor_history = {'epoch': [], 'loss': []}
    
    for epoch in range(REGRESSOR_EPOCHS):
        avg_loss, valid_batches = train_regressor_epoch(regressor, train_loader, optimizer, epoch, vae)
        
        regressor_history['epoch'].append(epoch + 1)
        regressor_history['loss'].append(avg_loss)
        scheduler.step()
        
        if epoch % 100 == 0:
            print(f"Regressor Epoch {epoch+1}, Loss: {avg_loss:.6f}")
    
    print("回归器训练完成！")
    
    test_results = test_regressor_model(regressor, test_loader, vae, scaler_y)
    
    visualize_regressor_results(regressor_history, test_results)
    
    save_regressor_model(regressor, scaler_y, regressor_history, test_results)
    
    print("\n回归器训练和测试流程完成！")
