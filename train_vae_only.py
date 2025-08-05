# 仅训练VAE的脚本
import torch
import os
import torch.nn as nn
import torch.optim as optim
from dataload import get_dataloader
from model import VAE, ComplexLinear, ComplexLinear
import joblib
import json
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Dict, List, Tuple, Optional, Any, Union

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# GPU设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"使用设备: {device}")

# 获取当前脚本文件所在目录的绝对路径
# 动机: 使用动态路径计算，以确保脚本在不同环境下都能正确找到相关文件。
#       这样可以避免因项目移动或在不同机器上运行时产生的路径问题。
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 参数
DATA_DIR = os.path.join(CURRENT_DIR, 'datas')
MODELS_DIR = os.path.join(CURRENT_DIR, 'trained_models')
RESULTS_DIR = os.path.join(CURRENT_DIR, 'training_results')
BATCH_SIZE = 16
VAE_EPOCHS = 500
Z_DIM = 150  # 从20增加到100，减少维度压缩比 

def init_weights(m: nn.Module) -> None:
    """初始化网络权重为Kaiming Normal，并确保能处理ComplexLinear"""
    if isinstance(m, nn.Linear): # 适用于普通实数层
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, ComplexLinear): # 专门处理我们自定义的复数层
        # 对内部的实部和虚部Linear层分别进行Kaiming初始化
        torch.nn.init.kaiming_normal_(m.fc_real.weight, mode='fan_in', nonlinearity='relu')
        torch.nn.init.kaiming_normal_(m.fc_imag.weight, mode='fan_in', nonlinearity='relu')
        if m.fc_real.bias is not None:
            torch.nn.init.constant_(m.fc_real.bias, 0.0)
        if m.fc_imag.bias is not None:
            torch.nn.init.constant_(m.fc_imag.bias, 0.0)

def vae_loss_fn(recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, epoch: int = 0, total_epochs: int = 500) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    计算VAE损失函数（修复复数损失计算）
    
    Args:
        recon_x: 重构输出
        x: 原始输入
        mu: 均值
        logvar: 对数方差
        epoch: 当前epoch
        total_epochs: 总epoch数
        
    Returns:
        total_loss, recon_loss, kl_loss
    """
    # 动机: 我们的目标是为下游任务提供信息最丰富的mu，因此重构损失是黄金标准。
    #      我们将KL损失的权重降得很低，使其主要起正则化作用，防止过拟合，
    #      同时让模型全力优化重构损失。
    
    # 1. 重构损失: 使用被证明效果较好的“幅值+相位”组合
    magnitude_loss = torch.mean(torch.abs(torch.abs(recon_x) - torch.abs(x)))
    phase_diff = torch.angle(recon_x) - torch.angle(x)
    # 将相位差限制在[-π, π]范围内
    phase_diff = torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff))
    phase_loss = torch.mean(torch.abs(phase_diff))
    recon_loss = magnitude_loss + 0.2 * phase_loss
    
    # 2. KL散度损失: 使用sqrt组合
    kl_loss_real = -0.5 * torch.mean(1 + logvar.real - mu.real.pow(2) - logvar.real.exp())
    kl_loss_imag = -0.5 * torch.mean(1 + logvar.imag - mu.imag.pow(2) - logvar.imag.exp())
    kl_loss = torch.sqrt(kl_loss_real**2 + kl_loss_imag**2)#自创的，目前看效果更好
    # 直接相加，是更标准的做法
    # kl_loss = kl_loss_real + kl_loss_imag
    
    # 3. KL退火与总损失: 延长退火期并使用较低的KL权重
    kl_weight = min(1.0, epoch / (total_epochs * 0.3))
    total_loss = recon_loss + kl_weight * 0.25 * kl_loss
    
    return total_loss, recon_loss, kl_loss

def train_vae_epoch(vae: VAE, loader, optimizer: optim.Optimizer, current_epoch: int = 0, total_epochs: int = 500) -> Tuple[float, float, float, int]:
    """
    训练VAE一个epoch
    
    Args:
        vae: VAE模型
        loader: 数据加载器
        optimizer: 优化器
        
    Returns:
        avg_loss, avg_recon_loss, avg_kl_loss, valid_batches
    """
    vae.train()
    
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    valid_batches = 0
    error_count = 0
    
    for batch_idx, (x, _) in enumerate(loader):
        try:
            # 数据类型转换和设备迁移
            if isinstance(x, np.ndarray):
                x_complex = torch.from_numpy(x).contiguous().to(dtype=torch.complex64, device=device)
            else:
                x_complex = x.to(dtype=torch.complex64, device=device)
            
            # 前向传播
            recon_x, mu, logvar, _ = vae(x_complex)
            loss, recon_loss, kl_loss = vae_loss_fn(recon_x, x_complex, mu, logvar, current_epoch, total_epochs)
            
            # 检查损失值是否有效
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"Batch {batch_idx}: 损失值为 NaN 或 Inf，跳过")
                error_count += 1
                continue
                
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=0.5)
            optimizer.step()
            
            # 累计损失
            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            total_recon_loss += recon_loss.item() * batch_size
            total_kl_loss += kl_loss.item() * batch_size
            valid_batches += batch_size
            
        except Exception as e:
            logger.error(f"Batch {batch_idx} 训练出错: {str(e)}")
            error_count += 1
            continue
    
    # 计算平均损失
    if valid_batches > 0:
        avg_loss = total_loss / valid_batches
        avg_recon_loss = total_recon_loss / valid_batches
        avg_kl_loss = total_kl_loss / valid_batches
    else:
        avg_loss = avg_recon_loss = avg_kl_loss = 0.0
    
    if error_count > 0:
        logger.warning(f"本epoch有 {error_count} 个batch出错")
    
    return avg_loss, avg_recon_loss, avg_kl_loss, valid_batches

def test_vae_model(vae: VAE, test_loader) -> Dict[str, Any]:
    """
    测试VAE模型性能
    
    Args:
        vae: 训练好的VAE模型
        test_loader: 测试数据加载器
        
    Returns:
        包含测试结果的字典
    """
    logger.info("="*50)
    logger.info("VAE模型测试")
    logger.info("="*50)
    
    vae.eval()
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    valid_batches = 0
    error_count = 0
    
    # 收集重构样本用于可视化
    original_samples: List[np.ndarray] = []
    reconstructed_samples: List[np.ndarray] = []
    
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(test_loader):
            try:
                # 数据类型转换
                if isinstance(x, np.ndarray):
                    x_complex = torch.from_numpy(x).contiguous().to(dtype=torch.complex64, device=device)
                else:
                    x_complex = x.to(dtype=torch.complex64, device=device)
                
                # 前向传播
                recon_x, mu, logvar, _ = vae(x_complex)
                _, recon_loss, kl_loss = vae_loss_fn(recon_x, x_complex, mu, logvar, 0, 1)
                
                batch_size = x.size(0)
                total_recon_loss += recon_loss.item() * batch_size
                total_kl_loss += kl_loss.item() * batch_size
                valid_batches += batch_size
                
                # 收集样本用于可视化（只取第一个batch的第一个样本）
                if batch_idx == 0 and len(original_samples) == 0:
                    original_samples.append(x_complex[0].cpu().numpy())
                    reconstructed_samples.append(recon_x[0].cpu().numpy())
                
            except Exception as e:
                logger.error(f"Batch {batch_idx} 测试出错: {str(e)}")
                error_count += 1
                continue
    
    # 计算平均损失
    avg_recon_loss = total_recon_loss / valid_batches if valid_batches > 0 else 0.0
    avg_kl_loss = total_kl_loss / valid_batches if valid_batches > 0 else 0.0
    
    # 打印测试结果
    logger.info(f"测试集重构损失: {avg_recon_loss:.6f}")
    logger.info(f"测试集KL损失: {avg_kl_loss:.6f}")#应该逐渐增加并稳定在0.01-0.1之间
    logger.info(f"测试样本数: {valid_batches}")
    if error_count > 0:
        logger.warning(f"测试时有 {error_count} 个batch出错")
    
    return {
        'recon_loss': avg_recon_loss,
        'kl_loss': avg_kl_loss,
        'samples': valid_batches,
        'original': original_samples,
        'reconstructed': reconstructed_samples
    }

def visualize_vae_results(vae_history: Dict[str, List], test_results: Dict[str, Any]) -> None:
    """
    可视化VAE训练结果
    
    Args:
        vae_history: VAE训练历史
        test_results: VAE测试结果
    """
    logger.info("生成可视化图表...")
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建2x3的子图布局
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('VAE 训练与评估结果', fontsize=16)

    # 展平axes数组以便索引
    ax = axes.ravel()

    # 1. 训练损失曲线
    epochs = vae_history['epoch']
    ax[0].plot(epochs, vae_history['total_loss'], 'b-', linewidth=2, label='总损失')
    ax[0].plot(epochs, vae_history['recon_loss'], 'r-', linewidth=2, label='重构损失')
    ax[0].plot(epochs, vae_history['kl_loss'], 'g-', linewidth=2, label='KL损失')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].set_title('VAE训练损失')
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)
    
    # 2. 重构损失变化
    ax[1].plot(epochs, vae_history['recon_loss'], 'r-', linewidth=2)
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Reconstruction Loss')
    ax[1].set_title('重构损失变化')
    ax[1].grid(True, alpha=0.3)
    
    # 3. KL损失变化
    ax[2].plot(epochs, vae_history['kl_loss'], 'g-', linewidth=2)
    ax[2].set_xlabel('Epoch')
    ax[2].set_ylabel('KL Loss')
    ax[2].set_title('KL散度变化')
    ax[2].grid(True, alpha=0.3)
    
    # 4. 测试结果总结
    metrics_text = f"""
VAE测试结果:
重构损失: {test_results['recon_loss']:.4f}
KL损失: {test_results['kl_loss']:.4f}
测试样本: {test_results['samples']}
    """
    ax[3].text(0.1, 0.5, metrics_text, transform=ax[3].transAxes, fontsize=12,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    ax[3].set_title('测试指标')
    ax[3].axis('off')

    # 5. & 6. 重构效果对比
    if test_results['original'] and test_results['reconstructed']:
        original = test_results['original'][0]
        reconstructed = test_results['reconstructed'][0]
        
        # 绘制原始数据
        ax[4].plot(original.real, label='实部', color='C0')
        ax[4].plot(original.imag, label='虚部', color='C1')
        ax[4].set_title('原始样本')
        ax[4].legend()
        ax[4].grid(True, alpha=0.3)
        
        # 绘制重构数据
        ax[5].plot(reconstructed.real, label='实部', color='C0')
        ax[5].plot(reconstructed.imag, label='虚部', color='C1')
        ax[5].set_title('重构样本')
        ax[5].legend()
        ax[5].grid(True, alpha=0.3)
    else:
        ax[4].text(0.5, 0.5, '无可用样本', ha='center', va='center')
        ax[4].set_title('原始样本')
        ax[4].axis('off')
        ax[5].text(0.5, 0.5, '无可用样本', ha='center', va='center')
        ax[5].set_title('重构样本')
        ax[5].axis('off')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 定义图片保存路径
    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_path = os.path.join(RESULTS_DIR, 'vae_training_results.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    
    plt.show()
    plt.close()  # 释放内存
    
    logger.info(f"可视化完成！图片已保存为: {save_path}")

def save_vae_model(vae: VAE, scaler_X_real, scaler_X_imag, vae_history: Dict[str, List], test_results: Dict[str, Any]) -> None:
    """
    保存VAE模型和测试结果
    
    Args:
        vae: VAE模型
        scaler_X_real: 实部标准化器
        scaler_X_imag: 虚部标准化器
        vae_history: 训练历史
        test_results: 测试结果
    """
    logger.info("保存VAE模型和结果...")
    
    try:
        # 定义文件保存路径
        os.makedirs(MODELS_DIR, exist_ok=True)
        os.makedirs(RESULTS_DIR, exist_ok=True)
        model_path = os.path.join(MODELS_DIR, 'vae_model.pth')
        scaler_x_real_path = os.path.join(MODELS_DIR, 'scaler_X_real.save')
        scaler_x_imag_path = os.path.join(MODELS_DIR, 'scaler_X_imag.save')
        history_path = os.path.join(RESULTS_DIR, 'vae_training_history.json')
        test_results_path = os.path.join(RESULTS_DIR, 'vae_test_results.json')
        
        # 保存模型
        torch.save({
            'vae_state_dict': vae.state_dict(),
            'model_config': {
                'x_dim': vae.x_dim,
                'z_dim': vae.z_dim
            }
        }, model_path)
        
        # 保存标准化器
        joblib.dump(scaler_X_real, scaler_x_real_path)
        joblib.dump(scaler_X_imag, scaler_x_imag_path)
        
        # 保存训练历史
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(vae_history, f, indent=2, ensure_ascii=False)
        
        # 保存测试结果
        test_results_save = {
            'recon_loss': test_results['recon_loss'],
            'kl_loss': test_results['kl_loss'],
            'samples': test_results['samples']
        }
        with open(test_results_path, 'w', encoding='utf-8') as f:
            json.dump(test_results_save, f, indent=2, ensure_ascii=False)
        
        logger.info("VAE模型和结果已保存！")
        
    except Exception as e:
        logger.error(f"保存模型时出错: {str(e)}")
        raise

def main() -> None:
    """主训练函数"""
    logger.info("开始训练VAE...")
    
    try:
        # 加载数据
        train_loader, test_loader, scaler_X_real, scaler_X_imag, _ = get_dataloader(
            DATA_DIR, batch_size=BATCH_SIZE
        )
        
        # 获取输入维度
        sample_X, _ = next(iter(train_loader))
        x_dim = sample_X.shape[1]
        logger.info(f"输入维度: {x_dim}")
        
        # 根据输入维度动态调整潜在空间维度
        z_dim = min(Z_DIM, max(50, x_dim // 8))  # 压缩比约为6:1
        logger.info(f"潜在空间维度: {z_dim} (压缩比: {x_dim/z_dim:.1f}:1)")
        
        # 初始化VAE
        vae = VAE(x_dim, z_dim)
        vae = vae.to(device)
        vae.apply(init_weights)
        
        # 训练VAE
        optimizer = optim.AdamW(vae.parameters(), lr=5e-5)
        # 动机: 引入学习率调度器。当模型训练进入瓶颈期（损失不再下降）时，
        #      自动降低学习率可以帮助模型进行更精细的搜索，有助于达到更低的重构损失。
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20, factor=0.5, verbose=True)
        vae_history = {'epoch': [], 'total_loss': [], 'recon_loss': [], 'kl_loss': []}
        
        logger.info(f"开始训练，总epoch数: {VAE_EPOCHS}")
        for epoch in range(VAE_EPOCHS):
            avg_loss, avg_recon_loss, avg_kl_loss, valid_batches = train_vae_epoch(
                vae, train_loader, optimizer, epoch, VAE_EPOCHS
            )
            
            vae_history['epoch'].append(epoch + 1)
            vae_history['total_loss'].append(avg_loss)
            vae_history['recon_loss'].append(avg_recon_loss)
            vae_history['kl_loss'].append(avg_kl_loss)
            
            if epoch % 50 == 0:
                logger.info(f"VAE Epoch {epoch+1}, Loss: {avg_loss:.6f}")
                logger.info(f"  Recon: {avg_recon_loss:.6f}, KL: {avg_kl_loss:.6f}")
                logger.info(f"  有效batch数: {valid_batches}")
            # 使用总损失更新学习率调度器
            scheduler.step(avg_loss)
            
            # 早停机制：如果KL损失过小，提前停止
            if epoch > 100 and avg_kl_loss < 0.01:
                logger.warning(f"KL损失过小 ({avg_kl_loss:.6f})，在第 {epoch+1} epoch 提前停止训练")
                break
        
        logger.info("VAE训练完成！")
        
        # 测试VAE模型
        test_results = test_vae_model(vae, test_loader)
        
        # 可视化结果
        visualize_vae_results(vae_history, test_results)
        
        # 保存模型和结果
        save_vae_model(vae, scaler_X_real, scaler_X_imag, vae_history, test_results)
        
        logger.info("VAE训练和测试流程完成！")
        
    except Exception as e:
        logger.error(f"训练过程中出现错误: {str(e)}")
        raise

if __name__ == '__main__':
    main()