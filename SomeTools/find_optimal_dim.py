# 自动寻找最佳潜在空间维度
import torch
import torch.nn as nn
import torch.optim as optim
from dataload import get_dataloader
from model import VAE
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Dict, List, Tuple
import json
import time

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# GPU设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def vae_loss_fn(recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """简化的VAE损失函数"""
    # 重构损失
    magnitude_loss = torch.mean(torch.abs(torch.abs(recon_x) - torch.abs(x)))
    phase_diff = torch.angle(recon_x) - torch.angle(x)
    phase_diff = torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff))
    phase_loss = torch.mean(torch.abs(phase_diff))
    recon_loss = magnitude_loss + 0.1 * phase_loss
    
    # KL损失
    kl_loss_real = -0.5 * torch.mean(1 + logvar.real - mu.real.pow(2) - logvar.real.exp())
    kl_loss_imag = -0.5 * torch.mean(1 + logvar.imag - mu.imag.pow(2) - logvar.imag.exp())
    kl_loss = torch.sqrt(kl_loss_real**2 + kl_loss_imag**2)
    
    total_loss = recon_loss + 0.5 * kl_loss
    return total_loss, recon_loss, kl_loss

def train_vae_quick(vae: VAE, train_loader, test_loader, epochs: int = 50) -> Dict[str, float]:
    """快速训练VAE并返回评估指标"""
    optimizer = optim.Adam(vae.parameters(), lr=1e-4)
    vae.train()
    
    # 训练
    for epoch in range(epochs):
        for batch_idx, (x, _) in enumerate(train_loader):
            if isinstance(x, np.ndarray):
                x_complex = torch.from_numpy(x).contiguous().to(dtype=torch.complex64, device=device)
            else:
                x_complex = x.to(dtype=torch.complex64, device=device)
            
            recon_x, mu, logvar, _ = vae(x_complex)
            loss, recon_loss, kl_loss = vae_loss_fn(recon_x, x_complex, mu, logvar)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # 评估
    vae.eval()
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for x, _ in test_loader:
            if isinstance(x, np.ndarray):
                x_complex = torch.from_numpy(x).contiguous().to(dtype=torch.complex64, device=device)
            else:
                x_complex = x.to(dtype=torch.complex64, device=device)
            
            recon_x, mu, logvar, _ = vae(x_complex)
            _, recon_loss, kl_loss = vae_loss_fn(recon_x, x_complex, mu, logvar)
            
            batch_size = x.size(0)
            total_recon_loss += recon_loss.item() * batch_size
            total_kl_loss += kl_loss.item() * batch_size
            total_samples += batch_size
    
    avg_recon_loss = total_recon_loss / total_samples
    avg_kl_loss = total_kl_loss / total_samples
    
    return {
        'recon_loss': avg_recon_loss,
        'kl_loss': avg_kl_loss,
        'total_loss': avg_recon_loss + 0.5 * avg_kl_loss
    }

def calculate_compression_ratio(x_dim: int, z_dim: int) -> float:
    """计算压缩比"""
    return x_dim / z_dim

def calculate_kl_efficiency(kl_loss: float, z_dim: int) -> float:
    """计算KL效率（每维度的KL损失）"""
    return kl_loss / z_dim

def find_optimal_dimension(data_dir: str = './datas', batch_size: int = 16) -> Dict:
    """寻找最佳潜在空间维度"""
    logger.info("开始寻找最佳潜在空间维度...")
    
    # 加载数据
    train_loader, test_loader, _, _, _ = get_dataloader(data_dir, batch_size=batch_size)
    sample_X, _ = next(iter(train_loader))
    x_dim = sample_X.shape[1]
    logger.info(f"输入维度: {x_dim}")
    
    # 定义要测试的维度范围
    compression_ratios = [2, 4, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50]
    z_dims = [max(10, x_dim // ratio) for ratio in compression_ratios]
    z_dims = list(set(z_dims))  # 去重
    z_dims.sort()
    
    logger.info(f"测试维度: {z_dims}")
    
    results = []
    
    for z_dim in z_dims:
        logger.info(f"测试潜在空间维度: {z_dim}")
        start_time = time.time()
        
        try:
            # 创建模型
            vae = VAE(x_dim, z_dim)
            vae = vae.to(device)
            
            # 快速训练
            metrics = train_vae_quick(vae, train_loader, test_loader, epochs=30)
            
            # 计算额外指标
            compression_ratio = calculate_compression_ratio(x_dim, z_dim)
            kl_efficiency = calculate_kl_efficiency(metrics['kl_loss'], z_dim)
            
            result = {
                'z_dim': z_dim,
                'compression_ratio': compression_ratio,
                'recon_loss': metrics['recon_loss'],
                'kl_loss': metrics['kl_loss'],
                'total_loss': metrics['total_loss'],
                'kl_efficiency': kl_efficiency,
                'training_time': time.time() - start_time
            }
            
            results.append(result)
            logger.info(f"  重构损失: {metrics['recon_loss']:.4f}, KL损失: {metrics['kl_loss']:.4f}")
            
        except Exception as e:
            logger.error(f"维度 {z_dim} 测试失败: {str(e)}")
            continue
    
    # 分析结果
    if not results:
        logger.error("没有成功的结果")
        return {}
    
    # 找到最佳维度（基于综合评分）
    for result in results:
        # 综合评分：重构损失越低越好，KL损失适中，压缩比合理
        recon_score = 1.0 / (1.0 + result['recon_loss'])  # 重构损失越低越好
        kl_score = min(1.0, result['kl_loss'] / 0.1)  # KL损失适中最好
        compression_score = 1.0 / (1.0 + abs(result['compression_ratio'] - 10))  # 压缩比接近10最好
        
        result['composite_score'] = 0.5 * recon_score + 0.3 * kl_score + 0.2 * compression_score
    
    # 按综合评分排序
    results.sort(key=lambda x: x['composite_score'], reverse=True)
    
    best_result = results[0]
    logger.info(f"最佳潜在空间维度: {best_result['z_dim']}")
    logger.info(f"  压缩比: {best_result['compression_ratio']:.1f}:1")
    logger.info(f"  重构损失: {best_result['recon_loss']:.4f}")
    logger.info(f"  KL损失: {best_result['kl_loss']:.4f}")
    logger.info(f"  综合评分: {best_result['composite_score']:.4f}")
    
    # 可视化结果
    visualize_dimension_search(results, x_dim)
    
    # 保存结果
    save_dimension_results(results, best_result)
    
    return best_result

def visualize_dimension_search(results: List[Dict], x_dim: int):
    """可视化维度搜索结果"""
    z_dims = [r['z_dim'] for r in results]
    recon_losses = [r['recon_loss'] for r in results]
    kl_losses = [r['kl_loss'] for r in results]
    compression_ratios = [r['compression_ratio'] for r in results]
    composite_scores = [r['composite_score'] for r in results]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # 重构损失 vs 维度
    ax1.plot(z_dims, recon_losses, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('潜在空间维度')
    ax1.set_ylabel('重构损失')
    ax1.set_title('重构损失 vs 潜在空间维度')
    ax1.grid(True, alpha=0.3)
    
    # KL损失 vs 维度
    ax2.plot(z_dims, kl_losses, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('潜在空间维度')
    ax2.set_ylabel('KL损失')
    ax2.set_title('KL损失 vs 潜在空间维度')
    ax2.grid(True, alpha=0.3)
    
    # 压缩比 vs 维度
    ax3.plot(z_dims, compression_ratios, 'go-', linewidth=2, markersize=8)
    ax3.set_xlabel('潜在空间维度')
    ax3.set_ylabel('压缩比')
    ax3.set_title('压缩比 vs 潜在空间维度')
    ax3.grid(True, alpha=0.3)
    
    # 综合评分 vs 维度
    ax4.plot(z_dims, composite_scores, 'mo-', linewidth=2, markersize=8)
    ax4.set_xlabel('潜在空间维度')
    ax4.set_ylabel('综合评分')
    ax4.set_title('综合评分 vs 潜在空间维度')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dimension_search_results.png', dpi=200, bbox_inches='tight')
    plt.show()
    plt.close()

def save_dimension_results(results: List[Dict], best_result: Dict):
    """保存维度搜索结果"""
    output = {
        'input_dimension': results[0]['z_dim'] * results[0]['compression_ratio'],
        'all_results': results,
        'best_result': best_result,
        'recommendation': {
            'optimal_z_dim': best_result['z_dim'],
            'compression_ratio': best_result['compression_ratio'],
            'reasoning': f"选择{best_result['z_dim']}维作为最佳潜在空间维度，压缩比为{best_result['compression_ratio']:.1f}:1，"
                        f"重构损失为{best_result['recon_loss']:.4f}，KL损失为{best_result['kl_loss']:.4f}"
        }
    }
    
    with open('dimension_search_results.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    logger.info("结果已保存到 dimension_search_results.json")

if __name__ == '__main__':
    best_dim = find_optimal_dimension()
    print(f"\n推荐的最佳潜在空间维度: {best_dim.get('z_dim', 'N/A')}") 