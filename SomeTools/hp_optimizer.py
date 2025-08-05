# 超参数优化器
import torch
import torch.optim as optim
from dataload import get_dataloader
from model import VAE
import numpy as np
import logging
import json
from itertools import product

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def vae_loss_fn(recon_x, x, mu, logvar, kl_weight=0.5, phase_weight=0.1):
    """可配置的VAE损失函数"""
    magnitude_loss = torch.mean(torch.abs(torch.abs(recon_x) - torch.abs(x)))
    phase_diff = torch.angle(recon_x) - torch.angle(x)
    phase_diff = torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff))
    phase_loss = torch.mean(torch.abs(phase_diff))
    recon_loss = magnitude_loss + phase_weight * phase_loss
    
    kl_loss_real = -0.5 * torch.mean(1 + logvar.real - mu.real.pow(2) - logvar.real.exp())
    kl_loss_imag = -0.5 * torch.mean(1 + logvar.imag - mu.imag.pow(2) - logvar.imag.exp())
    kl_loss = torch.sqrt(kl_loss_real**2 + kl_loss_imag**2)
    
    return recon_loss + kl_weight * kl_loss, recon_loss, kl_loss

def train_and_evaluate(config, train_loader, test_loader):
    """训练并评估指定配置"""
    sample_X, _ = next(iter(train_loader))
    x_dim = sample_X.shape[1]
    
    vae = VAE(x_dim, config['z_dim']).to(device)
    
    if config['optimizer'] == 'adam':
        optimizer = optim.Adam(vae.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    else:
        optimizer = optim.AdamW(vae.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    
    # 快速训练
    vae.train()
    for epoch in range(20):
        for x, _ in train_loader:
            if isinstance(x, np.ndarray):
                x_complex = torch.from_numpy(x).to(dtype=torch.complex64, device=device)
            else:
                x_complex = x.to(dtype=torch.complex64, device=device)
            
            recon_x, mu, logvar, _ = vae(x_complex)
            loss, _, _ = vae_loss_fn(recon_x, x_complex, mu, logvar, config['kl_weight'], config['phase_weight'])
            
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
                x_complex = torch.from_numpy(x).to(dtype=torch.complex64, device=device)
            else:
                x_complex = x.to(dtype=torch.complex64, device=device)
            
            recon_x, mu, logvar, _ = vae(x_complex)
            _, recon_loss, kl_loss = vae_loss_fn(recon_x, x_complex, mu, logvar, config['kl_weight'], config['phase_weight'])
            
            batch_size = x.size(0)
            total_recon_loss += recon_loss.item() * batch_size
            total_kl_loss += kl_loss.item() * batch_size
            total_samples += batch_size
    
    return {
        'recon_loss': total_recon_loss / total_samples,
        'kl_loss': total_kl_loss / total_samples
    }

def optimize_hyperparameters():
    """优化超参数"""
    logger.info("开始超参数优化...")
    
    # 加载数据
    train_loader, test_loader, _, _, _ = get_dataloader('./datas', batch_size=16)
    
    # 定义搜索空间
    search_space = {
        'z_dim': [100, 150, 200],
        'lr': [1e-5, 5e-5, 1e-4],
        'optimizer': ['adam', 'adamw'],
        'weight_decay': [1e-6, 1e-5],
        'kl_weight': [0.3, 0.5, 0.7],
        'phase_weight': [0.05, 0.1, 0.2]
    }
    
    # 生成所有组合
    keys = search_space.keys()
    values = search_space.values()
    combinations = list(product(*values))
    
    logger.info(f"测试 {len(combinations)} 种参数组合")
    
    results = []
    best_score = -1
    best_config = None
    
    for i, combo in enumerate(combinations):
        config = dict(zip(keys, combo))
        logger.info(f"测试 {i+1}/{len(combinations)}: {config}")
        
        try:
            metrics = train_and_evaluate(config, train_loader, test_loader)
            
            # 计算评分
            recon_score = 1.0 / (1.0 + metrics['recon_loss'])
            kl_score = min(1.0, metrics['kl_loss'] / 0.1)
            compression_score = 1.0 / (1.0 + abs(1198/config['z_dim'] - 8))
            score = 0.4 * recon_score + 0.4 * kl_score + 0.2 * compression_score
            
            result = {
                'config': config,
                'metrics': metrics,
                'score': score
            }
            results.append(result)
            
            logger.info(f"  重构损失: {metrics['recon_loss']:.4f}, KL损失: {metrics['kl_loss']:.4f}, 评分: {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_config = result
                logger.info(f"  *** 新的最佳配置！ ***")
                
        except Exception as e:
            logger.error(f"配置失败: {str(e)}")
            continue
    
    # 保存结果
    results.sort(key=lambda x: x['score'], reverse=True)
    
    with open('hyperparameter_results.json', 'w', encoding='utf-8') as f:
        json.dump({
            'best_config': best_config,
            'top_5': results[:5],
            'all_results': results
        }, f, indent=2, ensure_ascii=False)
    
    # 打印最佳配置
    if best_config:
        logger.info("\n" + "="*50)
        logger.info("🎯 最佳超参数配置:")
        logger.info("="*50)
        for param, value in best_config['config'].items():
            logger.info(f"{param:15}: {value}")
        logger.info(f"{'重构损失':15}: {best_config['metrics']['recon_loss']:.4f}")
        logger.info(f"{'KL损失':15}: {best_config['metrics']['kl_loss']:.4f}")
        logger.info(f"{'综合评分':15}: {best_config['score']:.4f}")
        logger.info("="*50)
    
    return best_config

if __name__ == '__main__':
    best = optimize_hyperparameters()
    print(f"\n推荐的最佳配置: {best['config'] if best else 'None'}") 