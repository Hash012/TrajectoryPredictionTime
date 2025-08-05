# 全面超参数优化器
import torch
import torch.nn as nn
import torch.optim as optim
from dataload import get_dataloader
from model import VAE
import numpy as np
import logging
import json
import time
from typing import Dict, List, Tuple, Any
import itertools
from sklearn.model_selection import ParameterGrid

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class HyperparameterOptimizer:
    def __init__(self, data_dir: str = './datas'):
        self.data_dir = data_dir
        self.best_config = None
        self.all_results = []
        
    def vae_loss_fn(self, recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, 
                   logvar: torch.Tensor, kl_weight: float = 0.5, phase_weight: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """可配置的VAE损失函数"""
        # 重构损失
        magnitude_loss = torch.mean(torch.abs(torch.abs(recon_x) - torch.abs(x)))
        phase_diff = torch.angle(recon_x) - torch.angle(x)
        phase_diff = torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff))
        phase_loss = torch.mean(torch.abs(phase_diff))
        recon_loss = magnitude_loss + phase_weight * phase_loss
        
        # KL损失
        kl_loss_real = -0.5 * torch.mean(1 + logvar.real - mu.real.pow(2) - logvar.real.exp())
        kl_loss_imag = -0.5 * torch.mean(1 + logvar.imag - mu.imag.pow(2) - logvar.imag.exp())
        kl_loss = torch.sqrt(kl_loss_real**2 + kl_loss_imag**2)
        
        total_loss = recon_loss + kl_weight * kl_loss
        return total_loss, recon_loss, kl_loss
    
    def train_vae_with_config(self, config: Dict[str, Any], train_loader, test_loader, 
                            quick_mode: bool = True) -> Dict[str, float]:
        """使用指定配置训练VAE"""
        # 获取输入维度
        sample_X, _ = next(iter(train_loader))
        x_dim = sample_X.shape[1]
        z_dim = config['z_dim']
        
        # 创建模型
        vae = VAE(x_dim, z_dim)
        vae = vae.to(device)
        
        # 优化器
        if config['optimizer'] == 'adam':
            optimizer = optim.Adam(vae.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        elif config['optimizer'] == 'adamw':
            optimizer = optim.AdamW(vae.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        else:
            optimizer = optim.SGD(vae.parameters(), lr=config['lr'], momentum=0.9, weight_decay=config['weight_decay'])
        
        # 学习率调度器
        if config['scheduler'] == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8)
        elif config['scheduler'] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        else:
            scheduler = None
        
        # 训练
        epochs = 20 if quick_mode else 100
        vae.train()
        
        for epoch in range(epochs):
            for x, _ in train_loader:
                if isinstance(x, np.ndarray):
                    x_complex = torch.from_numpy(x).to(dtype=torch.complex64, device=device)
                else:
                    x_complex = x.to(dtype=torch.complex64, device=device)
                
                recon_x, mu, logvar, _ = vae(x_complex)
                loss, _, _ = self.vae_loss_fn(recon_x, x_complex, mu, logvar, 
                                            config['kl_weight'], config['phase_weight'])
                
                optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪
                if config['grad_clip'] > 0:
                    torch.nn.utils.clip_grad_norm_(vae.parameters(), config['grad_clip'])
                
                optimizer.step()
            
            if scheduler:
                scheduler.step()
        
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
                _, recon_loss, kl_loss = self.vae_loss_fn(recon_x, x_complex, mu, logvar,
                                                        config['kl_weight'], config['phase_weight'])
                
                batch_size = x.size(0)
                total_recon_loss += recon_loss.item() * batch_size
                total_kl_loss += kl_loss.item() * batch_size
                total_samples += batch_size
        
        return {
            'recon_loss': total_recon_loss / total_samples,
            'kl_loss': total_kl_loss / total_samples,
            'total_loss': (total_recon_loss + config['kl_weight'] * total_kl_loss) / total_samples
        }
    
    def calculate_score(self, metrics: Dict[str, float], config: Dict[str, Any]) -> float:
        """计算综合评分"""
        recon_score = 1.0 / (1.0 + metrics['recon_loss'])
        kl_score = min(1.0, metrics['kl_loss'] / 0.1)
        
        # 压缩比评分
        compression_ratio = 1198 / config['z_dim']  # 假设输入维度为1198
        compression_score = 1.0 / (1.0 + abs(compression_ratio - 8))
        
        # 训练效率评分（学习率、批次大小等）
        efficiency_score = 1.0 / (1.0 + abs(config['lr'] - 1e-4) / 1e-4)
        
        # 综合评分
        total_score = (0.4 * recon_score + 0.3 * kl_score + 
                      0.2 * compression_score + 0.1 * efficiency_score)
        
        return total_score
    
    def optimize_hyperparameters(self, quick_mode: bool = True) -> Dict[str, Any]:
        """优化超参数"""
        logger.info("开始超参数优化...")
        
        # 加载数据
        train_loader, test_loader, _, _, _ = get_dataloader(self.data_dir, batch_size=16)
        
        # 定义超参数搜索空间
        param_grid = {
            'z_dim': [100, 150, 200, 250],  # 潜在空间维度
            'lr': [1e-5, 5e-5, 1e-4, 5e-4],  # 学习率
            'batch_size': [8, 16, 32],  # 批次大小
            'optimizer': ['adam', 'adamw'],  # 优化器
            'weight_decay': [1e-6, 1e-5, 1e-4],  # 权重衰减
            'kl_weight': [0.3, 0.5, 0.7, 1.0],  # KL损失权重
            'phase_weight': [0.05, 0.1, 0.2],  # 相位损失权重
            'scheduler': ['none', 'step', 'cosine'],  # 学习率调度器
            'grad_clip': [0, 1.0, 5.0]  # 梯度裁剪
        }
        
        # 生成参数组合
        param_combinations = list(ParameterGrid(param_grid))
        logger.info(f"总共需要测试 {len(param_combinations)} 种参数组合")
        
        # 限制测试数量（快速模式）
        if quick_mode and len(param_combinations) > 50:
            # 随机选择50个组合
            np.random.shuffle(param_combinations)
            param_combinations = param_combinations[:50]
            logger.info(f"快速模式：随机选择 {len(param_combinations)} 种组合进行测试")
        
        best_score = -1
        best_config = None
        
        for i, config in enumerate(param_combinations):
            logger.info(f"测试配置 {i+1}/{len(param_combinations)}: {config}")
            
            try:
                # 根据批次大小重新加载数据
                if config['batch_size'] != 16:
                    train_loader, test_loader, _, _, _ = get_dataloader(
                        self.data_dir, batch_size=config['batch_size']
                    )
                
                # 训练和评估
                metrics = self.train_vae_with_config(config, train_loader, test_loader, quick_mode)
                score = self.calculate_score(metrics, config)
                
                result = {
                    'config': config,
                    'metrics': metrics,
                    'score': score
                }
                self.all_results.append(result)
                
                logger.info(f"  重构损失: {metrics['recon_loss']:.4f}")
                logger.info(f"  KL损失: {metrics['kl_loss']:.4f}")
                logger.info(f"  综合评分: {score:.4f}")
                
                if score > best_score:
                    best_score = score
                    best_config = result
                    logger.info(f"  *** 新的最佳配置！ ***")
                
            except Exception as e:
                logger.error(f"配置 {config} 测试失败: {str(e)}")
                continue
        
        self.best_config = best_config
        return best_config
    
    def analyze_results(self) -> Dict[str, Any]:
        """分析优化结果"""
        if not self.all_results:
            return {}
        
        # 按评分排序
        self.all_results.sort(key=lambda x: x['score'], reverse=True)
        
        # 分析最佳配置
        best = self.all_results[0]
        
        # 分析各参数的影响
        param_analysis = {}
        for param in ['z_dim', 'lr', 'kl_weight', 'phase_weight']:
            values = [r['config'][param] for r in self.all_results]
            scores = [r['score'] for r in self.all_results]
            
            # 计算参数值与评分的相关性
            unique_values = sorted(set(values))
            avg_scores = []
            for val in unique_values:
                matching_scores = [s for v, s in zip(values, scores) if v == val]
                avg_scores.append(np.mean(matching_scores))
            
            param_analysis[param] = {
                'values': unique_values,
                'avg_scores': avg_scores,
                'best_value': unique_values[np.argmax(avg_scores)]
            }
        
        analysis = {
            'best_config': best,
            'top_5_configs': self.all_results[:5],
            'param_analysis': param_analysis,
            'total_tests': len(self.all_results)
        }
        
        return analysis
    
    def save_results(self, filename: str = 'hyperparameter_optimization_results.json'):
        """保存优化结果"""
        analysis = self.analyze_results()
        
        # 准备保存的数据
        save_data = {
            'best_config': analysis['best_config'],
            'top_5_configs': analysis['top_5_configs'],
            'param_analysis': analysis['param_analysis'],
            'total_tests': analysis['total_tests'],
            'all_results': self.all_results
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"结果已保存到 {filename}")
        
        # 打印最佳配置
        best = analysis['best_config']
        logger.info("\n" + "="*50)
        logger.info("🎯 最佳超参数配置:")
        logger.info("="*50)
        for param, value in best['config'].items():
            logger.info(f"{param:15}: {value}")
        logger.info(f"{'重构损失':15}: {best['metrics']['recon_loss']:.4f}")
        logger.info(f"{'KL损失':15}: {best['metrics']['kl_loss']:.4f}")
        logger.info(f"{'综合评分':15}: {best['score']:.4f}")
        logger.info("="*50)

def main():
    """主函数"""
    optimizer = HyperparameterOptimizer()
    
    # 快速优化模式
    logger.info("开始快速超参数优化...")
    best_config = optimizer.optimize_hyperparameters(quick_mode=True)
    
    if best_config:
        optimizer.save_results()
        
        # 使用最佳配置进行完整训练
        logger.info("\n使用最佳配置进行完整训练...")
        full_config = optimizer.optimize_hyperparameters(quick_mode=False)
        optimizer.save_results('full_hyperparameter_optimization_results.json')
    else:
        logger.error("超参数优化失败")

if __name__ == '__main__':
    main() 