"""
使用随机森林快速评估 VAE 潜向量 (μ) 的信息量。
-----------------------------------------------------
运行方式：
    python evaluate_latent_rf.py
输出：
    • 训练集 / 测试集 R²、MAE、MAPE
    • 简单判断信息瓶颈在 VAE 还是回归器
"""
import os
import joblib
import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from dataload import get_dataloader
from model import VAE

# -------------------------- 路径配置 --------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, 'datas')
MODEL_DIR = os.path.join(CURRENT_DIR, 'trained_models')

# ------------------------- 加载模型 --------------------------
vae_ckpt = torch.load(os.path.join(MODEL_DIR, 'vae_model.pth'), map_location='cpu')
config = vae_ckpt['model_config']
vae = VAE(config['x_dim'], config['z_dim'])
vae.load_state_dict(vae_ckpt['vae_state_dict'])
vae.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vae.to(device)

# ------------------------ 加载数据 ---------------------------
train_loader, test_loader, _, _, scaler_y = get_dataloader(DATA_DIR, batch_size=256, shuffle=False)

def extract_mu(loader):
    mus, ts = [], []
    with torch.no_grad():
        for x, t in loader:
            if isinstance(x, np.ndarray):
                x_complex = torch.from_numpy(x).to(dtype=torch.complex64, device=device)
            else:
                x_complex = x.to(dtype=torch.complex64, device=device)
            _, mu, _, _ = vae(x_complex)
            mus.append(mu.cpu().numpy())
            ts.append(t.numpy())
    return np.vstack(mus), np.vstack(ts)

mu_train_c, t_train_norm = extract_mu(train_loader)
mu_test_c, t_test_norm = extract_mu(test_loader)

# 将复数潜向量拆分成实部和虚部
mu_train = np.hstack([mu_train_c.real, mu_train_c.imag])
mu_test = np.hstack([mu_test_c.real, mu_test_c.imag])

# 反标准化目标值
T_train = scaler_y.inverse_transform(t_train_norm)
T_test = scaler_y.inverse_transform(t_test_norm)

# ---------------------- 训练随机森林 -------------------------
rf = RandomForestRegressor(n_estimators=300, random_state=0)
rf.fit(mu_train, T_train.ravel())

pred_train = rf.predict(mu_train)
pred_test = rf.predict(mu_test)

# ------------------------- 评估指标 ---------------------------
train_r2 = r2_score(T_train, pred_train)
train_mae = mean_absolute_error(T_train, pred_train)

test_r2 = r2_score(T_test, pred_test)
test_mae = mean_absolute_error(T_test, pred_test)

mape = lambda y, p: np.mean(np.abs((y - p) / y)) * 100
train_mape = mape(T_train, pred_train)
test_mape = mape(T_test, pred_test)

print('=' * 60)
print('随机森林评估 VAE 潜向量信息量')
print('=' * 60)
print(f'TRAIN  R²: {train_r2:.4f} | MAE: {train_mae:.4f} | MAPE: {train_mape:.2f}%')
print(f'TEST   R²: {test_r2:.4f} | MAE: {test_mae:.4f} | MAPE: {test_mape:.2f}%')

# ---------------------- 信息瓶颈判断 --------------------------
if test_r2 >= 0.5:
    print('\n结论：潜向量信息量充足，重点优化回归器网络即可。')
else:
    print('\n结论：潜向量信息量不足，需要进一步改进 VAE（例如分阶段联合训练、调整 γ）。')
