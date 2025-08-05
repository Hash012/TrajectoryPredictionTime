# ==================== 导入必要的库 ====================
import numpy as np                    # 数值计算库
import pandas as pd                   # 数据处理库
import os                            # 操作系统接口
import torch                         # PyTorch深度学习框架
from torch.utils.data import Dataset, DataLoader  # PyTorch数据加载工具
from sklearn.preprocessing import StandardScaler  # 数据标准化工具

class TrajectoryComplexDataset(Dataset):
    def __init__(self, xx_path, yy_path, tt_path, scaler_X_real=None, scaler_X_imag=None, scaler_y=None):
    
        xx = pd.read_excel(xx_path, header=None).values  # shape: (样本数, n)
        yy = pd.read_excel(yy_path, header=None).values  # shape: (样本数, n)
        tt = pd.read_excel(tt_path, header=None).values  # shape: (样本数, 1) 或 (样本数,)

        # 合成为复数
        X_complex = xx + 1j * yy  # shape: (样本数, n)
        self.X = X_complex
        # 先对y做log1p变换
        self.y = np.log1p(tt)
        
        # # ==================== 目标变量 't' 的鲁棒预处理 ====================
        # # 动机: 经过数据分析(data_analyzer.py)，我们发现目标变量t存在极端离群值，
        # #       导致模型训练困难，MSE/MAE指标异常巨大。为了解决这个问题，
        # #       我们采用“先缩尾，再log变换”的策略，从数据层面根除离群值的影响。

        # # 步骤 1: 缩尾处理 (Winsorization)
        # # 目的: 消除极端离群值的影响，同时保留绝大多数数据的分布。
        # # 方法: 计算99百分位数，并将所有超过该值的数据点强制替换为该值。
        # #       这是一种比直接删除离群值更温和、信息损失更少的处理方式。
        # p99 = np.percentile(tt, 99)
        # tt_winsorized = np.clip(tt, a_min=None, a_max=p99)
        
        # # 步骤 2: 对数变换 (Log Transform)
        # # 目的: 对经过缩尾处理后仍然高度右偏的数据进行变换，使其分布更接近正态，
        # #       并压缩数值范围，使模型更容易学习。
        # # 方法: 使用np.log1p，即log(1+x)，它对0值友好，数值上更稳定。
        # self.y = np.log1p(tt_winsorized)
        # # ====================================================================
        
        # 标准化处理
        if scaler_X_real is None:
            # 训练时：创建新的标准化器
            self.scaler_X_real = StandardScaler()
            self.scaler_X_imag = StandardScaler()
            # StandardScaler.fit返回: 训练好的标准化器
            self.scaler_X_real.fit(self.X.real)
            self.scaler_X_imag.fit(self.X.imag)
            
        else:
            # 预测时：使用预训练的标准化器
            self.scaler_X_real = scaler_X_real
            self.scaler_X_imag = scaler_X_imag
            
            
        if scaler_y is None:
            # 训练时：创建新的标准化器
            self.scaler_y = StandardScaler()
            # StandardScaler.fit返回: 训练好的标准化器
            self.scaler_y.fit(self.y)
        else:
            # 预测时：使用预训练的标准化器
            self.scaler_y = scaler_y
        
        # StandardScaler.transform返回: 标准化后的特征数据
        self.X_norm_real = self.scaler_X_real.transform(self.X.real)
        self.X_norm_imag = self.scaler_X_imag.transform(self.X.imag)
        self.X_norm = self.X_norm_real + 1j * self.X_norm_imag
        # StandardScaler.transform返回: 标准化后的标签数据
        self.y_norm = self.scaler_y.transform(self.y)
    
    def __len__(self):
        """
        返回数据集大小
        
        返回:
            int: 数据集中的样本数量
            
        说明:
            返回经过维度调整后的实际样本数量
        """
        return self.X.shape[0]
    
    def get_X_dim(self):
        return self.X.shape[1]
    
    def __getitem__(self, idx):
        """
        获取指定索引的数据样本
        
        参数:
            idx (int): 样本索引
            
        返回:
            tuple: (标准化后的特征向量, 标准化后的时间标签)
                - 特征向量: np.ndarray, shape=(2*n,), dtype=float32
                - 时间标签: np.ndarray, shape=(1,), dtype=float32
                
        数据格式:
            - 特征向量：将复数的实部和虚部拼接成2*n维向量
            - 时间标签：标准化后的时间值
            - 数据类型：float32，符合PyTorch要求
            
        说明:
            返回归一化后的实部和虚部拼接向量，以及归一化后的时间
            这种格式便于VAE模型处理
        """
        # 返回归一化后的复数特征向量，以及归一化后的时间
        return self.X_norm[idx], self.y_norm[idx]

def get_dataloader(data_dir, batch_size=32, shuffle=True, scaler_X_real=None, scaler_X_imag=None, scaler_y=None, train_ratio=0.8, random_state=42):
    """
    创建数据加载器
    
    功能：读取轨迹数据文件，创建PyTorch DataLoader
    
    参数:
        data_dir (str): 数据文件目录路径
        batch_size (int): 批次大小，默认32
        shuffle (bool): 是否打乱数据，默认True
        scaler_X_real (StandardScaler, optional): 预训练的输入特征实部标准化器
        scaler_X_imag (StandardScaler, optional): 预训练的输入特征虚部标准化器
        scaler_y (StandardScaler, optional): 预训练的输出标签标准化器
        train_ratio (float): 训练集比例，默认0.8（80%训练，20%测试）
        random_state (int): 随机种子，确保结果可重现，默认42
        
    返回:
        tuple: (train_loader, test_loader, scaler_X_real, scaler_X_imag, scaler_y)
            - train_loader: 训练集数据加载器
            - test_loader: 测试集数据加载器
            - scaler_X_real: 输入特征实部标准化器
            - scaler_X_imag: 输入特征虚部标准化器
            - scaler_y: 输出标签标准化器
            
    异常:
        FileNotFoundError: 当数据文件不存在时抛出
        
    使用场景:
        - 训练时：创建新的标准化器，返回训练集和测试集数据加载器
        - 预测时：使用预训练的标准化器，确保数据一致性
        
    数据集划分:
        - 训练集：用于模型训练，包含大部分数据
        - 测试集：用于模型评估，包含剩余数据
        - 划分比例：默认80%训练，20%测试
    """
    # 动机: 确保脚本的路径处理逻辑清晰。
    #      `get_dataloader`函数现在只负责接收一个已经处理好的、明确的`data_dir`绝对路径，
    #      然后基于它来拼接文件名。所有路径的计算和定义都由调用它的上层脚本完成。
    # 使用os.path.normpath确保路径格式正确
    xx_path = os.path.normpath(os.path.join(data_dir, 'xx.xlsx'))
    yy_path = os.path.normpath(os.path.join(data_dir, 'yy.xlsx'))
    tt_path = os.path.normpath(os.path.join(data_dir, 'tt.xlsx'))
    
    # 检查文件是否存在
    if not os.path.exists(xx_path):
        raise FileNotFoundError(f"找不到文件: {xx_path}")
    if not os.path.exists(yy_path):
        raise FileNotFoundError(f"找不到文件: {yy_path}")
    if not os.path.exists(tt_path):
        raise FileNotFoundError(f"找不到文件: {tt_path}")
    
    print(f"加载数据文件:")
    print(f"  xx: {xx_path}")
    print(f"  yy: {yy_path}")
    print(f"  tt: {tt_path}")
    
    # 创建完整数据集
    # TrajectoryComplexDataset返回: 自定义数据集实例
    full_dataset = TrajectoryComplexDataset(xx_path, yy_path, tt_path, scaler_X_real, scaler_X_imag, scaler_y)
    
    # 计算训练集和测试集的大小
    total_size = len(full_dataset)
    train_size = int(train_ratio * total_size)
    test_size = total_size - train_size
    
    print(f"数据集划分:")
    print(f"  总样本数: {total_size}")
    print(f"  训练集: {train_size} 样本 ({train_ratio*100:.0f}%)")
    print(f"  测试集: {test_size} 样本 ({(1-train_ratio)*100:.0f}%)")
    
    # 使用torch.utils.data.random_split划分数据集
    from torch.utils.data import random_split
    train_dataset, test_dataset = random_split(
        full_dataset, 
        [train_size, test_size],
        generator=torch.Generator().manual_seed(random_state)
    )
    
    # 创建训练集和测试集数据加载器
    # DataLoader返回: PyTorch数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)  # 测试集不打乱
    
    # 返回：训练集DataLoader, 测试集DataLoader, 输入特征标准化器, 输出标签标准化器
    return train_loader, test_loader, full_dataset.scaler_X_real, full_dataset.scaler_X_imag, full_dataset.scaler_y 