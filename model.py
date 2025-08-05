# ==================== 导入必要的库 ====================
import torch                    # PyTorch深度学习框架
import torch.nn as nn          # 神经网络模块

class ComplexLayerNorm(nn.Module):
    """
    复数层标准化 (Complex Layer Normalization) 模块
    
    功能：对复数张量进行层标准化，分别对实部和虚部进行操作
    继承：torch.nn.Module
    """
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(ComplexLayerNorm, self).__init__()
        self.norm_real = nn.LayerNorm(normalized_shape, eps, elementwise_affine)
        self.norm_imag = nn.LayerNorm(normalized_shape, eps, elementwise_affine)

    def forward(self, x):
        x_real = x.real
        x_imag = x.imag
        x_real_norm = self.norm_real(x_real)
        x_imag_norm = self.norm_imag(x_imag)
        return torch.complex(x_real_norm, x_imag_norm)

class ModReLU(nn.Module):
    """
    复数ReLU激活函数模块
    
    功能：对复数张量的模长应用ReLU激活，保持相位不变
    继承：torch.nn.Module
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 复数张量
            
        返回:
            复数张量，保持相位不变，对幅度应用ReLU
        """
        # 计算模长
        magnitude = torch.abs(x)
        # 应用ReLU到模长
        magnitude_relu = torch.relu(magnitude)
        # 保持相位不变，更新模长
        return magnitude_relu * (x / (magnitude + 1e-8))  # 避免除零
    
class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(ComplexLinear, self).__init__()
        self.fc_real = nn.Linear(in_features, out_features)
        self.fc_imag = nn.Linear(in_features, out_features)
    
    def forward(self, x):
        return self.fc_real(x.real) + 1j * self.fc_imag(x.imag)

class VAE(nn.Module):
    """
    变分自编码器(VAE)模型
    
    功能：将轨迹数据编码到潜在空间，并从潜在空间重构轨迹
    继承：torch.nn.Module
    
    设计思路：
    1. 编码器：将高维轨迹数据压缩到低维潜在空间
    2. 潜在空间：学习数据的分布表示，使用均值和方差参数化
    3. 解码器：从潜在空间重构原始轨迹数据
    4. 重参数化技巧：使模型可微分，支持端到端训练
    
    主要组件：
    - encoder: 编码器网络，将输入映射到潜在空间参数
    - fc_mu: 均值映射层
    - fc_logvar: 对数方差映射层  
    - decoder: 解码器网络，从潜在变量重构数据
    
    应用场景：
    - 轨迹数据降维和特征提取
    - 轨迹数据的生成和重构
    - 为时间预测提供潜在特征表示
    """
    def __init__(self, x_dim, z_dim):
        """
        初始化VAE模型
        
        参数:
            n (int): 轨迹点数（每个轨迹包含n个散点）
            z_dim (int): 潜在空间维度
            
        功能:
            构建编码器和解码器网络结构
            
        网络架构:
            - 编码器: 2*n -> 256 -> 128 -> (z_dim, z_dim)
            - 解码器: z_dim -> 128 -> 2*n
        """
        super().__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        
        # 编码器：将输入映射到潜在空间的均值和方差
        self.encoder = nn.Sequential(
            ComplexLinear(x_dim, 800), 
            ModReLU(),
            ComplexLinear(800, 456),
            ModReLU(),
            ComplexLinear(456, 128),
            ModReLU()
        )
        
        # 潜在空间参数
        self.fc_mu = ComplexLinear(128, z_dim)      # 均值
        self.fc_logvar = ComplexLinear(128, z_dim)  # 对数方差
        
        # 解码器：从潜在空间重构轨迹
        self.decoder = nn.Sequential(
            ComplexLinear(z_dim, 128),
            ModReLU(), 
            ComplexLinear(128, 800),
            ModReLU(),
            ComplexLinear(800, x_dim)  
        )
    
    def encode(self, x):
        """
        编码函数：将输入数据编码到潜在空间
        
        参数:
            x (torch.Tensor): 输入数据，shape=(batch_size, 2*n)
            
        返回:
            tuple: (mu, logvar)
                - mu (torch.Tensor): 潜在空间均值，shape=(batch_size, z_dim)
                - logvar (torch.Tensor): 潜在空间对数方差，shape=(batch_size, z_dim)
                
        说明:
            编码器学习将输入数据映射到潜在空间的分布参数
            使用对数方差是为了数值稳定性
        """
        # self.encoder(x)返回: 编码后的特征向量
        h = self.encoder(x)
        # 返回: (均值, 对数方差)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        """
        重参数化技巧：从均值和方差采样得到潜在变量
        
        参数:
            mu (torch.Tensor): 均值，shape=(batch_size, z_dim)
            logvar (torch.Tensor): 对数方差，shape=(batch_size, z_dim)
            
        返回:
            torch.Tensor: 采样的潜在变量，shape=(batch_size, z_dim)
            
        数学原理:
            z = μ + σ * ε, 其中 ε ~ N(0,1)
            这种技巧使得采样过程可微分，支持反向传播
            
        说明:
            重参数化技巧是VAE的核心，解决了随机采样不可微分的问题
        """
        # torch.exp返回: 指数运算后的标准差
        std_real = torch.exp(0.5*logvar.real)  # 计算标准差
        std_imag = torch.exp(0.5*logvar.imag)  # 计算标准差
        # torch.randn_like返回: 与std相同形状的随机噪声
        eps_real = torch.randn_like(std_real)   # 采样噪声
        eps_imag = torch.randn_like(std_imag)   # 采样噪声
        # 返回: 重参数化后的潜在变量
        return mu + eps_real*std_real + 1j * eps_imag*std_imag           # 重参数化公式：z = μ + σ * ε
    
    def decode(self, z):
        """
        解码函数：从潜在变量重构轨迹数据
        
        参数:
            z (torch.Tensor): 潜在变量，shape=(batch_size, z_dim)
            
        返回:
            torch.Tensor: 重构的轨迹数据，shape=(batch_size, 2*n)
            
        说明:
            解码器学习从低维潜在空间重构高维轨迹数据
            输出维度与输入维度相同，实现数据重构
        """
        # self.decoder(z)返回: 重构的轨迹数据
        return self.decoder(z)
    
    def forward(self, x):
        """
        前向传播：完整的VAE流程
        
        参数:
            x (torch.Tensor): 输入轨迹数据，shape=(batch_size, 2*n)
            
        返回:
            tuple: (recon_x, mu, logvar, z)
                - recon_x (torch.Tensor): 重构的轨迹数据，shape=(batch_size, 2*n)
                - mu (torch.Tensor): 潜在空间均值，shape=(batch_size, z_dim)
                - logvar (torch.Tensor): 潜在空间对数方差，shape=(batch_size, z_dim)
                - z (torch.Tensor): 采样的潜在变量，shape=(batch_size, z_dim)
                
        流程:
            1. 编码：x -> (mu, logvar)
            2. 重参数化：(mu, logvar) -> z
            3. 解码：z -> recon_x
        """
        # self.encode(x)返回: (均值, 对数方差)
        mu, logvar = self.encode(x)
        # self.reparameterize返回: 采样的潜在变量
        z = self.reparameterize(mu, logvar)
        # self.decode(z)返回: 重构的轨迹数据
        recon_x = self.decode(z)
        # 返回: (重构轨迹, 均值, 对数方差, 潜在变量)
        return recon_x, mu, logvar, z

class Regressor(nn.Module):
    """
    回归器模型
    """
    def __init__(self, z_dim):
        super().__init__()
        
        # 1. 复数特征提取器
        self.feature_extractor = nn.Sequential(
            ComplexLayerNorm(z_dim),
            ComplexLinear(z_dim, 1024),
            ModReLU(),  
            ComplexLinear(1024, 512),
            ModReLU(),
            ComplexLinear(512, 256),
            ModReLU(),
            ComplexLinear(256, 128),
            ModReLU(),
            ComplexLinear(128, 64),
            ModReLU(),
            ComplexLinear(64, 32),
            ModReLU(),
            ComplexLinear(32, 16),
            ModReLU(),
            ComplexLinear(16, 8),
            ModReLU()
        )
        
        # 2. 实数回归头
        # 输入维度为 16 (8 from real part + 8 from imag part)
        self.regression_head = nn.Linear(16, 1)
    
    def forward(self, z):
        # 步骤1: 提取复数特征
        complex_features = self.feature_extractor(z)
        
        # 步骤2: 桥接 - 将复数特征的实部和虚部拼接为实数特征向量
        real_features = torch.cat((complex_features.real, complex_features.imag), dim=1)
        
        # 步骤3: 使用实数头进行最终预测，确保输出为实数
        return self.regression_head(real_features)
