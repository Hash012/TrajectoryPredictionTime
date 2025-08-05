# 复杂轨迹数据的VAE-Regressor预测模型

## 1. 项目简介

本项目旨在利用深度学习模型，根据复杂的二维轨迹数据 (`x`, `y`) 来预测一个相关的目标值 `t`（例如时间）。

项目采用两阶段的策略：
1.  **特征提取**：首先，使用一个变分自编码器（Variational Autoencoder, VAE）对输入的复数轨迹数据进行压缩，从未经标注的数据中学习到一个低维度的、信息丰富的潜在特征表示（latent representation）。
2.  **回归预测**：然后，使用一个简单的回归器模型，以VAE提取出的特征作为输入，来预测目标值 `t`。

这种方法将无监督的特征学习与有监督的回归任务相结合，旨在提高模型的预测精度和泛化能力。

## 2. 文件结构

```
ComplexVae-Regressor4Predict/
│
├── datas/                  # 存放原始数据
│   ├── xx.xlsx             # x坐标轨迹数据
│   ├── yy.xlsx             # y坐标轨迹数据
│   └── tt.xlsx             # 目标值t的数据
│
├── trained_models/         # 存放训练好的模型和标准化器
│   ├── vae_model.pth       # VAE模型权重
│   ├── regressor_model.pth # 回归器模型权重
│   ├── scaler_X_real.save  # x轨迹实部标准化器
│   ├── scaler_X_imag.save  # x轨迹虚部标准化器
│   └── scaler_y.save       # y目标值标准化器
│
├── training_results/       # 存放训练过程的可视化结果
│   ├── vae_training_results.png
│   └── regressor_training_results.png
│
├── prediction_results/     # 存放最终的预测结果
│   ├── prediction_results.csv
│   └── prediction_results.png
│
├── model.py                # 定义VAE和Regressor网络结构
├── dataload.py             # 数据加载和预处理逻辑
├── train_vae_only.py       # 脚本：仅训练VAE模型
├── train_regressor_only.py # 脚本：仅训练回归器模型
├── predict.py              # 脚本：使用已训练模型进行预测
└── README.md               # 本说明文件
```

## 3. 环境设置

建议使用 Conda 来管理项目环境，以确保依赖库的一致性。

1.  **创建新的Conda环境**（假设环境名为 `torch_env`）：
    ```bash
    conda create -n torch_env python=3.10 -y
    ```

2.  **激活环境**:
    ```bash
    conda activate torch_env
    ```

3.  **安装必要的库**:
    ```bash
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    pip install pandas numpy scikit-learn joblib matplotlib openpyxl
    ```
    *注意：请根据您机器的CUDA版本调整 `pytorch-cuda` 的版本号。如果您的机器没有NVIDIA GPU，可以安装CPU版本的PyTorch。*

## 4. 使用流程

请严格按照以下顺序执行脚本，以确保流程的正确性。所有命令都应在 `ComplexVae-Regressor4Predict` 目录下执行。

### 第1步：训练VAE模型

首先，我们需要训练VAE来学习数据的潜在特征。
```bash
python train_vae_only.py
```
该脚本会：
- 从 `datas/` 目录加载数据。
- 训练VAE模型。
- 将训练好的 `vae_model.pth` 和数据标准化器 (`scaler_*.save`) 保存到 `trained_models/` 目录。
- 在 `training_results/` 目录中保存训练过程的可视化图表。

### 第2步：训练回归器模型

在VAE训练完成后，我们使用它提取的特征来训练回归器。
```bash
python train_regressor_only.py
```
该脚本会：
- 加载预训练的VAE模型。
- 训练回归器模型。
- 将训练好的 `regressor_model.pth` 保存到 `trained_models/` 目录。
- 在 `training_results/` 目录中保存训练过程的可视化图表。

### 第3步：执行预测

最后，使用训练好的VAE和回归器对测试数据进行预测。
```bash
python predict.py
```
该脚本会：
- 加载所有必要的模型和标准化器。
- 遍历 `datas/` 目录中的所有样本并进行预测。
- 将详细的预测结果（真实值、预测值、误差）输出到终端。

## 5. 输出说明

运行 `predict.py` 脚本后，会在 `prediction_results/` 目录下生成两个文件：

-   `prediction_results.csv`: 包含每个测试样本的真实值、预测值和绝对误差的CSV表格文件。
-   `prediction_results.png`: 将上述表格可视化生成的图片文件，方便快速查看和报告。

##6. 调试心得回忆

- Encoder 和 Decoder 不宜过深，否则过强，则潜在空间的特征不好。可以让两者的神经网络层不退称（差一层）
- 合理的潜在空间维度受制于压缩比的合理区间
- 用mu预测更稳定，因此可以牺牲kl
- 退火
- Kaiming Normal初始化适用于ReLU系列
- HuberLoss对极端数据更稳定
- ……
