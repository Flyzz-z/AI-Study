# GAN 模型总结 (Generative Adversarial Networks)

[GAN介绍](https://zhuanlan.zhihu.com/p/266677860)

## 1. 核心架构 (Architecture)
GAN 由两个相互对抗的神经网络组成：

### **生成器 (Generator, G)**
*   **目标**: 凭空“造假”。输入一个随机噪声向量 (Latent Vector)，输出一张逼真的假样本 (Fake Sample)。
*   **设计**:
    *   通常使用 **转置卷积 (Transposed Convolution)** 进行上采样。
    *   将低维的噪声 (如 $100 \times 1 \times 1$) 逐步放大成高维图像 (如 $3 \times 64 \times 64$)。
    *   输出层通常使用 `Tanh` 激活函数，将像素值映射到 $[-1, 1]$。

### **判别器 (Discriminator, D)**
*   **目标**: 火眼金睛“鉴伪”。输入一张图片，判断它是真实图片 (Real) 还是生成器生成的假图 (Fake)。
*   **设计**:
    *   本质是一个**二分类器**。
    *   使用标准的 **卷积层 (Convolution)** 进行下采样和特征提取。
    *   在 WGAN 中，它被称为 Critic（评论家），输出不是概率，而是分数值。

> **形象比喻**:
> 生成器像是一个**伪造假币的罪犯**，判别器像是**警察**。罪犯努力制造看起来像真钱的假币，警察努力区分真伪。随着对抗的进行，两者的水平都越来越高，最终假币几乎无法被区分。

---

## 2. 损失函数 (Loss Functions)

### 2.1 原始 GAN 损失 (Minimax Loss)
经典的二元交叉熵损失：
$$ \min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_{z}(z)}[\log(1 - D(G(z)))] $$
*   **D 的视角**: 最大化公式。希望 $D(x) \to 1$ (真图判真)，$D(G(z)) \to 0$ (假图判假)。
*   **G 的视角**: 最小化公式。希望 $D(G(z)) \to 1$ (让 D 把假图判为真)。

### 2.2 WGAN 损失 (Wasserstein Loss)
为了解决原始 GAN 训练不稳定（主要由 JS 散度导致梯度消失）的问题，WGAN 引入了 Wasserstein 距离。
*   **公式**: $L_D = - \mathbb{E}[D(x)] + \mathbb{E}[D(G(z))]$
*   **区别**: 判别器最后一层**不加 Sigmoid**。
*   **含义**: 判别器单纯给出一个“真实度评分”。它希望真图分高，假图分低。生成器希望假图分高。

### 2.3 梯度惩罚 (Gradient Penalty, WGAN-GP)
*   **背景**: WGAN 要求判别器满足 **1-Lipschitz 连续性**（梯度变化不能太快）。早期做法是粗暴地剪裁权重 (Weight Clipping)，导致参数分布糟糕。
*   **改进**: WGAN-GP (Gradient Penalty) 直接惩罚梯度。
*   **做法**:
    1.  在真图和假图之间随机采样插值点 $\hat{x}$。
    2.  计算 D 对 $\hat{x}$ 的梯度 $\nabla_{\hat{x}} D(\hat{x})$。
    3.  如果梯度的模长 $||\nabla||$ 不等于 1，就施加惩罚 $\lambda (||\nabla|| - 1)^2$。
*   **优点**: 极大地稳定了训练，允许使用更深的网络。

---

## 3. 训练流程 (Training Pipeline)

GAN 的训练是一个**交替迭代**的过程。通常在一个 Loop 中按顺序执行以下步骤：

### **第一步：训练判别器 (Update Discriminator)**
1.  **固定生成器**，只更新判别器的权重。
2.  **获取数据**:
    *   从数据集读取一批 **真图 (Real Inputs)**。
    *   生成一批随机噪声，通过生成器得到 **假图 (Fake Inputs)**。
3.  **前向传播**:
    *   D 给真图打分 -> `real_validity`
    *   D 给假图打分 -> `fake_validity`
4.  **计算损失**:
    *   $Loss_D = -mean(real\_validity) + mean(fake\_validity)$ (WGAN 形式)
    *   如果是 WGAN-GP，计算并加上 **Gradient Penalty**。
5.  **反向传播**: 更新 D 的参数。
*   *注意：在 WGAN 中，为了保持判别器足够强大（以提供准确的梯度指引），通常每更新 1 次 G，会更新 n 次 D (如 n=5)。*

### **第二步：训练生成器 (Update Generator)**
1.  **固定判别器**，只更新生成器的权重。
2.  **生成数据**: 再次采样一批新的随机噪声，生成 **假图**。
3.  **前向传播**: 将假图传入 D，得到评分。
4.  **计算损失**:
    *   $Loss_G = -mean(fake\_validity)$
    *   G 的目标仅仅是让 D 给出的分数越高越好（取负号即由最小化变为最大化）。
5.  **反向传播**: 更新 G 的参数。

### **第三步：循环**
重复上述步骤，直到生成的图像质量达到预期，或 Loss 趋于平稳（虽然 GAN 的 Loss 不一定收敛）。
