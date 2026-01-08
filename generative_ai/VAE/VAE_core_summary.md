# VAE（Variational Autoencoder）核心总结

## 1. 建模假设（生成模型）
VAE 首先是一个**生成模型假设**：

```math
z \sim p(z)=\mathcal{N}(0,I), \quad x \sim p_\theta(x\mid z)
```

- 只有 **decoder** 属于生成模型  
- encoder 不属于生成模型

---

## 2. 后验问题
我们真正想要的是：

```math
p(z\mid x)
```

- 由生成模型唯一确定  
- 因为包含不可计算的 $p(x)$，实际算不出来  

---

## 3. Encoder 的角色
Encoder 定义的是近似后验：

```math
q_\phi(z\mid x)
```

- 是人为引入的推断工具  
- 输出的是**分布参数（向量）**，不是 z 本身  

---

## 4. z 的三种分布（关键区分）

| 语境 | z 的分布 |
|---|---|
| 生成模型定义 | $p(z)$（已知先验） |
| 真实但不可算 | $p(z\mid x)$ |
| 训练中使用 | $q(z\mid x)$ |

---

## 5. 损失函数（ELBO）

```math
\mathcal{L}
=
-\mathbb{E}_{q(z\mid x)}[\log p(x\mid z)]
+
\mathrm{KL}(q(z\mid x)\|p(z))
```

---

## 6. 重构项的含义（关键）
```math
\mathbb{E}_{q(z\mid x)}[\log p(x\mid z)]
```

含义：
- 对 encoder 认为可能的所有 z  
- decoder 都要能以高概率生成原始 x  

作用：
- 防止 z 成为“废变量”
- 迫使 z 携带关于 x 的信息

---

## 7. KL 项的含义（关键）
```math
\mathrm{KL}(q(z\mid x)\|p(z))
```

含义：
- 约束所有 z 的整体分布接近先验  
- 防止潜空间碎裂，保证可生成性  

---

## 8. KL 项强弱的影响

| KL 强度 | 结果 |
|---|---|
| 太强 | Posterior Collapse（z 无信息） |
| 太弱 | 潜空间碎裂，不可生成 |
| 合适 | 连续、可插值、可生成 |

---

## 9. 为什么要对 z 取期望
- encoder 输出的是分布而不是确定值  
- loss 衡量的是平均意义下的解释能力  
- 实现时通过采样近似期望  

---

## 10. 使用方式区分

| 场景 | 使用方式 |
|---|---|
| 生成新样本 | $z\sim p(z)$ → decoder |
| 表示学习 | 使用 encoder 的 $\mu(x)$ |
| 重构 / 分析 | encoder + decoder |



## 11. 使用方式区分

## VAE 重参数化（Reparameterization Trick）

### 目的
在训练 VAE 时：
- Encoder 输出的是分布参数 $\mu(x), \sigma(x)$
- 我们需要从分布 $q(z|x) = \mathcal{N}(\mu, \sigma^2)$采样 z  
- **直接采样不可微，梯度无法回传给 μ 和 σ**  

### 核心公式
$$z = \mu(x) + \sigma(x) \odot \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)$$
- **μ(x)**：均值向量，决定中心位置  
- **σ(x)**：标准差向量，控制每维随机幅度  
- **ε**：标准正态噪声，保证采样随机性  
- **⊙**：逐元素相乘

### 原理
1. 高斯分布的线性变换仍然是高斯：
   - $ \varepsilon \sim \mathcal{N}(0, I) $
   - $ z = \mu + \sigma \odot \varepsilon \sim \mathcal{N}(\mu, \sigma^2) $
2. 将随机性 ε 与可学习参数 μ, σ 分开：
   - ε 随机、不可学习  
   - μ, σ 可微 → 梯度可回传
3. 确保 decoder 可以看到具体 z，同时 encoder 能被训练

### 直观理解
- μ → 潜在空间中心  
- σ → 每维允许的随机偏移  
- ε → 随机扰动  
- z = μ + σ⊙ε → 一个具体样本，保留 μ, σ 定义的分布特性  

> 通过这种方式，我们“公式化”地构建了一个高斯分布，并实现了可微采样，VAE 训练和生成样本同时兼顾。

## 一句话终极定锚（非常重要）

> **VAE =
>  假设世界用 z 生成 x，
>  用 encoder 近似反推 z，
>  用 ELBO 同时约束“信息性”和“可生成性”。**
