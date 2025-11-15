# MLP (Multi-Layer Perceptron) Implementation

基于Eigen库的C++多层感知机实现。

## 项目结构

```
MLP/
├── include/          # 头文件
│   ├── mlp.h        # 主MLP类定义
│   ├── activation.h # 激活函数工具类
│   └── loss.h       # 损失函数工具类
├── src/             # 源文件
│   ├── mlp.cpp      # MLP类实现
│   ├── activation.cpp # 激活函数实现
│   └── loss.cpp     # 损失函数实现
├── examples/        # 示例代码
│   └── example.cpp  # 使用示例
├── Eigen/           # Eigen线性代数库
└── CMakeLists.txt   # 构建配置
```

## 功能特性

- ✅ 支持多种激活函数（ReLU, Sigmoid, Tanh, Leaky ReLU）
- ✅ 支持多种损失函数（MSE, Cross-Entropy, Binary Cross-Entropy）
- ✅ 灵活的网络架构配置
- ✅ 批量训练和预测
- ✅ 模型保存和加载
- ✅ 数值稳定性处理

## 构建和运行

```bash
# 创建构建目录
mkdir build && cd build

# 配置项目
cmake ..

# 构建
make

# 运行示例
./mlp_example
```

## 使用示例

```cpp
#include "include/mlp.h"

// 创建网络：2输入 -> 4隐藏 -> 1输出
std::vector<int> architecture = {2, 4, 1};
MLP mlp(architecture, 0.1, "sigmoid");

// 训练数据
std::vector<Eigen::VectorXd> inputs = {...};
std::vector<Eigen::VectorXd> targets = {...};

// 训练网络
mlp.train(inputs, targets, 1000, 1, true);

// 预测
auto prediction = mlp.predict(input);
```

## 待实现功能

- [ ] 权重初始化（Xavier, He初始化）
- [ ] 正则化（L1, L2, Dropout）
- [ ] 优化器（Momentum, Adam, RMSprop）
- [ ] 学习率调度
- [ ] 批归一化
- [ ] 更复杂的网络架构支持
- [ ] GPU加速支持

## 注意事项

1. 确保Eigen库已正确安装
2. 需要C++11或更高版本
3. 建议使用CMake 3.10或更高版本

## 许可证

本项目仅供学习和研究使用。