#include "mlp.h"
#include "loss.h"
#include <iostream>
#include <vector>

int main() {
    std::cout << "=== 反向传播测试 ===" << std::endl;
    
    // 创建一个简单的网络：2输入 -> 2隐藏 -> 1输出
    std::vector<int> layer_sizes = {2, 2, 1};
    double learning_rate = 0.1;
    std::string activation = "sigmoid";
    
    MLP mlp(layer_sizes, learning_rate, activation);
    
    std::cout << "网络结构: ";
    for (int i = 0; i < layer_sizes.size(); ++i) {
        std::cout << layer_sizes[i];
        if (i < layer_sizes.size() - 1) std::cout << " -> ";
    }
    std::cout << std::endl;
    
    // 简单的训练数据：XOR问题
    std::vector<Eigen::VectorXd> inputs;
    std::vector<Eigen::VectorXd> targets;
    
    // 正确初始化Eigen向量
    Eigen::VectorXd input1(2); input1 << 0, 0;
    Eigen::VectorXd input2(2); input2 << 0, 1;
    Eigen::VectorXd input3(2); input3 << 1, 0;
    Eigen::VectorXd input4(2); input4 << 1, 1;
    
    Eigen::VectorXd target1(1); target1 << 0;
    Eigen::VectorXd target2(1); target2 << 1;
    Eigen::VectorXd target3(1); target3 << 1;
    Eigen::VectorXd target4(1); target4 << 0;
    
    inputs.push_back(input1);
    inputs.push_back(input2);
    inputs.push_back(input3);
    inputs.push_back(input4);
    
    targets.push_back(target1);
    targets.push_back(target2);
    targets.push_back(target3);
    targets.push_back(target4);
    
    // 训练前测试
    std::cout << "\n=== 训练前预测 ===" << std::endl;
    for (size_t i = 0; i < inputs.size(); ++i) {
        Eigen::VectorXd prediction = mlp.predict(inputs[i]);
        std::cout << "输入: [" << inputs[i](0) << ", " << inputs[i](1) << "]" 
                  << " 预测: " << prediction(0) 
                  << " 目标: " << targets[i](0) << std::endl;
    }
    
    // 训练网络
    std::cout << "\n=== 开始训练 ===" << std::endl;
    int epochs = 1000;
    int batch_size = 1;
    bool verbose = true;
    
    mlp.train(inputs, targets, epochs, batch_size, verbose);
    
    // 训练后测试
    std::cout << "\n=== 训练后预测 ===" << std::endl;
    for (size_t i = 0; i < inputs.size(); ++i) {
        Eigen::VectorXd prediction = mlp.predict(inputs[i]);
        std::cout << "输入: [" << inputs[i](0) << ", " << inputs[i](1) << "]" 
                  << " 预测: " << prediction(0) 
                  << " 目标: " << targets[i](0) << std::endl;
    }
    
    // 计算最终损失
    double final_loss = mlp.evaluate(inputs, targets);
    std::cout << "\n最终损失: " << final_loss << std::endl;
    
    return 0;
}