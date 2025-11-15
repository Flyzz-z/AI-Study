#include "../include/mlp.h"
#include "../include/loss.h"
#include <iostream>
#include <vector>

int main() {
    std::cout << "=== MLP Loss 测试 ===" << std::endl;

    // 网络结构：2输入 -> 2隐藏 -> 1输出
    std::vector<int> layer_sizes = {2, 2, 1};
    double learning_rate = 0.2; // 提高学习率以加快收敛
    std::string activation = "sigmoid";

    MLP mlp(layer_sizes, learning_rate, activation);

    // XOR 数据集
    std::vector<Eigen::VectorXd> inputs;
    std::vector<Eigen::VectorXd> targets;

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

    // 训练前平均损失
    double loss_before = 0.0;
    for (size_t i = 0; i < inputs.size(); ++i) {
        Eigen::VectorXd pred = mlp.predict(inputs[i]);
        loss_before += Loss::mse(pred, targets[i]);
    }
    loss_before /= static_cast<double>(inputs.size());
    std::cout << "训练前平均MSE: " << loss_before << std::endl;

    // 训练
    int epochs = 5000;  // 增加训练轮数，便于明显下降
    int batch_size = 4; // 使用整批训练，稳定梯度
    bool verbose = false; // 关闭详细日志，只看损失
    mlp.train(inputs, targets, epochs, batch_size, verbose);

    // 训练后平均损失
    double loss_after = 0.0;
    for (size_t i = 0; i < inputs.size(); ++i) {
        Eigen::VectorXd pred = mlp.predict(inputs[i]);
        loss_after += Loss::mse(pred, targets[i]);
    }
    loss_after /= static_cast<double>(inputs.size());
    std::cout << "训练后平均MSE: " << loss_after << std::endl;

    // 简单判断效果
    if (loss_after < loss_before) {
        std::cout << "[OK] 训练降低了损失，MLP实现正常。" << std::endl;
        return 0;
    } else {
        std::cout << "[WARN] 损失未降低，需检查实现或调整超参数。" << std::endl;
        return 1;
    }
}