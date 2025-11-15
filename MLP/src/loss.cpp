#include "../include/loss.h"
#include <stdexcept>

// MSE损失函数实现
double Loss::mse(const Eigen::VectorXd& prediction, const Eigen::VectorXd& target) {
    // MSE = 1/n * Σ(prediction - target)²
    if (prediction.size() != target.size()) {
        throw std::invalid_argument("Prediction and target must have the same size");
    }
    
    Eigen::VectorXd diff = prediction - target;
    return diff.squaredNorm() / prediction.size();
}

// MSE梯度计算
Eigen::VectorXd Loss::mse_gradient(const Eigen::VectorXd& prediction, const Eigen::VectorXd& target) {
    // d(MSE)/d(prediction) = 2/n * (prediction - target)
    if (prediction.size() != target.size()) {
        throw std::invalid_argument("Prediction and target must have the same size");
    }
    
    return 2.0 * (prediction - target) / prediction.size();
}