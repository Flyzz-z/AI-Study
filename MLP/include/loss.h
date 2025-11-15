#ifndef LOSS_H
#define LOSS_H

#include "../Eigen/Dense"
#include <cmath>
#include <string>

class Loss {
public:
    // 计算MSE损失
    static double mse(const Eigen::VectorXd& prediction, const Eigen::VectorXd& target);
    
    // 计算MSE梯度
    static Eigen::VectorXd mse_gradient(const Eigen::VectorXd& prediction, const Eigen::VectorXd& target);
    
private:
    Loss() = delete;  // 静态类，禁止实例化
};

#endif // LOSS_H