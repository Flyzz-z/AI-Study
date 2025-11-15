#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <cmath>
#include <string>
#include <stdexcept>
#include "../Eigen/Dense"

class Activation {
public:
    // 激活函数计算（标量版本）
    static double compute(const std::string& type, double x);
    
    // 激活函数计算（向量版本）
    static Eigen::VectorXd compute_vector(const std::string& type, const Eigen::VectorXd& x);
    
    // 激活函数导数计算（标量版本）
    static double compute_derivative(const std::string& type, double x);
    
    // 激活函数导数计算（向量版本）
    static Eigen::VectorXd compute_derivative_vector(const std::string& type, const Eigen::VectorXd& x);
    
    // 具体激活函数实现（标量版本）
    static double sigmoid(double x);
    static double relu(double x);
    static double tanh_func(double x);
    static double softmax_component(double x, const double* exp_sum);
    
    // 具体激活函数实现（向量版本）
    static Eigen::VectorXd sigmoid_vector(const Eigen::VectorXd& x);
    static Eigen::VectorXd relu_vector(const Eigen::VectorXd& x);
    static Eigen::VectorXd tanh_vector(const Eigen::VectorXd& x);
    
    // 导数函数（标量版本）
    static double sigmoid_derivative(double x);
    static double relu_derivative(double x);
    static double tanh_derivative(double x);
	
    // 导数函数（向量版本）
    static Eigen::VectorXd sigmoid_derivative_vector(const Eigen::VectorXd& x);
    static Eigen::VectorXd relu_derivative_vector(const Eigen::VectorXd& x);
    static Eigen::VectorXd tanh_derivative_vector(const Eigen::VectorXd& x);
    
    // 验证激活函数类型
    static bool is_valid_type(const std::string& type);
    
private:
    Activation() = delete;  // 静态类，禁止实例化
};

#endif // ACTIVATION_H