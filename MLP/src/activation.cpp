#include "../include/activation.h"
#include <algorithm>

// 主接口函数（标量版本）
double Activation::compute(const std::string &type, double x) {
  if (type == "sigmoid") {
    return sigmoid(x);
  } else if (type == "relu") {
    return relu(x);
  } else if (type == "tanh") {
    return tanh_func(x);
  } else {
    throw std::invalid_argument("Unknown activation type: " + type);
  }
}

// 主接口函数（向量版本）
Eigen::VectorXd Activation::compute_vector(const std::string &type,
                                           const Eigen::VectorXd &x) {
  if (type == "sigmoid") {
    return sigmoid_vector(x);
  } else if (type == "relu") {
    return relu_vector(x);
  } else if (type == "tanh") {
    return tanh_vector(x);
  } else {
    throw std::invalid_argument("Unknown activation type: " + type);
  }
}

// 导数计算函数（标量版本）
double Activation::compute_derivative(const std::string &type, double x) {
  if (type == "sigmoid") {
    return sigmoid_derivative(x);
  } else if (type == "relu") {
    return relu_derivative(x);
  } else if (type == "tanh") {
    return tanh_derivative(x);
  } else {
    throw std::invalid_argument("Unknown activation type: " + type);
  }
}

// 导数计算函数（向量版本）
Eigen::VectorXd
Activation::compute_derivative_vector(const std::string &type,
                                      const Eigen::VectorXd &x) {
  if (type == "sigmoid") {
    return sigmoid_derivative_vector(x);
  } else if (type == "relu") {
    return relu_derivative_vector(x);
  } else if (type == "tanh") {
    return tanh_derivative_vector(x);
  } else {
    throw std::invalid_argument("Unknown activation type: " + type);
  }
}

// 具体激活函数实现
double Activation::sigmoid(double x) {
  // Sigmoid函数：1 / (1 + exp(-x))
  // 数值稳定性处理：当x很大或很小时避免溢出
  if (x > 10.0) {
    return 1.0; // 当x很大时，exp(-x)接近0
  } else if (x < -10.0) {
    return 0.0; // 当x很小时，exp(-x)很大，1/(1+exp(-x))接近0
  } else {
    return 1.0 / (1.0 + std::exp(-x));
  }
}

double Activation::relu(double x) { return std::max(0.0, x); }

double Activation::tanh_func(double x) { return std::tanh(x); }

// 导数函数实现
double Activation::sigmoid_derivative(double x) {
  // Sigmoid导数：sigmoid(x) * (1 - sigmoid(x))
  double s = sigmoid(x);
  return s * (1.0 - s);
}

double Activation::relu_derivative(double x) { return (x > 0) ? 1.0 : 0.0; }

double Activation::tanh_derivative(double x) {
  double tanh_x = tanh_func(x);
  return 1.0 - tanh_x * tanh_x;
}

// 验证函数
bool Activation::is_valid_type(const std::string &type) {
  return type == "sigmoid" || type == "relu" || type == "tanh" ||
         type == "leaky_relu";
}

// 向量版本激活函数实现
Eigen::VectorXd Activation::sigmoid_vector(const Eigen::VectorXd &x) {
  // Sigmoid: 1 / (1 + exp(-x))
  // 使用数组操作来避免逐个元素计算
  return 1.0 / (1.0 + (-x.array()).exp());
}

Eigen::VectorXd Activation::relu_vector(const Eigen::VectorXd &x) {
  // ReLU: max(0, x)
  return x.array().max(0.0);
}

Eigen::VectorXd Activation::tanh_vector(const Eigen::VectorXd &x) {
  // Tanh: 使用Eigen内置的tanh函数
  return x.array().tanh();
}

// 向量版本导数函数实现
Eigen::VectorXd
Activation::sigmoid_derivative_vector(const Eigen::VectorXd &x) {
  // Sigmoid导数: sigmoid(x) * (1 - sigmoid(x))
  Eigen::VectorXd sig = sigmoid_vector(x);
  return sig.array() * (1.0 - sig.array());
}

Eigen::VectorXd Activation::relu_derivative_vector(const Eigen::VectorXd &x) {
  // ReLU导数: (x > 0) ? 1.0 : 0.0
  return (x.array() > 0.0).cast<double>();
}

Eigen::VectorXd Activation::tanh_derivative_vector(const Eigen::VectorXd &x) {
  // Tanh导数: 1 - tanh²(x)
  Eigen::VectorXd tanh_x = tanh_vector(x);
  return 1.0 - tanh_x.array().square();
}

