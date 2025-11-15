#include "mlp.h"
#include "activation.h"
#include "loss.h"
#include "src/Core/Matrix.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <stdexcept>

// 构造函数
MLP::MLP(const std::vector<int>& layer_sizes, double learning_rate, const std::string& activation)
    : layer_sizes_(layer_sizes), learning_rate_(learning_rate), activation_type_(activation) {
    
    // 验证输入参数
    if (layer_sizes.size() < 2) {
        throw std::invalid_argument("Network must have at least 2 layers (input and output)");
    }
    
    if (learning_rate <= 0) {
        throw std::invalid_argument("Learning rate must be positive");
    }
    
    if (!Activation::is_valid_type(activation)) {
        throw std::invalid_argument("Invalid activation function type");
    }
    
    // 初始化网络结构
    initialize_weights();
}

// 权重初始化
void MLP::initialize_weights() {
    // 初始化权重矩阵
    for(size_t i = 0; i < layer_sizes_.size() - 1; ++i) {
        weights_.emplace_back(layer_sizes_[i+1], layer_sizes_[i]);
        
        // 针对不同激活函数使用不同初始化策略
        if (activation_type_ == "sigmoid") {
            // sigmoid 函数使用 Xavier 初始化
            double xavier_stddev = xavier_initializer(layer_sizes_[i], layer_sizes_[i+1]);
            weights_[i] = Eigen::MatrixXd::Random(layer_sizes_[i+1], layer_sizes_[i]) * xavier_stddev;
        } else if (activation_type_ == "relu") {
            // relu 函数使用 He 初始化
            double he_stddev = he_initializer(layer_sizes_[i]);
            weights_[i] = Eigen::MatrixXd::Random(layer_sizes_[i+1], layer_sizes_[i]) * he_stddev;	
        } else if (activation_type_ == "tanh") {
            // tanh 函数使用 Xavier 初始化（与sigmoid相同）
            double xavier_stddev = xavier_initializer(layer_sizes_[i], layer_sizes_[i+1]);
            weights_[i] = Eigen::MatrixXd::Random(layer_sizes_[i+1], layer_sizes_[i]) * xavier_stddev;
        } else {
            // 默认使用 Xavier 初始化
            double xavier_stddev = xavier_initializer(layer_sizes_[i], layer_sizes_[i+1]);
            weights_[i] = Eigen::MatrixXd::Random(layer_sizes_[i+1], layer_sizes_[i]) * xavier_stddev;
        }
    }

    // 初始化偏置
    for(size_t i = 0; i < layer_sizes_.size() - 1; ++i) {
        biases_.emplace_back(layer_sizes_[i+1]);
        biases_[i].setZero();  // 偏置通常初始化为0
    }
}

// 前向传播
Eigen::VectorXd MLP::forward(const Eigen::VectorXd& input) {
    // 1. 验证输入尺寸
		if(input.size() != layer_sizes_[0]) {
			throw std::invalid_argument("Input size must match first layer size");
		}

		z_values_.clear();
		activations_.clear();


    // 2. 逐层计算：z = Wx + b, a = activation(z)
		auto vec = input;
		for(size_t i = 0; i < layer_sizes_.size() - 1; ++i) {
			// 计算线性组合（使用上一层的激活 vec，而非原始 input）
			Eigen::VectorXd z = weights_[i] * vec + biases_[i];
			z_values_.emplace_back(z);
			// 应用激活函数
			vec = Activation::compute_vector(activation_type_, z);
			activations_.emplace_back(vec);
		}

    return vec; // 返回最终输出
}

// 反向传播
void MLP::backward(const Eigen::VectorXd& input, const Eigen::VectorXd& target) {
    // 1. 前向传播保存中间值（确保z_values_和activations_已填充）
    forward(input);
    
    // 2. 计算输出层误差：δ^L = ∇_a L ⊙ σ'(z^L)
    Eigen::VectorXd output_error = Loss::mse_gradient(activations_.back(), target);
    // 乘以输出层激活函数的导数
    output_error = output_error.array() * Activation::compute_derivative_vector(activation_type_, z_values_.back()).array();
    
    // 3. 存储所有层的梯度（从输出层到输入层）
    std::vector<Eigen::MatrixXd> weight_gradients;
    std::vector<Eigen::VectorXd> bias_gradients;
    
    // 4. 反向传播：从输出层到第一层
    for(int i = layer_sizes_.size() - 2; i >= 0; --i) {
        // 计算当前层的权重梯度：∂L/∂W^l = δ^l * (a^(l-1))^T
        Eigen::MatrixXd grad_w;
        if (i == 0) {
            // 第一层使用输入作为前一层的激活值
            grad_w = output_error * input.transpose();
        } else {
            // 其他层使用前一层保存的激活值
            grad_w = output_error * activations_[i-1].transpose();
        }
        weight_gradients.insert(weight_gradients.begin(), grad_w);
        
        // 计算当前层的偏置梯度：∂L/∂b^l = δ^l
        bias_gradients.insert(bias_gradients.begin(), output_error);
        
        // 如果不是第一层，计算前一层的误差：δ^(l-1) = (W^l)^T * δ^l ⊙ σ'(z^(l-1))
        if (i > 0) {
            Eigen::VectorXd error = weights_[i].transpose() * output_error;
            output_error = error.array() * Activation::compute_derivative_vector(activation_type_, z_values_[i-1]).array();
        }
    }
    
    // 5. 更新权重和偏置（梯度下降）
    for(size_t i = 0; i < weights_.size(); ++i) {
        weights_[i] -= learning_rate_ * weight_gradients[i];
        biases_[i] -= learning_rate_ * bias_gradients[i];
    }
}

// 训练函数
void MLP::train(const std::vector<Eigen::VectorXd>& inputs, 
               const std::vector<Eigen::VectorXd>& targets, 
               int epochs, int batch_size, bool verbose) {
    // 基本参数与数据校验
    if (inputs.size() != targets.size()) {
        throw std::invalid_argument("Inputs and targets must have the same size");
    }
    if (inputs.empty()) {
        throw std::invalid_argument("Training data must not be empty");
    }
    if (epochs <= 0) {
        throw std::invalid_argument("Epochs must be positive");
    }
    if (batch_size <= 0) {
        throw std::invalid_argument("Batch size must be positive");
    }

    // 维度校验
    const int input_dim = layer_sizes_.front();
    const int output_dim = layer_sizes_.back();
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (inputs[i].size() != input_dim) {
            throw std::invalid_argument("Input vector size does not match network input dimension");
        }
        if (targets[i].size() != output_dim) {
            throw std::invalid_argument("Target vector size does not match network output dimension");
        }
    }

    // 训练循环（简单SGD，逐样本更新）
    for (int epoch = 1; epoch <= epochs; ++epoch) {
        double epoch_loss = 0.0;

        // 遍历所有样本并执行反向传播更新
        for (size_t i = 0; i < inputs.size(); ++i) {
            // 反向传播（内部会进行一次参数更新）
            backward(inputs[i], targets[i]);

            // 计算该样本损失（使用MSE）
            Eigen::VectorXd pred = forward(inputs[i]);
            epoch_loss += Loss::mse(pred, targets[i]);
        }

        // 计算并输出平均损失
        epoch_loss /= static_cast<double>(inputs.size());
        if (verbose) {
            std::cout << "Epoch " << epoch << "/" << epochs
                      << " - MSE: " << epoch_loss << std::endl;
        }
    }
}

// 预测函数
Eigen::VectorXd MLP::predict(const Eigen::VectorXd& input) {
    return forward(input);
}


// 损失计算
double MLP::compute_loss(const Eigen::VectorXd& prediction, const Eigen::VectorXd& target) {
    // 维度与数值检查
    if (prediction.size() != target.size()) {
        throw std::invalid_argument("compute_loss: prediction/target size mismatch");
    }
    if (!prediction.allFinite() || !target.allFinite()) {
        throw std::invalid_argument("compute_loss: non-finite values in prediction/target");
    }

    // 使用均方误差（MSE）作为默认损失
    return Loss::mse(prediction, target);
}

double MLP::evaluate(const std::vector<Eigen::VectorXd>& inputs, 
                    const std::vector<Eigen::VectorXd>& targets) {
    double total_loss = 0.0;
    for (size_t i = 0; i < inputs.size(); ++i) {
        Eigen::VectorXd pred = forward(inputs[i]);
        total_loss += compute_loss(pred, targets[i]);
    }
    return total_loss / static_cast<double>(inputs.size());
}

// 工具函数
void MLP::print_network_info() const {
    std::cout << "=== MLP Network Information ===" << std::endl;
    std::cout << "Layer sizes: ";
    for (size_t i = 0; i < layer_sizes_.size(); ++i) {
        std::cout << layer_sizes_[i];
        if (i < layer_sizes_.size() - 1) std::cout << " -> ";
    }
    std::cout << std::endl;
    std::cout << "Learning rate: " << learning_rate_ << std::endl;
    std::cout << "Activation function: " << activation_type_ << std::endl;
    std::cout << "Total parameters: " << "TODO" << std::endl;
}

// 辅助函数
double MLP::activate(double x) {
    return Activation::compute(activation_type_, x);
}

double MLP::activate_derivative(double x) {
    return Activation::compute_derivative(activation_type_, x);
}

double MLP::xavier_initializer(int fan_in, int fan_out) {
    // Xavier/Glorot 初始化：适用于sigmoid和tanh激活函数
    // 权重从均匀分布中采样：U[-sqrt(6/(fan_in + fan_out)), sqrt(6/(fan_in + fan_out))]
    // 或者从正态分布中采样：N(0, sqrt(2/(fan_in + fan_out)))
    double limit = std::sqrt(2.0 / (fan_in + fan_out));
    return limit;
}

double MLP::he_initializer(int fan_in) {
    // He/Kaiming 初始化：适用于ReLU激活函数
    // 权重从正态分布中采样：N(0, sqrt(2/fan_in))
    // 或者从均匀分布中采样：U[-sqrt(6/fan_in), sqrt(6/fan_in)]
    double stddev = std::sqrt(2.0 / fan_in);
    return stddev;
}

// 获取权重统计信息
std::vector<double> MLP::get_weight_stats() const {
    std::vector<double> stats;
    double total_mean = 0.0, total_std = 0.0, total_min = 0.0, total_max = 0.0;
    int total_elements = 0;
    
    for (const auto& weight_matrix : weights_) {
        double mean = weight_matrix.mean();
        double variance = ((weight_matrix.array() - mean).square()).mean();
        double stddev = std::sqrt(variance);
        double min_val = weight_matrix.minCoeff();
        double max_val = weight_matrix.maxCoeff();
        
        total_mean += mean * weight_matrix.size();
        total_std += stddev * weight_matrix.size();
        if (total_elements == 0) {
            total_min = min_val;
            total_max = max_val;
        } else {
            total_min = std::min(total_min, min_val);
            total_max = std::max(total_max, max_val);
        }
        total_elements += weight_matrix.size();
    }
    
    if (total_elements > 0) {
        stats.push_back(total_mean / total_elements);  // mean
        stats.push_back(total_std / total_elements);   // stddev
        stats.push_back(total_min);                      // min
        stats.push_back(total_max);                      // max
    }
    
    return stats;
}

// 获取偏置统计信息
std::vector<double> MLP::get_bias_stats() const {
    std::vector<double> stats;
    double total_mean = 0.0, total_std = 0.0, total_min = 0.0, total_max = 0.0;
    int total_elements = 0;
    
    for (const auto& bias_vector : biases_) {
        double mean = bias_vector.mean();
        double variance = ((bias_vector.array() - mean).square()).mean();
        double stddev = std::sqrt(variance);
        double min_val = bias_vector.minCoeff();
        double max_val = bias_vector.maxCoeff();
        
        total_mean += mean * bias_vector.size();
        total_std += stddev * bias_vector.size();
        if (total_elements == 0) {
            total_min = min_val;
            total_max = max_val;
        } else {
            total_min = std::min(total_min, min_val);
            total_max = std::max(total_max, max_val);
        }
        total_elements += bias_vector.size();
    }
    
    if (total_elements > 0) {
        stats.push_back(total_mean / total_elements);  // mean
        stats.push_back(total_std / total_elements);   // stddev
        stats.push_back(total_min);                      // min
        stats.push_back(total_max);                      // max
    }
    
    return stats;
}