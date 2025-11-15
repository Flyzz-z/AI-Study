#ifndef MLP_H
#define MLP_H

#include "../Eigen/Dense"
#include <vector>
#include <functional>
#include <string>
#include <random>

class MLP {
private:
    std::vector<int> layer_sizes_;                    // 网络层大小
    std::vector<Eigen::MatrixXd> weights_;            // 权重矩阵
    std::vector<Eigen::VectorXd> biases_;             // 偏置向量
    std::vector<Eigen::VectorXd> activations_;        // 激活值
    std::vector<Eigen::VectorXd> z_values_;           // 线性组合值
    
    double learning_rate_;
    std::string activation_type_;
    
    // 激活函数
    double activate(double x);
    double activate_derivative(double x);
    
    // 初始化方法
    void initialize_weights();
    double xavier_initializer(int fan_in, int fan_out);
    double he_initializer(int fan_in);
    
public:
    // 构造函数和析构函数
    MLP(const std::vector<int>& layer_sizes, double learning_rate = 0.01, 
        const std::string& activation = "relu");
    ~MLP() = default;
    
    // 核心功能
    Eigen::VectorXd forward(const Eigen::VectorXd& input);
    void backward(const Eigen::VectorXd& input, const Eigen::VectorXd& target);
    
    // 训练接口
    void train(const std::vector<Eigen::VectorXd>& inputs, 
               const std::vector<Eigen::VectorXd>& targets, 
               int epochs, int batch_size = 1, bool verbose = true);
    
    // 预测接口
    Eigen::VectorXd predict(const Eigen::VectorXd& input);
    std::vector<Eigen::VectorXd> predict_batch(const std::vector<Eigen::VectorXd>& inputs);
    
    // 评估功能
    double compute_loss(const Eigen::VectorXd& prediction, const Eigen::VectorXd& target);
    double evaluate(const std::vector<Eigen::VectorXd>& inputs, 
                   const std::vector<Eigen::VectorXd>& targets);
    
    // 工具函数
    void print_network_info() const;
    
    // 获取器
    std::vector<int> get_layer_sizes() const { return layer_sizes_; }
    double get_learning_rate() const { return learning_rate_; }
    std::string get_activation_type() const { return activation_type_; }
    
    // 测试函数：获取权重和偏置的统计信息
    std::vector<double> get_weight_stats() const;
    std::vector<double> get_bias_stats() const;
};

#endif // MLP_H