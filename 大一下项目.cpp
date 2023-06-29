#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>

using namespace std;

// 定义激活函数
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// 定义激活函数的导数
double sigmoid_derivative(double x) {
    return sigmoid(x) * (1 - sigmoid(x));
}

// 定义卷积层
class ConvLayer {
public:
    int in_width;
    int in_height;
    int in_depth;
    int kernel_size;
    int kernel_num;
    int out_width;
    int out_height;
    int out_depth;
    vector<vector<vector<vector<double>>>> kernels;
    vector<double> biases;

    ConvLayer(int in_width, int in_height, int in_depth, int kernel_size, int kernel_num) {
        this->in_width = in_width;
        this->in_height = in_height;
        this->in_depth = in_depth;
        this->kernel_size = kernel_size;
        this->kernel_num = kernel_num;
        this->out_width = in_width - kernel_size + 1;
        this->out_height = in_height - kernel_size + 1;
        this->out_depth = kernel_num;

        // 初始化卷积核和偏置
        default_random_engine e(0);
        normal_distribution<double> n(0, 1);
        for (int i = 0; i < kernel_num; i++) {
            vector<vector<vector<double>>> kernel;
            for (int j = 0; j < in_depth; j++) {
                vector<vector<double>> k;
                for (int u = 0; u < kernel_size; u++) {
                    vector<double> row;
                    for (int v = 0; v < kernel_size; v++) {
                        row.push_back(n(e));
                    }
                    k.push_back(row);
                }
                kernel.push_back(k);
            }
            kernels.push_back(kernel);
            biases.push_back(n(e));
        }
    }

    // 前向传播
    vector<vector<vector<double>>> forward(vector<vector<vector<double>>>& input) {
        vector<vector<vector<double>>> output(out_depth);
        for (int i = 0; i < out_depth; i++) {
            output[i] = vector<vector<double>>(out_height);
            for (int j = 0; j < out_height; j++) {
                output[i][j] = vector<double>(out_width);
            }
        }

        for (int d = 0; d < out_depth; d++) {
            for (int h = 0; h < out_height; h++) {
                for (int w = 0; w < out_width; w++) {
                    double sum = 0.0;
                    for (int i = 0; i < in_depth; i++) {
                        for (int j = 0; j < kernel_size; j++) {
                            for (int k = 0; k < kernel_size; k++) {
                                sum += input[i][h + j][w + k] * kernels[d][i][j][k];
                            }
                        }
                    }
                    sum += biases[d];
                    output[d][h][w] = sigmoid(sum);
                }
            }
        }

        return output;
    }
};

// 定义全连接层
class FCLayer {
public:
    int in_num;
    int out_num;
    vector<vector<double>> weights;
    vector<double> biases;

    FCLayer(int in_num, int out_num) {
        this->in_num = in_num;
        this->out_num = out_num;

        // 初始化权重和偏置
        default_random_engine e(0);
        normal_distribution<double> n(0, 1);
        for (int i = 0; i < out_num; i++) {
            vector<double> w(in_num);
            for (int j = 0; j < in_num; j++) {
                w[j] = n(e);
            }
            weights.push_back(w);
            biases.push_back(n(e));
        }
    }

    // 前向传播
    vector<double> forward(vector<double>& input) {
        vector<double> output(out_num);

        for (int i = 0; i < out_num; i++) {
            double sum = 0;
                for (int j = 0; j < in_num; j++) {
                    sum += input[j] * weights[i][j];
                }
            sum += biases[i];
            output[i] = sigmoid(sum);
        }

        return output;
    }
};

// 定义CNN模型
class CNN {
public:
    ConvLayer conv;
    FCLayer fc;

    CNN(int in_width, int in_height, int in_depth, int kernel_size, int kernel_num, int out_num) :
        conv(in_width, in_height, in_depth, kernel_size, kernel_num),
        fc(conv.out_width* conv.out_height* conv.out_depth, out_num) {}

    // 前向传播
    vector<double> forward(vector<vector<vector<double>>>& input) {
        auto conv_output = conv.forward(input);
        vector<double> fc_input(conv.out_width * conv.out_height * conv.out_depth);
        for (int i = 0; i < conv.out_depth; i++) {
            for (int j = 0; j < conv.out_height; j++) {
                for (int k = 0; k < conv.out_width; k++) {
                    fc_input[i * conv.out_height * conv.out_width + j * conv.out_width + k] = conv_output[i][j][k];
                }
            }
        }
        auto fc_output = fc.forward(fc_input);
        return fc_output;
    }
};

int main() {
    // 输入数据
    vector<vector<vector<double>>> input(1);
    input[0] = { {0, 1, 1, 0},
                {1, 0, 0, 1},
                {1, 0, 0, 1},
                {0, 1, 1, 0} };

    // 构建CNN模型
    CNN cnn(4, 4, 1, 2, 2, 10);

    // 前向传播
    auto output = cnn.forward(input);

    // 输出结果
    for (auto o : output) {
        cout << o << " ";
    }
    cout << endl;

    return 0;
}
