#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <opencv2/opencv.hpp>
#include <fstream>
#include<graphics.h>
using namespace std;
using namespace cv;
namespace Matrix
{
	void resize(vector<vector<double>>& a, int m, int n)
	{
		a.resize(m);
		for (int i = 0; i < m; i++)
		{
			a[i].resize(n);
		}
	}
	double Loss(const vector<double>& a)
	{
		double c = 0;
		for (int i1 = 0; i1 < a.size(); i1++)
		{
			c = c + a[i1] * a[i1];
			c = 0.5 * sqrt(c);
		}
		return c;
	}
	double sum(const vector<vector<double>>& a)
	{
		double sum = 0;
		for (int i1 = 0; i1 < a.size(); i1++)
			for (int i2 = 0; i2 < a[i1].size(); i2++)
				sum += a[i1][i2];
		return sum;
	}
	vector<vector<double>>rot180(const vector<vector<double>>& a)
	{
		vector<vector<double>> b = a;
		for (int i1 = 0; i1 < a.size(); i1++)
			for (int i2 = 0; i2 < a[i1].size(); i2++)
			{
				b[i1][i2] = a[a.size() - 1 - i1][a[i1].size()-1-i2];
			}
		return b;
	}
	vector<double>hadamard(const vector<double>& a, const vector<double>& b)
	{
		vector<double> c;
		double tmp;
		if (a.size() != b.size())
			return c;
		for (int i1 = 0; i1 < a.size(); i1++)
		{
			tmp = a[i1] * b[i1];
			c.push_back(tmp);
		}
		return c;
	}
	vector<vector<double>>transpose(const vector<vector<double>>& a)
	{
		vector<vector<double>> c;
		c.resize(a[0].size());
		for (int i0 = 0; i0 < a[0].size(); i0++)
		{
			for (int i1 = 0; i1 < a.size(); i1++)
			{
				c[i0].push_back(a[i1][i0]);
			}
		}
		return c;
	}
	vector<vector<double>>hadamard(const vector<vector<double>>& a, const vector<vector<double>>& b)
	{

		vector<vector<double>> c;
		if (a.size() != b.size())
			return c;
		c.resize(a.size());
		for (int i0 = 0; i0 < a.size(); i0++)
		{
			if (a[i0].size() != b[i0].size())
				return c;
			for (int i1 = 0; i1 < a[i0].size(); i1++)
			{
				double tmp = 0;
				tmp = a[i0][i1] * b[i0][i1];
				c[i0].push_back(tmp);
			}
		}
		return c;
	}
	vector<vector<double>>multiply(const vector<vector<double>>& a, const  vector<vector<double>>& b)
	{

		vector<vector<double>> c;
		if (a[0].size() != b.size())
			return c;
		c.resize(a.size());
		for (int i0 = 0; i0 < a.size(); i0++)
		{
			for (int i2 = 0; i2 < b[i0].size(); i2++)
			{
				double tmp = 0;
				for (int i1 = 0; i1 < a[i0].size(); i1++)
				{
					tmp += a[i0][i1] * b[i1][i2];
				}
				c[i0].push_back(tmp);
			}
		}
		return c;
	}
	vector<double>multiply(const vector<double>& a, const vector<vector<double>>& b)
	{
		vector<double> c;
		if (a.size() != b.size())
			return c;
		for (int i2 = 0; i2 < b[0].size(); i2++)
		{
			double tmp = 0;
			for (int i1 = 0; i1 < a.size(); i1++)
			{
				tmp += a[i1] * b[i1][i2];
			}
			c.push_back(tmp);
		}
		return c;
	}
	vector<vector<double>>multiply(const vector<double>& a, const vector<double>& b)
	{
		vector<vector<double>> c;
		c.resize(a.size());
		for (int i1 = 0; i1 < a.size(); i1++)
		{
			for (int i2 = 0; i2 < b.size(); i2++)
				c[i1].push_back(a[i1] * b[i2]);
		}
		return c;
	}
	vector<double>multiply(const vector<vector<double>>& a, const  vector<double>& b)
	{
		vector<double> c;

		for (int i1 = 0; i1 < a.size(); i1++)
		{
			if (a[i1].size() != b.size())
				return c;
			double tmp = 0;
			for (int i2 = 0; i2 < b.size(); i2++)
			{
				tmp += b[i2] * a[i1][i2];
			}
			c.push_back(tmp);
		}
		return c;
	}
	vector<vector<double>> subtract(const vector<vector<double>>& a, const  vector<vector<double>>& b)
	{
		vector<vector<double>> c;
		if (a.size() != b.size() || a[0].size() != b[0].size())
			return c;
		c.resize(a.size());
		for (int i0 = 0; i0 < a.size(); i0++)
		{
			for (int i1 = 0; i1 < a[i0].size(); i1++)
			{
				c[i0].push_back(a[i0][i1] - b[i0][i1]);
			}
		}
		return c;
	}
	vector<double>subtract(const vector<double>& a, const  vector<double>& b)
	{
		vector<double> c;
		if (a.size() != b.size())
			return c;
		for (int i1 = 0; i1 < a.size(); i1++)
		{
			c.push_back(a[i1] - b[i1]);
		}
		return c;
	}
	vector<vector<double>>add(const vector<vector<double>>& a, const  vector<vector<double>>& b)
	{
		vector<vector<double>> c;
		if (a.size() != b.size() || a[0].size() != b[0].size())
			return c;
		c.resize(a.size());
		for (int i0 = 0; i0 < a.size(); i0++)
		{
			for (int i1 = 0; i1 < a[i0].size(); i1++)
			{
				c[i0].push_back(a[i0][i1] + b[i0][i1]);
			}
		}
		return c;
	}
	vector<double>add(const vector<double>& a, const  vector<double>& b)
	{
		vector<double> c;
		if (a.size() != b.size())
			return c;
		for (int i1 = 0; i1 < a.size(); i1++)
		{
			c.push_back(a[i1] + b[i1]);
		}
		return c;
	}
	vector<vector<double>>n_multiply(const vector<vector<double>>& a, const  double& k)
	{

		vector<vector<double>> c;
		c.resize(a.size());
		for (int i0 = 0; i0 < a.size(); i0++)
		{
			double tmp = 0;
			for (int i1 = 0; i1 < a[i0].size(); i1++)
			{
				c[i0].push_back(a[i0][i1] * k);
			}
		}
		return c;
	}
	vector<double>n_multiply(const vector<double>& a, const double& k)
	{
		vector<double> c;
		for (int i = 0; i < a.size(); i++)
		{
			c.push_back(a[i] * k);
		}
		return c;
	}
	vector<vector<double>>conv(const vector<vector<double>>& input, const  vector<vector<double>>& kernel, int stride = 1)
	{
		vector<vector<double>>c;
		int m = (input.size() - kernel.size()) / stride + 1;
		int n = (input[0].size() - kernel[0].size()) / stride + 1;
		resize(c, m, n);
		for (int i1 = 0, m = 0; i1 < input.size() - kernel.size() + 1; i1 += stride, m++)
			for (int i2 = 0, n = 0; i2 < input[i1].size() - kernel[0].size() + 1; i2 += stride, n++)
			{
				double tmp = 0;
				for (int i4 = 0; i4 < kernel.size(); i4++)
					for (int i5 = 0; i5 < kernel.size(); i5++)
					{
						tmp += input[i1 + i4][i2 + i5] * kernel[i4][i5];
					}
				c[m][n] = tmp;
			}
		return c;
	}
	vector<vector<vector<double>>>padding(const vector<vector<vector<double>>>& input,int circle_num,int fill_num)
	{
		vector<vector<vector<double>>> a = input;
		for (int i = 0; i < circle_num; i++)
		{
			for (int i0 = 0; i0 <a.size(); i0++)
			{
				// 在每一行的开头和结尾插入一个 0
				for (auto& row : a[i0]) {
					row.insert(row.begin(), fill_num);
					row.push_back(fill_num);
				}

				// 在二维向量的开头和结尾插入一行全为 0 的新行
				int rowSize = a[i0][0].size();
				a[i0].insert(a[i0].begin(), std::vector<double>(rowSize, fill_num));
				a[i0].push_back(std::vector<double>(rowSize, fill_num));
			}
		}
		return a;
	}
	vector<vector<double>>padding(const vector<vector<double>>& input, int circle_num, int fill_num)
	{
		vector<vector<double>> a = input;
		for (int i = 0; i < circle_num; i++)
		{

			// 在每一行的开头和结尾插入一个 0
			for (auto& row : a) {
				row.insert(row.begin(),fill_num);
				row.push_back(fill_num);
			}

			// 在二维向量的开头和结尾插入一行全为 0 的新行
			int rowSize = a[0].size();
			a.insert(a.begin(), std::vector<double>(rowSize, fill_num));
			a.push_back(std::vector<double>(rowSize, fill_num));
		}
		return a;
	}
	void out_matrix(const vector<vector<double>>& a)
	{
		for (int i0 = 0; i0 < a.size(); i0++)
		{
			for (int i1 = 0; i1 < a[i0].size(); i1++)
			{
				cout << a[i0][i1] << " ";
			}
			cout << endl;
		}
		cout << endl;
	}
	void out_matrix(const vector<double>& a)
	{
		for (int i0 = 0; i0 < a.size(); i0++)
		{
			cout << a[i0] << " ";
		}
		cout << endl;
	}
	void test()
	{
		vector<vector<double>> a = { {1,2,3,0} ,{4,5,6,0} ,{7,8,9,0},{0,0,0,0} };
		vector<vector<double>> b = { {1,2,3,0} ,{4,5,6,0} ,{7,8,9,0},{0,0,0,0} };
		//vector<double> c = { 1,2,3,0 };
		cout<<sum(b);
		out_matrix(padding(a,2,0));

		//out_matrix(multiply(c, a));
	//	out_matrix(add(a, b));
		//out_matrix(add(c, c));
	}
}
using namespace Matrix;


//数据预处理
//1.图片的读取，处理
//2.构建数据集
//3.图片处理包括但不限于尺寸放缩
//
class DataLoader
{
public:
	vector<vector<vector<vector<double>>>>batch;


	const char* train_images_path = "mnist/train";
	const char* train_labels_path = "mnist/";
	vector<vector<vector<vector<double>>>> read()
	{
		vector<vector<vector<vector<double>>>>target;
		Mat srcimg = imread("test.jpg");
		// 定义裁剪区域
		Rect roi(0, 0, srcimg.cols - 100, srcimg.rows);

		// 裁剪图片
		Mat dst = srcimg(roi);

		cvtColor(dst, dst, COLOR_BGR2GRAY);

		threshold(dst, dst, 150, 255, 1);//二值化
		vector < vector<Point> > edgepoint;//轮廓

		vector<Vec4i> lclass;

		findContours(dst, edgepoint, lclass, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point());//检索轮廓

		Mat* mat = new Mat[edgepoint.size()];
		target.resize(edgepoint.size());
		for (int i = 0; i < edgepoint.size(); ++i)
		{

			Rect rec = boundingRect(Mat(edgepoint[i]));

			mat[i] = dst(rec);
			padding(mat[i]);
			rectangle(dst, rec, Scalar(100, 80, 90), 1, 1, 0);

			drawContours(dst, edgepoint, i, Scalar(200), 1, 8, lclass);

			string str = to_string(i);
			if (mat[i].rows > 0.1 * srcimg.rows || mat[i].cols > 0.1 * srcimg.cols)//滤波器，排除障碍点
			{
				resize(mat[i], mat[i], Size(56, 56), INTER_AREA);
				resize(mat[i], mat[i], Size(28, 28), INTER_AREA);
				//imshow(str, mat[i]);
				std::vector<vector<double>> vec(mat[i].rows, vector<double>(mat[i].cols));
				for (int i1 = 0; i1 < mat[i].rows; i1++)
				{
					for (int j = 0; j < mat[i].cols; j++)
					{
						vec[i1][j] = mat[i].at<uchar>(i1, j)>100?255:0;
						mat[i].at<uchar>(i1, j)= mat[i].at<uchar>(i1, j) > 100 ? 255 : 0;
					}
				}
				target[i].push_back(vec);
				imwrite("newmnist/"+to_string(global_n++)+".jpg", mat[i]);
			}
		}
		//imshow("input", dst);
		/*	imwrite("s", dst);*/
		waitKey(0);
		delete[] mat;
		return target;
	}
	void padding(Mat& img)
	{
		int top, bottom, left, right;
		if (img.rows > img.cols)
		{
			top = 0;
			bottom = 0;
			left = (img.rows - img.cols) / 2;
			right = img.rows - img.cols - left;
		}
		else
		{
			top = (img.cols - img.rows) / 2;
			bottom = img.cols - img.rows - top;
			left = 0;
			right = 0;
		}

		// 扩充图片
		copyMakeBorder(img, img, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
		// 计算需要再次扩充的边界宽度
		int border_width = img.rows / 5;

		// 再次扩充图片
		cv::copyMakeBorder(img, img, border_width, border_width, border_width, border_width, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
	}
	vector<vector<vector<vector<double>>>>batch_build(const char* file_path, int batchsize = 1)
	{
		this->batch.resize(batchsize);
		for (int i = 0; i < batchsize; i++)
			this->batch[i] = read_image(file_path);
		return this->batch;
	}
	static vector<vector<vector<double>>> read_image(const char* file_path)
	{
		// 读取图像
		Mat image = imread(file_path, IMREAD_GRAYSCALE);
		// 转换图像
		vector<vector<vector<double>>> converted_image;
		converted_image.resize(image.channels());
		for (int i0 = 0; i0 < image.channels(); i0++)
		{
			converted_image[i0].resize(image.rows);
			for (int i = 0; i < image.rows; i++)
			{
				for (int j = 0; j < image.cols; j++)
				{
					converted_image[i0][i].push_back(image.at<uchar>(i, j) < 100 ? 0 : 255);
				}
			}
		}
		return converted_image;
	}
};




//激活函数
//1.包含多个激活函数及其导数
//2.通过函数指针实现对不同激活函数的方便切换
enum ActivationFunctionType { SIGMOID = 0, TANH = 1, RELU = 2, LRELU = 3 };
enum PoolingType { MAX = 0, AVERAGE = 1 };
class ActivationFunction {
private:
	double (ActivationFunction::* actFunc)(double);//函数指针，指向选用的激活函数
	double (ActivationFunction::* actFuncD)(double);//选用激活函数的导数
public:
	//构造函数，选择使用哪个激活函数
	//SIGMOID 0, TANH 1, RELU 2
	ActivationFunction(ActivationFunctionType type)
	{
		change_function(type);
	}
	ActivationFunction()
	{
		actFunc = &ActivationFunction::sigmoid;
		actFuncD = &ActivationFunction::sigmoid_derivative;
	}
	void change_function(ActivationFunctionType type)
	{
		switch (type)
		{
		case SIGMOID:
			actFunc = &ActivationFunction::sigmoid;
			actFuncD = &ActivationFunction::sigmoid_derivative;
			break;
		case TANH:
			actFunc = &ActivationFunction::tanh;
			actFuncD = &ActivationFunction::tanh_derivative;
			break;
		case RELU:
			actFunc = &ActivationFunction::relu;
			actFuncD = &ActivationFunction::relu_derivative;
			break;
		case LRELU:
			actFunc = &ActivationFunction::lrelu;
			actFuncD = &ActivationFunction::lrelu_derivative;
			break;
		default:
			actFunc = &ActivationFunction::sigmoid;
			actFuncD = &ActivationFunction::sigmoid_derivative;
			break;
		}
	}
	double getY(double x)
	{
		return (this->*actFunc)(x);
	}
	double getdY(double x)
	{
		return (this->*actFuncD)(x);
	}
	//以下为各激活函数及其导数
	double sigmoid(double x)
	{
		return 1.0 / 1 + exp(-x);
	}
	double sigmoid_derivative(double x)
	{
		return sigmoid(x) * (1 - sigmoid(x));
	}
	double tanh(double x)
	{
		return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
	}
	double tanh_derivative(double x)
	{
		return 4 / ((exp(x) + exp(-x)) * (exp(x) + exp(-x)));
	}
	double relu(double x)
	{
		if (x > 0)
		{
			return x;
		}
		else
		{
			return 0;
		}
	}
	double relu_derivative(double x)
	{
		if (x > 0)
		{
			return 1.0;
		}
		else
		{
			return 0;
		}
	}
	double lrelu(double x)
	{
		if (x > 0)
		{
			return x;
		}
		else
		{
			return 0.01 * x;
		}
	}
	double lrelu_derivative(double x)
	{
		if (x > 0)
		{
			return 1.0;
		}
		else
		{
			return 0.01;
		}
	}
};

//卷积核
class Filter
{
public:
	int width = 3;//宽度
	int channel = 1;//通道数
	double bias = 0;//偏置数
	vector<vector<vector<double>>>kernels;
	Filter(int channel, int width)
	{
		this->width = width;
		this->channel = channel;
		this->kernels.resize(channel);
		// 初始化权重
		random_device rd;
		default_random_engine generator(rd());//创建了一个默认的随机数生成器 generator。这个生成器可以产生均匀分布的随机整数。
		normal_distribution<double> distribution(0.0, 0.01);//一个均值为 0.0，标准差为 1.0 的正态分布。
		this->bias = distribution(generator);
		for (int i0 = 0; i0 < this->channel; i0++)
		{
			this->kernels[i0].resize(this->width);
			for (int i1 = 0; i1 < this->width; i1++)
			{
				for (int i2 = 0; i2 < this->width; i2++)
					this->kernels[i0][i1].push_back(distribution(generator));
			}
		}
	}
	void out_kernels()
	{
		for (int i0 = 0; i0 < this->channel; i0++)
		{
			for (int i1 = 0; i1 < this->width; i1++)
			{
				for (int i2 = 0; i2 < this->width; i2++)
					cout << this->kernels[i0][i1][i2] << " ";
				cout << endl;
			}
			cout << endl;
		}
		cout << this->bias;
		cout << endl;
	}
	void save(const string& path) {
		ofstream file(path);
		if (file.is_open()) {
			file << bias << endl;
			for (int i0 = 0; i0 < channel; i0++) {
				for (int i1 = 0; i1 < width; i1++) {
					for (int i2 = 0; i2 < width; i2++)
						file << kernels[i0][i1][i2] << " ";
					file << endl;
				}
				file << endl;
			}
			file.close();
		}
	}

	void load(const string& path) {
		ifstream file(path);
		if (file.is_open()) {
			file >> bias;
			for (int i0 = 0; i0 < channel; i0++) {
				for (int i1 = 0; i1 < width; i1++) {
					for (int i2 = 0; i2 < width; i2++)
						file >> kernels[i0][i1][i2];
				}
			}
			file.close();
		}
	}

};

//卷积层
class ConvLayer
{
private:
	int in_channel;//输入的通道数,即卷积核的通道数
	int width;//卷积核宽度
	int out_channel;//输出的通道数，即卷积核个数
	int padding = 0;//充填圈数
	int stride = 1;//步长
	ActivationFunction afunc;
	vector<Filter>filter;
	vector<vector<vector<double>>>input;
	vector<vector<vector<double>>>feature_map;
	vector<vector<vector<double>>>d_feature_map;
	vector<vector<vector<double>>>loss;

	//确定特征图的大小
	void feature_map_resize()
	{
		this->feature_map.resize(this->out_channel);
		this->d_feature_map.resize(this->out_channel);
		int m = (this->input[0].size() - width) / this->stride + 1;
		int n = (this->input[0][0].size() - width) / this->stride + 1;
		for (int i0 = 0; i0 < this->out_channel; i0++)
		{
			this->feature_map[i0].resize(m);
			this->d_feature_map[i0].resize(m);
			for (int i1 = 0; i1 < m; i1++)
			{
				this->feature_map[i0][i1].resize(n);
				this->d_feature_map[i0][i1].resize(n);
			}
		}
	}
	//边缘扩充
	void Padding()
	{
		for (int i = 0; i < this->padding; i++)
		{
			for (int i0 = 0; i0 < this->input.size(); i0++)
			{
				// 在每一行的开头和结尾插入一个 0
				for (auto& row : this->input[i0]) {
					row.insert(row.begin(), 0);
					row.push_back(0);
				}

				// 在二维向量的开头和结尾插入一行全为 0 的新行
				int rowSize = this->input[i0][0].size();
				this->input[i0].insert(this->input[i0].begin(), std::vector<double>(rowSize, 0));
				this->input[i0].push_back(std::vector<double>(rowSize, 0));
			}
		}
	}
public:
	//构造函数
	ConvLayer(int in_channel, int width, int out_channel, int padding = 0, int stride = 1, ActivationFunctionType activate_type = RELU)
	{
		this->in_channel = in_channel;
		this->width = width;
		this->out_channel = out_channel;
		this->padding = padding;
		this->stride = stride;
		this->afunc.change_function(activate_type);
		for (int i = 0; i < out_channel; i++)
		{
			this->filter.push_back(Filter(in_channel, width));
		}
	}

	//前向传播
	vector<vector<vector<double>>> forward(const vector<vector<vector<double>>>& input)
	{
		this->input = input;
		vector<vector<vector<double>>>null;
		this->feature_map = null;
		Padding();
		feature_map_resize();
		int m = 0, n = 0;
		for (int i0 = 0; i0 < this->out_channel; i0++)
		{
			for (int i1 = 0; i1 < this->in_channel; i1++)
				for (int i2 = 0, m = 0; i2 < this->input[i1].size() - this->width + 1; i2 += this->stride, m++)
					for (int i3 = 0, n = 0; i3 < this->input[i1][i2].size() - this->width + 1; i3 += this->stride, n++)
					{
						double tmp = 0;
						for (int i4 = 0; i4 < this->filter[i0].width; i4++)
							for (int i5 = 0; i5 < this->filter[i0].width; i5++)
							{
								tmp += this->input[i1][i2 + i4][i3 + i5] * this->filter[i0].kernels[i1][i4][i5];
								//getchar();

							}
						this->feature_map[i0][m][n] += tmp;
					}
			for (int i1 = 0; i1 < this->feature_map[i0].size(); i1++)
				for (int i2 = 0; i2 < this->feature_map[i0][i1].size(); i2++)
				{
					this->feature_map[i0][i1][i2] = this->afunc.getY(this->feature_map[i0][i1][i2] + this->filter[i0].bias);
					this->d_feature_map[i0][i1][i2] = this->afunc.getdY(this->feature_map[i0][i1][i2] + this->filter[i0].bias);
				}
		}
		return this->feature_map;
	}
	//反向传播
	vector<vector<vector<double>>> backward(vector<vector<vector<double>>>& loss, bool isLastConv=false )
	{
		if (isLastConv)
		{
			this->loss = loss;
		}
		else
		{
			this->loss.resize(this->out_channel);
			for (int i0 = 0; i0 < this->out_channel; i0++)
			{
				this->loss[i0] = hadamard(loss[i0], this->d_feature_map[i0]);
			}
		}
		vector<vector<vector<double>>>newloss;
		newloss.resize(this->in_channel);
	
		loss = Matrix::padding(loss, this->width - 1, 0);
		for (int i0 = 0; i0 < this->in_channel; i0++)
		{
			resize(newloss[i0], this->input[i0].size(), this->input[i0][0].size());
			for (int i1 = 0; i1 < this->out_channel; i1++)
				newloss[i0] =add(newloss[i0], conv(loss[i1], rot180(this->filter[i1].kernels[i0])));
		}
		loss = newloss;
		return newloss;
	}
	void weights_update(double learning_rate)
	{
		for (int i0 = 0; i0 < this->out_channel; i0++)
		{
			for (int i1 = 0; i1 < this->in_channel; i1++)
			{
				this->filter[i0].kernels[i1] =subtract(this->filter[i0].kernels[i1],n_multiply(conv(this->input[i1], this->loss[i0]),learning_rate));
				
			}
			double sum = 0;
			sum = Matrix::sum(this->loss[i0]);
			this->filter[i0].bias -= sum * learning_rate;
		}
	}
	//输出特征图
	void out_feature_map()
	{
		for (int i0 = 0; i0 < this->feature_map.size(); i0++)
		{
			for (int i1 = 0; i1 < this->feature_map[i0].size(); i1++)
			{
				for (int i2 = 0; i2 < this->feature_map[i0][i1].size(); i2++)
					cout << this->feature_map[i0][i1][i2] << " ";
				cout << endl;
			}
			cout << endl;
		}
		cout << endl;
	}
	void save(const string& path) {
		for (int i = 0; i < this->out_channel; i++)
		{
			this->filter[i].save(path+"fliter"+to_string(i));
		}
	}

	void load(const string& path) {
		for (int i = 0; i < this->out_channel; i++)
		{
			this->filter[i].load(path + "fliter" + to_string(i));
		}
	}
	vector<vector<vector<double>>>feature_map_get()
	{
		return this->feature_map;
	}
	vector<vector<vector<double>>>input_get()
	{
		return this->input;
	}
};

//池化层
class PoolLayer
{
private:
	int size = 2;//窗口大小
	int stride = 2;//步长
	int padding = 0;//扩充
	void (PoolLayer::* pool)(void);
	void (PoolLayer::* back_mode)(vector<vector<vector<double>>>&);
	vector<vector<vector<double>>>loss;
	vector<vector<vector<double>>>input;
	vector<vector<vector<double>>>record;//记录哪几个是最大
	vector<vector<vector<double>>>feature_map;

	//确定特征图的大小
	void feature_map_resize()
	{
		this->feature_map.resize(this->input.size());
		int m = (this->input[0].size() - size) / this->stride + 1;
		int n = (this->input[0][0].size() - size) / this->stride + 1;
		for (int i0 = 0; i0 < this->input.size(); i0++)
		{
			this->feature_map[i0].resize(m);
			for (int i1 = 0; i1 < m; i1++)
			{
				this->feature_map[i0][i1].resize(n);
			}
		}
	}
	//边缘扩充
	void Padding()
	{
		for (int i = 0; i < this->padding; i++)
		{
			for (int i0 = 0; i0 < this->input.size(); i0++)
			{
				// 在每一行的开头和结尾插入一个 0
				for (auto& row : this->input[i0]) {
					row.insert(row.begin(), 0);
					row.push_back(0);
				}

				// 在二维向量的开头和结尾插入一行全为 0 的新行
				int rowSize = this->input[i0][0].size();
				this->input[i0].insert(this->input[i0].begin(), std::vector<double>(rowSize, 0));
				this->input[i0].push_back(std::vector<double>(rowSize, 0));
			}
		}
	}
public:
	//构造函数
	PoolLayer(int size, int stride, int padding = 0, PoolingType type = MAX)
	{
		this->size = size;
		this->stride = stride;
		this->padding = padding;
		switch (type)
		{
		case MAX:
			pool = &PoolLayer::max_pooling;
			back_mode = &PoolLayer::max_back;
			break;
		case AVERAGE:
			pool = &PoolLayer::average_pooling;
			back_mode = &PoolLayer::average_back;
			break;
		default:
			pool = &PoolLayer::max_pooling;
			back_mode = &PoolLayer::max_back;
			break;
		}
	}
	vector<vector<vector<double>>> forward(const vector<vector<vector<double>>>& input)
	{
		this->input = input;
		vector<vector<vector<double>>>null;
		this->feature_map = null;
		Padding();
		feature_map_resize();
		(this->*pool)();
		return this->feature_map;
	}
	vector<vector<vector<double>>> backward(vector<vector<vector<double>>>& loss)
	{
		(this->*back_mode)(loss);
		return loss;
	}
	void max_pooling()
	{
		vector<vector<vector<double>>>null;
		this->record = null;
		this->record.resize(this->input.size());
		for (int i0 = 0; i0 < this->input.size(); i0++)
		{
			this->record[i0].resize(this->input[i0].size());
			for (int i1 = 0; i1 < this->input[i0].size(); i1++)
			{
				this->record[i0][i1].resize(this->input[i0][i1].size());
			}
		}
		for (int i1 = 0; i1 < this->input.size(); i1++)
			for (int i2 = 0, m = 0; i2 < this->input[i1].size() - this->size + 1; i2 += this->stride, m++)
				for (int i3 = 0, n = 0; i3 < this->input[i1][i2].size() - this->size + 1; i3 += this->stride, n++)
				{
					double tmp = input[i1][i2][i3];
					int maxm = 0, maxn = 0;
					for (int i4 = 0; i4 < this->size; i4++)
						for (int i5 = 0; i5 < this->size; i5++)
						{
							if (input[i1][i2 + i4][i3 + i5] >= tmp)
							{
								tmp = input[i1][i2 + i4][i3 + i5];
								maxm = i2 + i4;
								maxn = i3 + i5;
							}
						}
					this->record[i1][maxm][maxn] = 1;
					this->feature_map[i1][m][n] = tmp;
				}
	}
	void max_back(vector<vector<vector<double>>>& loss)
	{
		vector<vector<vector<double>>>newloss;
		newloss.resize(this->input.size());
		for (int i0 = 0; i0 < this->input.size(); i0++)
		{
			newloss[i0].resize(this->input[i0].size());
			for (int i1 = 0; i1 < this->input[i0].size(); i1++)
			{
				newloss[i0][i1].resize(this->input[i0][i1].size());
			}
		}
		for (int i1 = 0; i1 < this->input.size(); i1++)
			for (int i2 = 0, m = 0; i2 < this->input[i1].size() - this->size + 1; i2 += this->stride, m++)
				for (int i3 = 0, n = 0; i3 < this->input[i1][i2].size() - this->size + 1; i3 += this->stride, n++)
				{
					for (int i4 = 0; i4 < this->size; i4++)
						for (int i5 = 0; i5 < this->size; i5++)
						{
							newloss[i1][i2 + i4][i3 + i5] += loss[i1][m][n];
						}

				}
		loss = newloss;
		for (int i0 = 0; i0 < this->input.size(); i0++)
		{
			loss[i0] = hadamard(newloss[i0], this->record[i0]);
		}
	}
	void average_pooling()
	{
		for (int i1 = 0; i1 < this->input.size(); i1++)
			for (int i2 = 0, m = 0; i2 < this->input[i1].size() - this->size + 1; i2 += this->stride, m++)
				for (int i3 = 0, n = 0; i3 < this->input[i1][i2].size() - this->size + 1; i3 += this->stride, n++)
				{
					double tmp = 0;
					for (int i4 = 0; i4 < this->size; i4++)
						for (int i5 = 0; i5 < this->size; i5++)
						{
							tmp += this->input[i1][i2 + i4][i3 + i5];
						}
					this->feature_map[i1][m][n] = tmp / (this->size * this->size);
				}
	}
	void average_back(vector<vector<vector<double>>>& loss)
	{
		vector<vector<vector<double>>>newloss;
		newloss.resize(this->input.size());
		for (int i0 = 0; i0 < this->input.size(); i0++)
		{
			newloss[i0].resize(this->input[i0].size());
			for (int i1 = 0; i1 < this->input[i0].size(); i1++)
			{
				newloss.resize(this->input[i0][i1].size());
			}
		}
		int n = this->size * this->size;
		for (int i1 = 0; i1 < this->input.size(); i1++)
			for (int i2 = 0, m = 0; i2 < this->input[i1].size() - this->size + 1; i2 += this->stride, m++)
				for (int i3 = 0, n = 0; i3 < this->input[i1][i2].size() - this->size + 1; i3 += this->stride, n++)
				{
					for (int i4 = 0; i4 < this->size; i4++)
						for (int i5 = 0; i5 < this->size; i5++)
						{
							newloss[i1][i2 + i4][i3 + i5] += loss[i1][m][n] / n;
						}

				}
		loss = newloss;
	}
	void out_feature_map()
	{
		for (int i0 = 0; i0 < this->feature_map.size(); i0++)
		{
			for (int i1 = 0; i1 < this->feature_map[i0].size(); i1++)
			{
				for (int i2 = 0; i2 < this->feature_map[i0][i1].size(); i2++)
					cout << this->feature_map[i0][i1][i2] << " ";
				cout << endl;
			}
			cout << endl;
		}
		cout << endl;
	}
	vector<vector<vector<double>>>get_fold(vector<double> flatten)
	{
		vector<vector<vector<double>>>loss;
		int k = this->feature_map.size(), m = this->feature_map[0].size(), n = this->feature_map[0][0].size();
		loss.resize(k);
		for (int i0 = 0; i0 < k; i0++)
		{
			loss[i0].resize(m);
			for (int i1 = 0; i1 < m; i1++)
			{
				loss[i0][i1].resize(n);
				for (int i2 = 0; i2 < n; i2++)
					loss[i0][i1][i2] = flatten[i0 * m * n + i1 * m + i2];
			}
		}
		this->loss = loss;
		return loss;
	}
	vector<vector<vector<double>>>feature_map_get()
	{
		return this->feature_map;
	}
	vector<vector<vector<double>>>input_get()
	{
		return this->input;
	}
};

//全连接层
class FullConnectLayer
{
private:
	int in_size;
	int out_size;
	ActivationFunction afunc;
	vector<vector<double>>weights;
	vector<double>loss;
	vector<double> biases;
	vector<double>input;
	vector<double>feature_vector;
	vector<double>d_feature_vector;

public:
	FullConnectLayer(int in_size, int out_size, ActivationFunctionType activate_type = RELU)
	{
		this->in_size = in_size;
		this->out_size = out_size;
		this->afunc.change_function(activate_type);
		this->weights.resize(this->out_size);
		this->d_feature_vector.resize(this->out_size);
		this->biases.resize(this->out_size);
		random_device rd;
		default_random_engine generator(rd());//创建了一个默认的随机数生成器 generator。这个生成器可以产生均匀分布的随机整数。
		normal_distribution<double> distribution(0.0, 0.01);//一个均值为 0.0，标准差为 1.0 的正态分布。
		for (int i0 = 0; i0 < this->out_size; i0++)
		{
			for (int i1 = 0; i1 < this->in_size; i1++)
			{
				this->weights[i0].push_back(distribution(generator));
			}
			this->biases[i0] = distribution(generator);
		}
	}
	vector<double>forward(const vector<double>& input)
	{
		this->input = input;
		this->feature_vector = add(multiply(this->weights, input), this->biases);
		for (int i0 = 0; i0 < this->feature_vector.size(); i0++)
		{
			this->d_feature_vector[i0] = this->afunc.getdY(this->feature_vector[i0]);
			this->feature_vector[i0] = this->afunc.getY(this->feature_vector[i0]);
		}
		return this->feature_vector;
	}
	vector<double>backward(vector<double>& loss, double learning_rate, bool outlayer = false,int label = 0)
	{
		if (outlayer)
		{
			loss.resize(feature_vector.size(), 0);
			loss[label] = 1;
			//cout << Loss(subtract(feature_vector, loss)) << endl;
			loss = hadamard(subtract(feature_vector, loss), this->d_feature_vector);
		}
		else
		{
			loss = hadamard(loss, this->d_feature_vector);
		}
		this->loss = loss;
		//weights_update(learning_rate);
		//this->weights = subtract(this->weights, n_multiply(multiply(loss, this->input), learning_rate));
		//this->biases = subtract(this->biases, n_multiply(loss, learning_rate));
		loss = multiply(transpose(this->weights), loss);
		return loss;
	}
	void weights_update(double learning_rate)
	{
		this->weights = subtract(this->weights, n_multiply(multiply(this->loss, this->input), learning_rate));
		this->biases = subtract(this->biases, n_multiply(this->loss, learning_rate));
	}
	void out_feature_vector()
	{
		out_matrix(this->feature_vector);
	}
	void out_weights()
	{
		out_matrix(this->weights);
	}
	void out_biases()
	{
		out_matrix(this->biases);
	}
	vector<double>get_flatten(const vector<vector<vector<double>>>& feature_map)
	{
		vector<double>flatten;
		for (int i0 = 0; i0 < feature_map.size(); i0++)
		{
			for (int i1 = 0; i1 < feature_map[i0].size(); i1++)
			{
				for (int i2 = 0; i2 < feature_map[i0][i1].size(); i2++)
					flatten.push_back(feature_map[i0][i1][i2]);
			}
		}
		return flatten;
	}
	vector<double>feature_vector_get()
	{
		return this->feature_vector;
	}
	vector<double>input_get()
	{
		return this->input;
	}
	void save(const string& path) {
		ofstream file(path);
		if (file.is_open()) {
			for (int i = 0; i < out_size; i++) {
				file << biases[i] << " ";
				for (int j = 0; j < in_size; j++)
					file << weights[i][j] << " ";
				file << endl;
			}
			file.close();
		}
	}

	void load(const string& path) {
		ifstream file(path);
		if (file.is_open()) {
			for (int i = 0; i < out_size; i++) {
				file >> biases[i];
				for (int j = 0; j < in_size; j++)
					file >> weights[i][j];
			}
			file.close();
		}
	}

};

class Le_net5
{
private:
	ConvLayer* conv1;
	PoolLayer* pool1;
	ConvLayer* conv2;
	PoolLayer* pool2;
	FullConnectLayer* fc1;
	FullConnectLayer* fc2;
	FullConnectLayer* fc3;
public:
	Le_net5() : conv1(new ConvLayer(1, 5, 6, 2, 1, LRELU)), pool1(new PoolLayer(2, 2)), conv2(new ConvLayer(6, 5, 16, 0, 1, LRELU)), pool2(new PoolLayer(2, 2)), fc1(new FullConnectLayer(400, 120, LRELU)), fc2(new FullConnectLayer(120, 84, LRELU)), fc3(new FullConnectLayer(84, 10, LRELU))
	{
	}
	vector<double> forward(const vector<vector<vector<double>>>& input)
	{
		conv1->forward(input);
		pool1->forward(conv1->feature_map_get());
		conv2->forward(pool1->feature_map_get());
		pool2->forward(conv2->feature_map_get());
		fc1->forward(fc1->get_flatten(pool2->feature_map_get()));
		fc2->forward(fc1->feature_vector_get());
		fc3->forward(fc2->feature_vector_get());
		//fc3->out_biases();
		//fc3->out_feature_vector();
		return fc3->feature_vector_get();
	}
	void backward(int label, double learning_rate)
	{
		vector<double>loss;
		//fc3->out_biases();
		fc3->backward(loss, learning_rate, true, label);
		fc2->backward(loss, learning_rate);
		fc1->backward(loss, learning_rate);
		vector<vector<vector<double>>>newloss;
		newloss = pool2->get_fold(loss);
		pool2->backward(newloss);
		conv2->backward(newloss);
		pool1->backward(newloss);
		conv1->backward(newloss);
	}
	void update(double learning_rate)
	{
		fc3->weights_update(learning_rate);
		fc2->weights_update(learning_rate);
		fc1->weights_update(learning_rate);
		conv2->weights_update(learning_rate);
		conv1->weights_update(learning_rate);

	}
	void train(int epoch, int n, int testnum,int start,double learning_rate = 0.01)
	{
		int label[60000] = { 0 };
		int num[10] = { 0 };
		ifstream file("mnist/train60000/label.txt");
		if (file.is_open()) {
			for (int i = 0; i < n; i++) {
				file >> label[i];
				num[label[i]]++;
			}
			file.close();
		}
		double tmp = 0;
		int guess = 0, rate[1000][10] = { 0 };
		for (int i0 = start; i0 <= epoch; i0++)
		{
			if (i0 <= 20)
				learning_rate = mylearning_rate[0];
			else if (i0 <= 50)
				learning_rate = mylearning_rate[1];
			else if (i0 <= 100)
				learning_rate = mylearning_rate[2];
			else if (i0 <= 500)
				learning_rate = mylearning_rate[3];
			else
				learning_rate = mylearning_rate[4];
			//if (i0 <= 20)
			//	learning_rate = 0.01;
			//else if (i0 <= 50)
			//	learning_rate = 0.003;
			//else if (i0 <= 100)
			//	learning_rate = 0.003;
			//else if (i0 <= 500)
			//	learning_rate = 0.001;
			//else
			//	learning_rate = 0.0003;
			for (int i1 = 0; i1 < n; i1++)
			{
				//cout << "第" << i0 << "轮训练的第" << i1 << "张" << endl;
				string url = "mnist/train60000/" + to_string(i1) + ".jpg";
				vector<double>result = this->forward(DataLoader::read_image(url.c_str()));
				//cout << label[i1] << "  ";
				this->backward(label[i1], learning_rate);
				this->update(learning_rate);
				tmp = result[0];
				guess = 0;
				for (int i3 = 1; i3 < 10; i3++)
				{
					if (tmp < result[i3])
					{
						tmp = result[i3];
						guess = i3;
					}
				}
				if (guess == label[i1])
				{
					rate[i0][label[i1]]++;
				}
			}
			cout << "第" << i0 << "轮测试:" << endl;
			test(testnum);
			save("model/", i0);
		}

		for (int i0 = 1; i0 <= epoch; i0++)
		{
			int sum = 0;
			cout << "第" << i0 << "轮训练的正确率为";
			for (int i1 = 0; i1 < 10; i1++)
			{
				cout << (double)rate[i0][i1] / num[i1] << " ";
				sum += rate[i0][i1];
			}
			cout << endl;
			cout << "总正确率为" << (double)sum / n << endl;
		}
	}
	void test(int n)
	{
		int label[10000] = { 0 };
		int num[10] = { 0 };
		ifstream file("mnist/test10000/label.txt");
		if (file.is_open()) {
			for (int i = 0; i < n; i++) {
				file >> label[i];
				num[label[i]]++;
			}
			file.close();
		}
		double tmp = 0.0;
		int guess = 0, rate[10] = { 0 };
		for (int i1 = 0; i1 < n; i1++)
		{
			//cout << "第" << i0 << "轮测试的第" << i1 << "张" << endl;
			string url = "mnist/test10000/" + to_string(i1) + ".jpg";
			vector<double>result = this->forward(DataLoader::read_image(url.c_str()));
			tmp = result[0];
			guess = 0;
			for (int i3 = 1; i3 < 10; i3++)
			{
				if (tmp < result[i3])
				{
					tmp = result[i3];
					guess = i3;
				}
			}
			if (guess == label[i1])
			{
				rate[label[i1]]++;
			}

		}
		int sum = 0;
		for (int i1 = 0; i1 < 10; i1++)
		{
			cout << (double)rate[i1] / num[i1] << " ";
			sum += rate[i1];
		}
		cout << endl;
		cout << "总正确率为" << (double)sum / n << endl;
	}

	int checkNum(const vector<vector<vector<double>>>& input)
	{
		bool flag = false;
		if (input.size() == 1)
			if (input[0].size() == 28)
			{
				flag = true;
			}
		if (flag)
		{
			for (int i = 0; i < 28; i++)
				if (input[0][i].size() != 28)
				{
					flag = false;
				}
		}
		if (flag)
		{
			vector<double>result = this->forward(input);
			double tmp = result[0];
			int guess = 0;
			for (int i = 1; i < 10; i++)
			{
				if (tmp < result[i])
				{
					tmp = result[i];
					guess = i;
				}
			}
			return guess;
		}
		else
			return 100;
	}
	void save(string path, int epoch)
	{
		string a = path + to_string(epoch);
		_mkdir(a.c_str());
		conv1->save(a + "/conv1.txt");
		conv2->save(a + "/conv2.txt");
		fc1->save(a + "/fc1.txt");
		fc2->save(a + "/fc2.txt");
		fc3->save(a + "/fc3.txt");
	}
	void load(string path)
	{
		conv1->load(path + "conv1.txt");
		conv2->load(path + "conv2.txt");
		fc1->load(path + "fc1.txt");
		fc2->load(path + "fc2.txt");
		fc3->load(path + "fc3.txt");
	}
};
//在这里搭建网络
class Vgg16
{
public:
	//构造函数，于此完成
	Vgg16()
	{

	}



	//以下为训练与测试相关部分
	void forward()
	{

	}
	void backward()
	{

	}
	void train()
	{

	}
	void test()
	{

	}
};

//以下需求均为包括但不限于
//构造ui界面
//1.对训练集准确率和测试集准确率的输出
//2.画板，用于手写数字
//3.识别按钮（点击后对手写内容进行识别（此过程需要把手写的内容提取图片（eaysx有）），并输出结果）
//4.载入图片按钮，点击后选择图片文件，在画板显示出来，并输出结果
class UIForm
{
private:
	Le_net5 le_net5;
	ExMessage msg;//储存鼠标信息
	struct Point
	{
		int x;
		int y;

		Point() { x = -1; y = -1; }//缺省参数

		Point(int x, int y)
		{
			this->x = x;
			this->y = y;
		}

		bool operator==(const Point& p)//运算符==重载
		{
			return x == p.x && y == p.y;//运算符=重载
		}

		Point& operator=(const Point& p)
		{
			x = p.x; y = p.y;
			return *this;
		}
	};
	// 所有控件的父类
	struct Tool
	{
		int left;
		int top;
		int width;
		int height;

		Tool() {}

		Tool(int left, int top, int width, int height)
		{
			this->left = left;
			this->top = top;
			this->width = width;
			this->height = height;
		}

		virtual double call(MOUSEMSG) = 0;

		bool isIn(const Point& p)
		{
			if (p.x >= left && p.x <= left + width && p.y >= top && p.y <= top + height)
			{
				return true;
			}
			return false;
		}
	};
	// 作为矩形用
	struct Tool_Range :Tool
	{
		Tool_Range() {}

		Tool_Range(int left, int top, int width, int height) :Tool(left, top, width, height) {}

		double call(MOUSEMSG m)
		{
			return 0;
		}
	};
	// 可画画的区域
	struct Tablet :Tool
	{
	private:
		bool isDown;
		Point begPos;
	public:
		int size;
		COLORREF color;

		Tablet() {}

		Tablet(int left, int top, int width, int height, int size, COLORREF color) :Tool(left, top, width, height)
		{
			this->size = size;
			this->color = color;
			isDown = false;
		}

		double call(MOUSEMSG m)//进行画图
		{
			if (m.uMsg == WM_LBUTTONDOWN)
			{
				if (isIn(Point(m.x, m.y)))
				{
					isDown = true;
					begPos = Point(m.x, m.y);
				}
			}

			if (m.uMsg == WM_LBUTTONUP)
			{
				isDown = false;
			}

			if (m.uMsg == WM_MOUSEMOVE && isDown)
			{
				if (isIn(begPos) && isIn(Point(m.x, m.y)))		// 在区域内
				{
					setlinestyle(PS_ENDCAP_ROUND, size);//线条形式
					setlinecolor(color);//线条颜色
					HRGN rgn = CreateRectRgn(left, top, left + width, top + height);
					setcliprgn(rgn);//设置裁剪区，即绘图区
					DeleteObject(rgn);
					line(begPos.x, begPos.y, m.x, m.y);//画直线（起始点x，起始点y，终止点x，终止点y）
					setcliprgn(NULL);
				}
				begPos = Point(m.x, m.y);
			}

			return 1;
		}
	};

	// 简单标签
	struct Label :Tool
	{
		wstring s;
		COLORREF color;
		bool isCenteral;

		Label() {}

		Label(int left, int top, int width, int height, wstring s, COLORREF color, bool isCenteral = false) :Tool(left, top, width, height)
		{
			this->color = color;
			this->s = s;
		}

		double call(MOUSEMSG m)
		{
			setfillcolor(0xefefef);
			solidrectangle(left, top, left + width, top + height);
			settextcolor(color);
			setbkmode(TRANSPARENT);//透明文字背景
			settextstyle(16, 13, L"Courier");
			RECT rect = { left, top, left + width, top + height };
			if (isCenteral)//是否要居中
				drawtext(s.c_str(), &rect, DT_CENTER | DT_TOP | DT_WORDBREAK);
			else
				drawtext(s.c_str(), &rect, DT_LEFT | DT_TOP | DT_WORDBREAK);

			return 1;
		}

		bool isClick(MOUSEMSG m)//左键弹起
		{
			if (m.uMsg == WM_LBUTTONUP)
			{
				if (isIn(Point(m.x, m.y)))
				{
					return true;
				}
			}
			return false;
		}
	};
public:
	UIForm(int n)
	{
		load_model(n);
		iniForm();
	}
	void iniForm() //产生ui界面
	{
		const Point defaultWH(800, 600);	//（1100,768）									// 默认宽高
		initgraph(defaultWH.x, defaultWH.y);									// 初始化
		BeginBatchDraw();//转换图像至ui界面
		setfillcolor(0xffffff);
		fillrectangle(0, 0, defaultWH.x, defaultWH.y);							// 默认画图板背景
		setfillcolor(0xefefef);
		fillrectangle(defaultWH.x - 100, 0, defaultWH.x, defaultWH.y);			// 默认工具栏背景

		// 定义控件及其样式
		Tablet mainTablet(0, 0, defaultWH.x - 100, defaultWH.y, 25, 0x000000);//主要绘图界面
		Rect tablet(0, 0, defaultWH.x - 100, defaultWH.y);//绘图界面的矩形
		Label label1(730, 20, 40, 40, L"重新绘制", 0, 1);//申请这样一个类，需要再到下方去调用此类的call
		Label label2(730, 90, 40, 40, L"进行分析", 0, 1);
		setcolor(0);//转换边框颜色为黑色
		rectangle(720, 10, 780, 70);//“重新绘制”按钮
		rectangle(720, 80, 780, 140);//“进行分析”按钮
		while (true)
		{
			while (MouseHit())
			{
				MOUSEMSG m = GetMouseMsg();

				if (m.x >= 0 && m.x < defaultWH.x - 100 && m.y >= 0 && m.y <= defaultWH.y)
				{
					//控件之间的交互关系
					mainTablet.call(m);//进行画图
				}
				if (label1.isClick(m))
				{
					setbkcolor(WHITE);
					clearrectangle(0, 0, 700, 600);
					clearrectangle(720, 150, 800, 800);
				}
				if (label2.isClick(m))
				{
					saveimage(L"test.jpg");
					setbkcolor(WHITE);
					clearrectangle(720, 150, 800, 800);
					DataLoader read;
					vector<vector<vector<vector<double>>>>target = read.read();
					TCHAR str[20];
					int i = 0;
					for (; i < target.size(); i++)
					{
						_stprintf_s(str, _T("%d"), le_net5.checkNum(target[i]));
						outtextxy(720, 150+20*i, str);
					}
					
				}
			}
			label1.call(MOUSEMSG());
			label2.call(MOUSEMSG());
			FlushBatchDraw();//执行未完成的绘制任务
			Sleep(50);
		}
	}
	void load_model(int start)
	{
		string path = "model/";
		le_net5.load(path + to_string(start) + "/");
		//le_net5.test(1000);
	}
	int PutImage()
	{
		char imageName[] = "/test.jpg";
		Mat M = imread(imageName, IMREAD_COLOR);   // 读入图片 

		if (M.empty())     // 判断文件是否正常打开  
		{
			fprintf(stderr, "Can not load image %s\n", imageName);
			waitKey(6000);  // 等待6000 ms后窗口自动关闭   
			return -1;
		}

		imshow("image", M);  // 显示图片 
		waitKey();
		//imwrite("pic.bmp", M); // 存为bmp格式图片
		return 0;
	}
};