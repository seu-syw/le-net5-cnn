//此文件为最初搭建的框架模板











#pragma once
//以下需求均为包括但不限于
//构造ui界面
//1.对训练集准确率和测试集准确率的输出
//2.画板，用于手写数字
//3.识别按钮（点击后对手写内容进行识别（此过程需要把手写的内容提取图片（eaysx有）），并输出结果）
//4.载入图片按钮，点击后选择图片文件，在画板显示出来，并输出结果

class UIForm
{
private:
	ExMessage msg;//储存鼠标信息
public:
	void iniForm() {

	}

	bool judge_point(int left, int top, int right, int bottom)
	{
		if (msg.x<right && msg.x>left && msg.y > top && msg.y < bottom)
			return true;
		else
			return false;
	}
	//判断一个点是否在某矩形范围内
	 void formload() {
		 iniForm();
		 while (1)
		 {
			 if (peekmessage(&msg, EX_MOUSE))//检测鼠标信息
			 {
				 if (msg.message == WM_LBUTTONDOWN)
					 if (judge_point(250, 200, 650, 300))//按钮1的位置
					 {
						//这里写按钮1左单击执行的操作
					 }
					 else if (judge_point(250, 400, 650, 500))//按钮2的位置
					 {
						 //这里写按钮2做单击执行的操作
					 }
			 }
		 }
	}
};

//数据预处理
//1.图片的读取，处理
//2.构建数据集
//3.图片处理包括但不限于尺寸放缩
//
class DataLoader
{

};
//激活函数
//1.包含多个激活函数及其导数
//2.通过函数指针实现对不同激活函数的方便切换
//
class ActivationFunction {
private:
		double (*actFunc)(double);//函数指针，指向选用的激活函数
		double (*actFuncD)(double);//选用激活函数的导数
public:
	//构造函数，选择使用哪个激活函数
	ActivationFunction()
	{

	}

	//以下为各激活函数及其导数

};

//卷积核
class Filter
{

};

//卷积层
class ConvLayer
{
private:
	int width;
	int height;
	int fliterNum;
	Filter* filter;

public:
	//构造函数
	ConvLayer()
	{

	}
};

//池化层
class PoolLayer
{
private:

public:
	//构造函数
	PoolLayer()
	{

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