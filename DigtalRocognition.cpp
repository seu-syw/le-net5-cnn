#include <iostream>
double mylearning_rate[10] = { 0.0001,0.0001,0.00001,0.00001,0.00001 };
int global_n=0;
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include<easyx.h>
#include <opencv2/opencv.hpp>
#include"idxl-ubyte_to_jpg.h"
#include<direct.h>
//#include "MyCnn.h"
#include"SYWCnn.h"
using namespace std;


int main() {
	//test();
	UIForm form(36);
	Le_net5 le_net5;
	string path = "model/";

	int epoch = 100, n = 60000,testnum=10000, start = 0;;
	//cout << "输入训练轮数（大于100),每轮训练集数目（不大于60000),每轮测试集数目（不大于10000）" << endl;
	////cin >> epoch >> n>>testnum;
	//cout << "输入5个学习率梯度" << endl;
	////cin >> mylearning_rate[0] >> mylearning_rate[1] >> mylearning_rate[2] >> mylearning_rate[3] >> mylearning_rate[4];
	//cout << "从第几号模型开始" << endl;
	////cin >> start;
	start = 0;
	le_net5.load(path + to_string(start)+"/");
	le_net5.test(10000);
	////for (int i = 0; i < 100; i++)
	//	//cout << le_net5.checkNum(DataLoader::read_image(("mnist/test10000/" + to_string(i) + ".jpg").c_str())) << endl;;

	start ++;
	le_net5.train(epoch,n,testnum,start);
	while(1)
		cin>>start;
}



