//���ļ�Ϊ�����Ŀ��ģ��











#pragma once
//���������Ϊ������������
//����ui����
//1.��ѵ����׼ȷ�ʺͲ��Լ�׼ȷ�ʵ����
//2.���壬������д����
//3.ʶ��ť����������д���ݽ���ʶ�𣨴˹�����Ҫ����д��������ȡͼƬ��eaysx�У���������������
//4.����ͼƬ��ť�������ѡ��ͼƬ�ļ����ڻ�����ʾ��������������

class UIForm
{
private:
	ExMessage msg;//���������Ϣ
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
	//�ж�һ�����Ƿ���ĳ���η�Χ��
	 void formload() {
		 iniForm();
		 while (1)
		 {
			 if (peekmessage(&msg, EX_MOUSE))//��������Ϣ
			 {
				 if (msg.message == WM_LBUTTONDOWN)
					 if (judge_point(250, 200, 650, 300))//��ť1��λ��
					 {
						//����д��ť1�󵥻�ִ�еĲ���
					 }
					 else if (judge_point(250, 400, 650, 500))//��ť2��λ��
					 {
						 //����д��ť2������ִ�еĲ���
					 }
			 }
		 }
	}
};

//����Ԥ����
//1.ͼƬ�Ķ�ȡ������
//2.�������ݼ�
//3.ͼƬ��������������ڳߴ����
//
class DataLoader
{

};
//�����
//1.���������������䵼��
//2.ͨ������ָ��ʵ�ֶԲ�ͬ������ķ����л�
//
class ActivationFunction {
private:
		double (*actFunc)(double);//����ָ�룬ָ��ѡ�õļ����
		double (*actFuncD)(double);//ѡ�ü�����ĵ���
public:
	//���캯����ѡ��ʹ���ĸ������
	ActivationFunction()
	{

	}

	//����Ϊ����������䵼��

};

//�����
class Filter
{

};

//�����
class ConvLayer
{
private:
	int width;
	int height;
	int fliterNum;
	Filter* filter;

public:
	//���캯��
	ConvLayer()
	{

	}
};

//�ػ���
class PoolLayer
{
private:

public:
	//���캯��
	PoolLayer()
	{

	}
};

//����������
class Vgg16
{
public:
	//���캯�����ڴ����
	Vgg16()
	{
		
	}



	//����Ϊѵ���������ز���
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