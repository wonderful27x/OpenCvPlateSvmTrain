// OpenCvSvmTrain.h: 标准系统包含文件的包含文件
// 或项目特定的包含文件。

#pragma once

#include <iostream>
#include<string>
#include"utils.h"
#include<opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace ml;

#define SVM_XML "E:/VisualStudio/FILE/svm/plate/model/plateSvm.xml" //训练模型路径
#define SVM_POS "E:/VisualStudio/FILE/svm/plate/samples/posPlate"     //正车牌路径
#define SVM_NEG "E:/VisualStudio/FILE/svm/plate/samples/negPlate"     //负车牌路径

struct TrainStruct
{
	string fileName;
	int label;
};

//svm训练
void train();
//获取hog特征
void getSvmHogFeatures(const Mat image,Mat &features);

// TODO: 在此处引用程序需要的其他标头。
