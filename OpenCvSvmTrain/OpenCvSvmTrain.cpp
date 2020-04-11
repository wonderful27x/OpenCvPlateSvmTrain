// OpenCvSvmTrain.cpp: 定义应用程序的入口点。
//

#include "OpenCvSvmTrain.h"

int main()
{
	train();
	return 0;
}


/**--------------SVM(支持向量机)分类器训练与测试--------------------
 *  SVM:通常用来进行模式识别、分类以及回归分析
 *  使用SVM对于查找到的车牌号区域进行判断,进一步确定定位车牌号区域正确
 *
 *  流程:
 *   1、加载全部样本(训练数据) 训练数据有正确的,有错误的。也就是分为车牌区域与非车牌区域
 *   2、创建样本数据标签,标记对应样本分类(正确/错误)
 *   3、训练、保存
 *
 *  支持向量机（support vector machines, SVM）是一种分类模型，
 *  它的基本模型是定义在特征空间上的间隔最大的线性分类器，间隔最大使它有别于感知机；
 *  SVM还包括核技巧，这使它成为实质上的非线性分类器。SVM的的学习策略就是间隔最大化，
 *  可形式化为一个求解凸二次规划的问题，也等价于正则化的合页损失函数的最小化问题。
 *  SVM的的学习算法就是求解凸二次规划的最优化算法。

 *	然而很多情况下问题是线性不可分的，这时就需要使用核函数，
 *	核函数的目的就是把低维向量映射到高维，从而寻找出一个分类函数（超平面）
 *  https://zhuanlan.zhihu.com/p/31886934


*  深度学习就是调整参数的过程 ... ...
*  这些参数肯定不是最好的,但是基本稳定.
*  不同参数效果不同、训练检测速度不同,同时也影响着模型文件的大小.
*  大致参数调整可以从下面的地址 svm_test.png
*  http://www.csie.ntu.edu.tw/~cjlin/libsvm/

 */
void train()
{
	cout << "车牌识别svm模型训练" << endl;
	cout << "训练数据加载中..." << endl;
	//创建SVM
	Ptr<SVM> classifier = SVM::create();
	//正负车牌集合，路径/标签
	vector<TrainStruct> plates;
	//加载正样本,并打上标签1
	vector<string> posFiles;
	getFiles(SVM_POS, posFiles);
	for each (string posFile in posFiles)
	{
		plates.push_back({ posFile,1 });
	}
	//加载负样本，并打上标签-1
	vector<string> negFiles;
	getFiles(SVM_NEG, negFiles);
	for (string negFile:negFiles) {
		plates.push_back({negFile,-1});
	}
	//提取特征并保存对应的标签
	vector<int> responses;
	Mat samples;
	for (TrainStruct plate : plates) {
		//加载图片
		Mat image = imread(plate.fileName, IMREAD_GRAYSCALE);
		if (image.empty()) {
			printf("加载样本失败，image: %s \n", plate.fileName.c_str());
			continue;
		}
		//二值化
		threshold(image, image, 0, 255, THRESH_BINARY + THRESH_OTSU);
		//提取hog特征
		Mat features;
		getSvmHogFeatures(image, features);
		//归一化为一通道一行
		features = features.reshape(1, 1);
		//Mat支持每一行插入（Mat::push_back），这一点与vector类似，
		//将一行Mat插入到另外 一个Mat我们只需要保证两者的cols是相同的。
		//于是循环结束后samples就变成了m行n列的矩阵，m就等于样本的数量，n取决于提取的hog特征，
		//所以samples的每一行就代表了一个样本的特征，而responses对应的索引位置处的值则代表了该样本的标签。
		samples.push_back(features);
		responses.push_back(plate.label);
	}
	cout << "samples rows: " << samples.rows << " - samples cols: " << samples.cols << endl; if (true)return;
	//练习数据，必须是CV_32FC1 （32位浮点类型，单通道）。数据必须是CV_ROW_SAMPLE的，即特点向量以行来存储
	samples.convertTo(samples, CV_32FC1);
	//创建训练数据
	Ptr<TrainData> trainData = TrainData::create(samples,SampleTypes::ROW_SAMPLE,responses);
	//内核函数，默认就是径向基函数（高斯核函数） 对于大多数情况都是一个较好的选择
	classifier->setKernel(SVM::RBF);
	//内核函数的参数 默认就是1  rbf只会需要这一个参数
	classifier->setGamma(1);
	//svm的类型 默认就是这个 允许用异常值惩罚因子（C）进行不完全分类
	classifier->setType(SVM::C_SVC);
	//C越大错误率较小，间隔也较小 间隔指 对应样本离同类区域的距离
	classifier->setC(100);
	//迭代算法的终止条件 迭达到20000次终止  
	//SVM训练的过程就是一个通过 迭代 解决约束条件下的二次优化问题，允许算法在适当的条件下停止计算
	//p1->COUNT：终止条件为最大迭代次数 p2->20000：迭代20000次 p3->FLT_EPSILON误差FLT_EPSILON（当type=COUNT时这个参数无效）
	classifier->setTermCriteria(TermCriteria(TermCriteria::COUNT, 20000, FLT_EPSILON));
	
	cout << "训练数据加载完成，训练中..." << endl;
	ULONGLONG timeStart = GetTickCount64();
	//trainAuto,使用了K折交叉验证来优化参数，会自动寻找最优参数
	//最后一个参数设置true 则会创建更平衡的验证子集 
	//也就是如果是2类分类的话能得到更准确的结果
	//https://www.cnblogs.com/hust-yingjie/p/6582218.html
	//https://blog.csdn.net/rj1457365980/article/details/83053890
	classifier->trainAuto(
		trainData,                         //训练数据集
		10,                                //交叉验证次数，默认值为10
		SVM::getDefaultGrid(SVM::C),       //参数C，默认值为0.1~500，步进 5
		SVM::getDefaultGrid(SVM::GAMMA),   //参数GAMMA，默认值为1e-5~0.6，步进 15
		SVM::getDefaultGrid(SVM::P),       //参数P，默认值为0.01~100，步进 7
		SVM::getDefaultGrid(SVM::NU),      //参数NU，默认值为0.01~0.2，步进 3
		SVM::getDefaultGrid(SVM::COEF),    //参数COEF，默认值为0.1~300，步进 14
		SVM::getDefaultGrid(SVM::DEGREE),  //参数DEGREE，默认值为0.01~4  步进 7
		true
	);
	ULONGLONG timeEnd = GetTickCount64();
	//保存结果
	classifier->save(SVM_XML);
	printf("训练完成，用时：%I64us\n", (timeEnd - timeStart) / 1000);
}

//提取HOG特征
//HOG特征：方向梯度直方图，使用于轮廓提取
void getSvmHogFeatures(const Mat image, Mat& features)
{
	static int count = -1;
	//创建HOG特征
	HOGDescriptor hog(Size(128,64),Size(16,16),Size(8,8),Size(8,8),3);
	vector<float> descriptor;
	Mat trainImg = Mat(hog.winSize,CV_32S);
	//归一化
	resize(image, trainImg, hog.winSize);
	//计算hog特征
	hog.compute(trainImg, descriptor, Size(8, 8));
	//转成mat
	Mat featureMat = Mat(descriptor);
	featureMat.copyTo(features);
	if (count == 1) {
		cout << "winW: " << hog.winSize.width << "	winH: " << hog.winSize.height << endl;
		count = -1;
	}
	
}
