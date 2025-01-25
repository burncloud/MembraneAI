
#include"OpencvDetection.h"
#include<iostream>
#include<opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include<inference_engine.hpp>
using namespace cv;


int main() {
	
	InferenceEngine::Core ie;

	Detection detectoion = Detection("E://tongkongyuan//python//export//1.onnx");
	//Detection detectoion = Detection("C://Program Files (x86)//IntelSWTools//openvino_2020.4.287//deployment_tools//model_optimizer//1.bin",
		//"C://Program Files (x86)//IntelSWTools//openvino_2020.4.287//deployment_tools//model_optimizer//1.xml");

	Mat img = Mat::ones(Size(384, 288), CV_8UC3);
	/*Mat img = imread("E://yanqiu//data//2Í«¿×Ôµ+ÑÛÄÚ³ß000016.001.jpg");
	cvtColor(img, img, COLOR_BGR2RGB);
	resize(img, img, Size(512, 288));
	img.convertTo(img, CV_32F, 1.0 / 255);
	for (int row = 0; row < img.size[0]; row++)
	{
		float* imgV = img.ptr<float>(row);
		for (int col = 0; col < img.size[1];col++) {
			imgV[col] -= 0.5;
			imgV[col] /= 0.225;
		}
	}*/


	//img = img(Range::all(), Range(64, 448));
	//
	Mat score = detectoion.forward(img);
	//std::cout << score.size << std::endl;

	////Mat segm;
	//const int rows = score.size[2];
	//const int cols = score.size[3];
	//const int chns = score.size[1];
	////segm.create(rows, cols, CV_8UC3);
	//Mat maxCl = Mat::zeros(rows, cols, CV_8UC1);
	//Mat maxVal(rows, cols, CV_32FC1, score.data);
	//for (int row = 0; row < rows; row++)
	//{
	//	float* ptrMaxVal = maxVal.ptr<float>(row);
	//	for (int col = 0; col < cols; col++) {
	//		std::cout << ptrMaxVal[col] << std::endl;
	//	}
	//}



	//for (int row = 0; row < rows; row++)
	//{	
	//	uint8_t* ptrMaxCl = maxCl.ptr<uint8_t>(row);
	//	float* ptrMaxVal = maxVal.ptr<float>(row);
	//	for (int col = 0; col < cols; col++) {
	//		if (ptrMaxVal[col] > -3)
	//		{
	//			ptrMaxCl[col] = 255;
	//		}
	//		else {
	//			ptrMaxCl[col] = 0;
	//		}
	//	}
	//}
	//Mat maxCl_up;
	//resize(maxCl, maxCl_up, Size(384, 288), 0, 0, 1);

	//imshow("da", maxCl);
	//waitKey(0);
	//std::cout << maxVal.size << std::endl;
	/*for (int ch = 1; ch < chns; ch++)
	{
		for (int row = 0; row < rows; row++)
		{
			const float* ptrScore = score.ptr<float>(0, ch, row);
			uint8_t* ptrMaxCl = maxCl.ptr<uint8_t>(row);
			float* ptrMaxVal = maxVal.ptr<float>(row);
			for (int col = 0; col < cols; col++)
			{
				if (ptrScore[col] > ptrMaxVal[col])
				{
					ptrMaxVal[col] = ptrScore[col];
					ptrMaxCl[col] = (uchar)ch;
				}
			}
		}
	}*/




	std::cout << score.size << std::endl;
	const int cols = score.size[1];
	const int chns = score.size[0];
	for (int i = 0; i < chns; i++)
	{	
		const float* ptrScore = score.ptr<float>(0,i);
		for (int j = 0; j < cols; j++)
		{
			std::cout << ptrScore[j] << std::endl;
		}
		break;
	}

	return 0;
}