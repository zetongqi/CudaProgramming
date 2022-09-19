#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "opencv2/opencv.hpp"

__global__ void ThresholdGPU(
	const uchar * ImgPtr,
	int Thresh,
	int MaxSize,
	uchar * BinPtr
);

int Threshold(
	const cv::Mat & Img,
	int Thresh,
	cv::OutputArray & Bin
);

__global__ void FindHorizontalEdgeGPU(
	const uchar * ImgPtr,
	const int GradThresh,
	const int SliceSize,
	const int MaxSize,
	int * EdgeLocationX
);

int FindHorizontalEdge(
	const cv::Mat & Img,
	const cv::Rect & ROI,
	const int GradThresh,
	std::vector<cv::Point> & EdgeLocations
);

__global__ void ImageSubSample(
	const uchar * ImgPtr,
	const int ImgWidth,
	const int ImgHeight,
	const int KernelSize,
	const int SubImgWidth,
	const int SubImgHeight,
	uchar * SubImgPtr
);

int ImageSubSample(
	const cv::InputArray & Img,
	const int KernelSize,
	cv::OutputArray & SubImg
);
