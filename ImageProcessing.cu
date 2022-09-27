#include "ImageProcessing.cuh"

__global__ void ThresholdGPU(
	const uchar * ImgPtr,
	int Thresh,
	int MaxSize,
	uchar * BinPtr
)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < MaxSize)
	{
		if (ImgPtr[index] >= Thresh)
		{
			BinPtr[index] = 255;
		}
		else
		{
			BinPtr[index] = 0;
		}
	}
}

int Threshold(
	const cv::Mat & Img,
	int Thresh,
	cv::OutputArray & Bin
)
{
	uchar *ImgCuda, *BinCuda;
	int ImgSize = Img.rows * Img.cols * sizeof(uchar);
	cudaMalloc((void **)&ImgCuda, ImgSize);
	cudaMalloc((void **)&BinCuda, ImgSize);
	cudaMemcpy(ImgCuda, Img.data, ImgSize, cudaMemcpyHostToDevice);
	int ThreadNum = 512;
	ThresholdGPU <<<(Img.rows * Img.cols - 1)/ThreadNum + 1, ThreadNum >>> (
		ImgCuda, 
		Thresh,
		Img.rows * Img.cols, 
		BinCuda
	);
	Bin.create(Img.rows, Img.cols, CV_8UC1);
	cv::Mat BinMat = Bin.getMat();

	cudaMemcpy(BinMat.data, BinCuda, ImgSize, cudaMemcpyDeviceToHost);
	cudaFree(ImgCuda);
	cudaFree(BinCuda);

	return 0;
}

__global__ void FindHorizontalEdgeGPU(
	const uchar * ImgPtr,
	const int GradThresh,
	const int SliceSize,
	const int MaxSize,
	int * EdgeLocationX
)
{
	uchar MaxGrad = -255;
	int MaxIdx = -1;
	int index = (blockIdx.x * blockDim.x + threadIdx.x) * SliceSize;
	for (int idx = 1; idx < SliceSize; idx++)
	{
		if (index + idx < MaxSize)
		{
			uchar GradVal = std::abs(ImgPtr[index + idx] - ImgPtr[index + idx - 1]);
			if (GradVal >= GradThresh)
			{
				if (GradVal > MaxGrad)
				{
					MaxGrad = GradVal;
					MaxIdx = idx;
				}
			}
		}
	}
	//printf("%d %d %d\n", blockIdx.x, MaxGrad, MaxIdx);
	EdgeLocationX[blockIdx.x * blockDim.x + threadIdx.x] = MaxIdx;
}

int FindHorizontalEdge(
	const cv::Mat & Img,
	const cv::Rect & ROI,
	const int GradThresh,
	std::vector<cv::Point> & EdgeLocations
)
{
	cv::Mat EdgeImg = Img(ROI).clone();
	uchar * EdgeImgCuda;
	int Size = EdgeImg.rows * EdgeImg.cols * sizeof(uchar);
	cudaMalloc((void **)&EdgeImgCuda, Size);
	cudaMemcpy(EdgeImgCuda, EdgeImg.data, Size, cudaMemcpyHostToDevice);
	int * EdgeLocationXCuda;
	cudaMalloc((void **)&EdgeLocationXCuda, EdgeImg.rows * sizeof(int));
	int ThreadNum = 32;
	FindHorizontalEdgeGPU <<<EdgeImg.rows / ThreadNum + 1, ThreadNum >>> (
		EdgeImgCuda,
		GradThresh,
		EdgeImg.cols,
		EdgeImg.rows * EdgeImg.cols,
		EdgeLocationXCuda
	);
	int * EdgeLocationX;
	EdgeLocationX = (int *)malloc(EdgeImg.rows * sizeof(int));
	cudaMemcpy(EdgeLocationX, EdgeLocationXCuda, EdgeImg.rows * sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(EdgeImgCuda);
	cudaFree(EdgeLocationX);
	for (size_t idx = 0; idx < EdgeImg.rows; idx++)
	{
		cv::Point pt;
		pt.x = EdgeLocationX[idx];
		pt.y = idx;
		EdgeLocations.push_back(pt);
	}

	free(EdgeLocationX);

	return 0;
}

__global__ void ImageSubSample(
	const uchar * ImgPtr,
	const int ImgWidth,
	const int ImgHeight,
	const int KernelSize,
	const int SubImgWidth,
	const int SubImgHeight,
	uchar * SubImgPtr
)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	if (index >= SubImgWidth * SubImgHeight)
		return;
	
	int PixelIdx = (int)(index / SubImgWidth) * ImgWidth * KernelSize + (int)(index % SubImgWidth) * KernelSize;
	int PixelColIdx = PixelIdx % ImgWidth;
	int PixelRowIdx = PixelIdx / ImgWidth;

	int GrayValSum = 0;
	int Cnt = 0;
	
	for (int RowIdx = 0; RowIdx < KernelSize; RowIdx++)
	{
		for (int ColIdx = 0; ColIdx < KernelSize; ColIdx++)
		{
			if ((PixelColIdx + ColIdx < ImgWidth) && (PixelRowIdx + RowIdx < ImgHeight))
			{
				int Idx = (PixelRowIdx + RowIdx) * ImgWidth + (PixelColIdx + ColIdx);
				GrayValSum += ImgPtr[Idx];
				Cnt++;
			}
		}
	}

	int MeanGrayVal = GrayValSum / Cnt;
	SubImgPtr[index] = MeanGrayVal;
}

int ImageSubSample(
	const cv::InputArray & Img,
	const int KernelSize,
	cv::OutputArray & SubImg
)
{
	cv::Mat ImgMat = Img.getMat();
	uchar * ImgCuda;
	int Size = ImgMat.rows * ImgMat.cols * sizeof(uchar);
	cudaMalloc((void **)&ImgCuda, Size);

	std::clock_t c_start = std::clock();
	auto t_start = std::chrono::high_resolution_clock::now();

	cudaMemcpy(ImgCuda, ImgMat.data, Size, cudaMemcpyHostToDevice);

	uchar * SubImgCuda;
	int SubImgWidth = (int)(ImgMat.cols / KernelSize);
	int NotCompleteBlockWidth = SubImgWidth % KernelSize;
	if (NotCompleteBlockWidth != 0)
	{
		SubImgWidth++;
	}
	int SubImgHeight = (int)(ImgMat.rows / KernelSize);
	int NotCompleteBlockHeight = SubImgHeight % KernelSize;
	if (NotCompleteBlockHeight != 0)
	{
		SubImgHeight++;
	}
	int SubImgSize = SubImgWidth * SubImgHeight * sizeof(uchar);
	cudaMalloc((void **)&SubImgCuda, SubImgSize);

	int ThreadSize = 1024;
	ImageSubSample << <(SubImgWidth * SubImgHeight) / ThreadSize + 1, ThreadSize >> > (
		ImgCuda, 
		ImgMat.cols,
		ImgMat.rows,
		KernelSize, 
		SubImgWidth, 
		SubImgHeight, 
		SubImgCuda
	);

	SubImg.create(SubImgHeight, SubImgWidth, CV_8UC1);
	cv::Mat SubImgMat = SubImg.getMat();
	cudaMemcpy(SubImgMat.data, SubImgCuda, SubImgSize, cudaMemcpyDeviceToHost);

	std::clock_t c_end = std::clock();
	auto t_end = std::chrono::high_resolution_clock::now();
	std::cout << std::fixed << std::setprecision(2) << "CPU time used: "
		<< 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC << " ms\n"
		<< "Wall clock time passed: "
		<< std::chrono::duration<double, std::milli>(t_end - t_start).count()
		<< " ms\n";

	cudaFree(ImgCuda);
	cudaFree(SubImgCuda);

	return 0;
}

__global__ void ImageSobel(
	const uchar * ImgPtr,
	const int ImgWidth,
	const int ImgHeight,
	uchar * SubImgPtr
)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	//printf("%d %d %d \n", index, ImgWidth, ImgHeight);

	if (index > ImgWidth * ImgHeight)
	{
		return;
	}

	int ColIdx = index % ImgWidth;
	int RowIdx = index / ImgWidth;

	float value = 0;
	if ((ColIdx - 1 >= 0) && (ColIdx - 1 < ImgWidth) && (RowIdx - 1 >= 0) && (RowIdx - 1 < ImgHeight))
	{
		value += (-1 + (-1)) * ImgPtr[(RowIdx - 1) * ImgWidth + ColIdx - 1];
	}
	if ((RowIdx - 1 >= 0) && (RowIdx - 1 < ImgHeight))
	{
		value += -2 * ImgPtr[(RowIdx - 1) * ImgWidth + ColIdx];
	}
	if ((ColIdx + 1 >= 0) && (ColIdx + 1 < ImgWidth) && (RowIdx - 1 >= 0) && (RowIdx - 1 < ImgHeight))
	{
		value += (-1 + 1) * ImgPtr[(RowIdx - 1) * ImgWidth + ColIdx + 1];
	}
	if ((ColIdx - 1 >= 0) && (ColIdx - 1 < ImgHeight))
	{
		value += -2 * ImgPtr[RowIdx * ImgWidth + ColIdx - 1];
	}
	if ((ColIdx + 1 >= 0) && (ColIdx + 1 < ImgHeight))
	{
		value += 2 * ImgPtr[RowIdx * ImgWidth + ColIdx + 1];
	}
	if ((ColIdx - 1 >= 0) && (ColIdx - 1 < ImgWidth) && (RowIdx + 1 >= 0) && (RowIdx + 1 < ImgHeight))
	{
		value += (1 + (-1)) * ImgPtr[(RowIdx + 1) * ImgWidth + ColIdx - 1];
	}
	if ((RowIdx + 1 >= 0) && (RowIdx + 1 < ImgHeight))
	{
		value += 2 * ImgPtr[(RowIdx + 1) * ImgWidth + ColIdx];
	}
	if ((ColIdx + 1 >= 0) && (ColIdx + 1 < ImgWidth) && (RowIdx + 1 >= 0) && (RowIdx + 1 < ImgHeight))
	{
		value += (1 + 1) * ImgPtr[(RowIdx + 1) * ImgWidth + ColIdx + 1];
	}

	SubImgPtr[index] = abs(value);
}

int ImageSobel(
	const cv::InputArray & Img,
	cv::OutputArray & ImgSobel
)
{
	cv::Mat ImgMat = Img.getMat();
	ImgSobel.create(ImgMat.rows, ImgMat.cols, CV_8UC1);
	cv::Mat ImgSobelMat = ImgSobel.getMat();
	int InputImgSize = ImgMat.rows * ImgMat.cols * sizeof(uchar);
	uchar * ImgCuda;
	cudaMalloc((void **)&ImgCuda, InputImgSize);

	std::clock_t c_start = std::clock();
	auto t_start = std::chrono::high_resolution_clock::now();

	cudaMemcpy(ImgCuda, ImgMat.data, InputImgSize, cudaMemcpyHostToDevice);
	uchar * ImgSobelCuda;
	int OutputImgSize = ImgMat.rows * ImgMat.cols * sizeof(uchar);
	cudaMalloc((void **)&ImgSobelCuda, OutputImgSize);
	int ThreadNum = 512;
	ImageSobel << <(ImgMat.rows * ImgMat.cols) / ThreadNum + 1, ThreadNum >> > (
		ImgCuda,
		ImgMat.cols,
		ImgMat.rows,
		ImgSobelCuda
		);
	cudaMemcpy(ImgSobelMat.data, ImgSobelCuda, OutputImgSize, cudaMemcpyDeviceToHost);

	std::clock_t c_end = std::clock();
	auto t_end = std::chrono::high_resolution_clock::now();
	std::cout << std::fixed << std::setprecision(2) << "CPU time used: "
		<< 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC << " ms\n"
		<< "Wall clock time passed: "
		<< std::chrono::duration<double, std::milli>(t_end - t_start).count()
		<< " ms\n";

	cudaFree(ImgCuda);
	cudaFree(ImgSobelCuda);

	return 0;
}
