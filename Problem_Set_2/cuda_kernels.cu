#include "cuda_kernels.h"
#include <iostream>
#include <math.h>
#include <algorithm>
#include <SFML/Graphics.hpp>

#include "Macros.h"
#include "device_functions.hpp"

using namespace sf;

cudaDeviceProp g_CudaDeviceProp;


void CUDADeviceQuery()
{
	printf(" **CUDA Device Query (Runtime API) version (CUDART static linking)\n\n");

	int deviceCount = 0;
	cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

	if (error_id != cudaSuccess)
	{
		printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
		printf("Result = FAIL\n");
		exit(EXIT_FAILURE);
	}

	// This function call returns 0 if there are no CUDA capable devices.
	if (deviceCount == 0)
	{
		printf("There are no available device(s) that support CUDA\n");
	}
	else
	{
		printf("Detected %d CUDA Capable device(s)\n", deviceCount);
	}

	int dev, driverVersion = 0, runtimeVersion = 0;

	for (dev = 0; dev < deviceCount; ++dev)
	{
		cudaSetDevice(dev);
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, dev);

		printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);

		// Console log
		cudaDriverGetVersion(&driverVersion);
		cudaRuntimeGetVersion(&runtimeVersion);
		printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n", driverVersion / 1000, (driverVersion % 100) / 10, runtimeVersion / 1000, (runtimeVersion % 100) / 10);
		printf("  CUDA Capability Major/Minor version number:    %d.%d\n", deviceProp.major, deviceProp.minor);

		printf("  GPU Max Clock rate:                            %.0f MHz (%0.2f GHz)\n", deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);


#if CUDART_VERSION >= 5000
		// This is supported in CUDA 5.0 (runtime API device properties)
		printf("  Memory Clock rate:                             %.0f Mhz\n", deviceProp.memoryClockRate * 1e-3f);
		printf("  Memory Bus Width:                              %d-bit\n", deviceProp.memoryBusWidth);

		if (deviceProp.l2CacheSize)
		{
			printf("  L2 Cache Size:                                 %d bytes\n", deviceProp.l2CacheSize);
		}

#else
		// This only available in CUDA 4.0-4.2 (but these were only exposed in the CUDA Driver API)
		int memoryClock;
		getCudaAttribute<int>(&memoryClock, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, dev);
		printf("  Memory Clock rate:                             %.0f Mhz\n", memoryClock * 1e-3f);
		int memBusWidth;
		getCudaAttribute<int>(&memBusWidth, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, dev);
		printf("  Memory Bus Width:                              %d-bit\n", memBusWidth);
		int L2CacheSize;
		getCudaAttribute<int>(&L2CacheSize, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, dev);

		if (L2CacheSize)
		{
			printf("  L2 Cache Size:                                 %d bytes\n", L2CacheSize);
		}

#endif

		printf("  Maximum Texture Dimension Size (x,y,z)         1D=(%d), 2D=(%d, %d), 3D=(%d, %d, %d)\n",
			deviceProp.maxTexture1D, deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1],
			deviceProp.maxTexture3D[0], deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);
		printf("  Maximum Layered 1D Texture Size, (num) layers  1D=(%d), %d layers\n",
			deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1]);
		printf("  Maximum Layered 2D Texture Size, (num) layers  2D=(%d, %d), %d layers\n",
			deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1], deviceProp.maxTexture2DLayered[2]);


		printf("  Total amount of constant memory:               %lu bytes\n", deviceProp.totalConstMem);
		printf("  Total amount of shared memory per block:       %lu bytes\n", deviceProp.sharedMemPerBlock);
		printf("  Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
		printf("  Warp size:                                     %d\n", deviceProp.warpSize);
		printf("  Maximum number of threads per multiprocessor:  %d\n", deviceProp.maxThreadsPerMultiProcessor);
		printf("  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);
		printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
			deviceProp.maxThreadsDim[0],
			deviceProp.maxThreadsDim[1],
			deviceProp.maxThreadsDim[2]);
		printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
			deviceProp.maxGridSize[0],
			deviceProp.maxGridSize[1],
			deviceProp.maxGridSize[2]);
		printf("  Maximum memory pitch:                          %lu bytes\n", deviceProp.memPitch);
		printf("  Texture alignment:                             %lu bytes\n", deviceProp.textureAlignment);
		printf("  Concurrent copy and kernel execution:          %s with %d copy engine(s)\n", (deviceProp.deviceOverlap ? "Yes" : "No"), deviceProp.asyncEngineCount);
		printf("  Run time limit on kernels:                     %s\n", deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
		printf("  Integrated GPU sharing Host Memory:            %s\n", deviceProp.integrated ? "Yes" : "No");
		printf("  Support host page-locked memory mapping:       %s\n", deviceProp.canMapHostMemory ? "Yes" : "No");
		printf("  Alignment requirement for Surfaces:            %s\n", deviceProp.surfaceAlignment ? "Yes" : "No");
		printf("  Device has ECC support:                        %s\n", deviceProp.ECCEnabled ? "Enabled" : "Disabled");
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
		printf("  CUDA Device Driver Mode (TCC or WDDM):         %s\n", deviceProp.tccDriver ? "TCC (Tesla Compute Cluster Driver)" : "WDDM (Windows Display Driver Model)");
#endif
		printf("  Device supports Unified Addressing (UVA):      %s\n", deviceProp.unifiedAddressing ? "Yes" : "No");
		printf("  Device PCI Domain ID / Bus ID / location ID:   %d / %d / %d\n", deviceProp.pciDomainID, deviceProp.pciBusID, deviceProp.pciDeviceID);

		const char *sComputeMode[] =
		{
			"Default (multiple host threads can use ::cudaSetDevice() with device simultaneously)",
			"Exclusive (only one host thread in one process is able to use ::cudaSetDevice() with this device)",
			"Prohibited (no host thread can use ::cudaSetDevice() with this device)",
			"Exclusive Process (many threads in one process is able to use ::cudaSetDevice() with this device)",
			"Unknown",
			NULL
		};
		printf("  Compute Mode:\n");
		printf("     < %s >\n", sComputeMode[deviceProp.computeMode]);
	}
}

void CUDAInitDevice()
{
	int CudaDevice = 0;
	cudaSetDevice(CudaDevice);
	cudaGetDeviceProperties(&g_CudaDeviceProp, CudaDevice);
}

inline const char * GetCUDAError()
{
	cudaError err = cudaGetLastError();
	return cudaGetErrorString( err );
}

void CreateAndSetDeviceData(Uint8 *d_Data, const int SizeOfData)
{
	if ( cudaSuccess != cudaMalloc((void **)&d_Data, SizeOfData) )
    	printf( "Error in cudaMalloc. %s!\n", GetCUDAError() );
	if (cudaSuccess != cudaMemset(d_Data, 13, SizeOfData))
		printf("Error in cudaMemset. %s!\n", GetCUDAError());
}

void HostDeviceCopyOperation(void * h_Data, void * d_Data, int SizeOfData, const eHostDeviceCopyOperation operation)
{
	if (operation == eHostDeviceCopyOperation::HostToDevice)
	{
		if ( cudaSuccess != cudaMemcpy(d_Data, h_Data, SizeOfData, cudaMemcpyHostToDevice) )
			printf( "Error in cudaMemcpy host to device. %s!\n", GetCUDAError() );
	}
	else if (operation == eHostDeviceCopyOperation::DeviceToHost)
	{
		if ( cudaSuccess != cudaMemcpy(h_Data, d_Data, SizeOfData, cudaMemcpyDeviceToHost) )
			printf( "Error in cudaMemcpy device to host. %s!\n", GetCUDAError() );
	}
}

//===================================================================================================
__global__ void kernel_FillPixels(uchar4 * Pixels, const int ImgWidth, const int ImgHeight)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;

    if(i >= ImgWidth || j >= ImgHeight)
        return;

	int Idx = j * ImgWidth + i;
	
	Pixels[Idx].x = 255;
	Pixels[Idx].y = 0;
	Pixels[Idx].z = 255;
	Pixels[Idx].w = 255;
}

void GetGridDimAndBlockDim(dim3& GridDim, dim3& BlockDim, const int ImgWidth, const int ImgHeight)
{
	int BlockSize = static_cast<int>(sqrt(g_CudaDeviceProp.maxThreadsPerBlock));

	int NumBlocksX = ceil(ImgWidth / static_cast<Real>(BlockSize));
	int NumBlocksY = ceil(ImgHeight / static_cast<Real>(BlockSize));
	//We need at least 1 block
	NumBlocksX = (NumBlocksX == 0) ? 1 : NumBlocksX;
	NumBlocksY = (NumBlocksY == 0) ? 1 : NumBlocksY;

	GridDim = dim3(NumBlocksX, NumBlocksY, 1);
	BlockDim = dim3(BlockSize, BlockSize, 1);
}
void CUDAFillPixels(uchar4 *d_Pixels, const int ImgWidth, const int ImgHeight)
{
	dim3 GridDim, BlockDim;
	GetGridDimAndBlockDim(GridDim, BlockDim, ImgWidth, ImgHeight);

	kernel_FillPixels << < GridDim, BlockDim >> >(d_Pixels, ImgWidth, ImgHeight);

	if ( cudaSuccess != cudaGetLastError() )
    	printf( "Error in kernel_FillPixels! %s\n", GetCUDAError() );

	cudaDeviceSynchronize();
}

//===================================================================================================
//This kernel takes in an image represented as a uchar4 and splits
//it into three images consisting of only one color channel each
__global__
void separateChannels(const uchar4* inputImageRGBA,
	int numRows,
	int numCols,
	unsigned char* redChannel,
	unsigned char* greenChannel,
	unsigned char* blueChannel)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i >= numCols || j >= numRows)
		return;

	const int Idx = i + j * numCols;
	const uchar4 rgba = inputImageRGBA[Idx];

	redChannel[Idx]		= rgba.x;
	greenChannel[Idx]	= rgba.y;
	blueChannel[Idx]	= rgba.z;
}

//===================================================================================================
//This kernel takes in three color channels and recombines them
//into one image.  The alpha channel is set to 255 to represent
//that this image has no transparency.
__global__
void recombineChannels(const unsigned char* const redChannel,
	const unsigned char* const greenChannel,
	const unsigned char* const blueChannel,
	uchar4* const outputImageRGBA,
	int numRows,
	int numCols)
{
	const int2 thread_2D_pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
		blockIdx.y * blockDim.y + threadIdx.y);

	const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

	//make sure we don't try and access memory outside the image
	//by having any threads mapped there return early
	if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
		return;

	unsigned char red = redChannel[thread_1D_pos];
	unsigned char green = greenChannel[thread_1D_pos];
	unsigned char blue = blueChannel[thread_1D_pos];

	//Alpha should be 255 for no transparency
	uchar4 outputPixel = make_uchar4(red, green, blue, 255);

	outputImageRGBA[thread_1D_pos] = outputPixel;
}

//===================================================================================================
__global__
void gaussian_blur(
	const unsigned char* const inputChannel,
	unsigned char* const outputChannel,
	int numRows, int numCols,
	const float* const filter, const int filterWidth)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i >= numCols || j >= numRows)
		return;

	const int Idx = j * numCols + i;
	float Result = 0;
	
	for (int x = 0, int s = -filterWidth / 2; s <= filterWidth / 2; ++x, ++s)
	{
		for (int y = 0, int t = -filterWidth / 2; t <= filterWidth / 2; ++y, ++t)
		{
			int n_i = max(0, min(numCols-1, i + s));
			int n_j = max(0, min(numRows-1, j + t));
			
			const int NeighBourIdx = n_j * numCols + n_i;
			const int FilterIdx = y * filterWidth + x;
			Result += inputChannel[NeighBourIdx] * filter[FilterIdx];
		}
	}

	outputChannel[Idx] = Result;
}

void SetFilter(float *h_filter, const int blurKernelWidth)
{
	const float blurKernelSigma = 2.;

	// fill the filter we will convolve with

	float filterSum = 0.f; //for normalization

	for (int r = -blurKernelWidth / 2; r <= blurKernelWidth / 2; ++r) {
		for (int c = -blurKernelWidth / 2; c <= blurKernelWidth / 2; ++c) {
			float filterValue = expf(-(float)(c * c + r * r) / (2.f * blurKernelSigma * blurKernelSigma));
			(h_filter)[(r + blurKernelWidth / 2) * blurKernelWidth + c + blurKernelWidth / 2] = filterValue;
			filterSum += filterValue;
		}
	}

	float normalizationFactor = 1.f / filterSum;

	for (int r = -blurKernelWidth / 2; r <= blurKernelWidth / 2; ++r) {
		for (int c = -blurKernelWidth / 2; c <= blurKernelWidth / 2; ++c) {
			(h_filter)[(r + blurKernelWidth / 2) * blurKernelWidth + c + blurKernelWidth / 2] *= normalizationFactor;
		}
	}
}

void GaussianBlur(uchar4* d_ImageRGBA, int ImageWidht, int ImageHeight)
{
	Uint64 ChannelBufferSize = ImageWidht * ImageHeight * sizeof(Uint8);
	Uint8* d_redChannelIn;		cudaMalloc((void **)&d_redChannelIn, ChannelBufferSize);
	Uint8* d_redChannelOut;	cudaMalloc((void **)&d_redChannelOut, ChannelBufferSize);

	Uint8* d_greenChannelIn;	cudaMalloc((void **)&d_greenChannelIn, ChannelBufferSize);
	Uint8* d_greenChannelOut;	cudaMalloc((void **)&d_greenChannelOut, ChannelBufferSize);

	Uint8* d_blueChannelIn;		cudaMalloc((void **)&d_blueChannelIn, ChannelBufferSize);
	Uint8* d_blueChannelOut;	cudaMalloc((void **)&d_blueChannelOut, ChannelBufferSize);

	dim3 GridDim, BlockDim;
	GetGridDimAndBlockDim(GridDim, BlockDim, ImageWidht, ImageHeight);

	separateChannels << < GridDim, BlockDim >> >(d_ImageRGBA, ImageHeight, ImageWidht, d_redChannelIn, d_greenChannelIn, d_blueChannelIn);
	if (cudaSuccess != cudaGetLastError()) printf("Error in kernel separateChannels! Error: %s\n", GetCUDAError());
	cudaDeviceSynchronize();
	
	//now create the filter that they will use
	const int blurKernelWidth = 9;
	float *h_filter = new float[blurKernelWidth * blurKernelWidth];
	SetFilter(h_filter, blurKernelWidth);

	float *d_filter;
	cudaMalloc((void**)&d_filter, blurKernelWidth * blurKernelWidth * sizeof(float));
	HostDeviceCopyOperation(h_filter, d_filter, blurKernelWidth * blurKernelWidth * sizeof(float), eHostDeviceCopyOperation::HostToDevice);

	gaussian_blur << < GridDim, BlockDim >> >(d_redChannelIn, d_redChannelOut, ImageHeight, ImageWidht, d_filter, blurKernelWidth);
	if (cudaSuccess != cudaGetLastError()) printf("Error in kernel gaussian_blur! Error: %s\n", GetCUDAError());
	cudaDeviceSynchronize();

	gaussian_blur << < GridDim, BlockDim >> >(d_greenChannelIn, d_greenChannelOut, ImageHeight, ImageWidht, d_filter, blurKernelWidth);
	if (cudaSuccess != cudaGetLastError()) printf("Error in kernel gaussian_blur! Error: %s\n", GetCUDAError());
	cudaDeviceSynchronize();

	gaussian_blur << < GridDim, BlockDim >> >(d_blueChannelIn, d_blueChannelOut, ImageHeight, ImageWidht, d_filter, blurKernelWidth);
	if (cudaSuccess != cudaGetLastError()) printf("Error in kernel gaussian_blur! Error: %s\n", GetCUDAError());
	cudaDeviceSynchronize();

	recombineChannels << < GridDim, BlockDim >> >(d_redChannelOut, d_greenChannelOut, d_blueChannelOut, d_ImageRGBA, ImageHeight, ImageWidht);
	//recombineChannels << < GridDim, BlockDim >> >(d_redChannelIn, d_greenChannelIn, d_blueChannelIn, d_ImageRGBA, ImageHeight, ImageWidht);
	
	if (cudaSuccess != cudaGetLastError()) printf("Error in kernel recombineChannels! Error: %s\n", GetCUDAError());
	cudaDeviceSynchronize();

	cudaFree(d_filter);
	cudaFree(d_redChannelIn);
	cudaFree(d_redChannelOut);
	cudaFree(d_greenChannelIn);
	cudaFree(d_greenChannelOut);
	cudaFree(d_blueChannelIn);
	cudaFree(d_blueChannelOut);
}