#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

#include <SFML/Graphics.hpp>

#include "cuda.h"
#include "cuda_runtime.h"

enum class eHostDeviceCopyOperation
{
    HostToDevice,
    DeviceToHost
};

void CUDADeviceQuery();

void CUDAInitDevice();

void CreateAndSetDeviceData(sf::Uint8 *d_Data, const int SizeOfData);

void HostDeviceCopyOperation(void * h_Data, void * d_Data, const int SizeOfData, const eHostDeviceCopyOperation operation);

void CUDAFillPixels(uchar4 *d_Pixels, const int ImgWidth, const int ImgHeight);

void GaussianBlur(uchar4* d_ImageRGBA, int ImageWidht, int ImageHeight);

#endif // CUDA_KERNELS_H
