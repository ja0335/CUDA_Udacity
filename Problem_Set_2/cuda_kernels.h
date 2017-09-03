#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

#include <SFML/Graphics.hpp>

enum class eHostDeviceCopyOperation
{
    HostToDevice,
    DeviceToHost
};

void CUDADeviceQuery();

void CUDAInitDevice();

void CreateAndSetDeviceData(sf::Uint8 *d_Data, const size_t SizeOfData);

void HostDeviceCopyOperation(void * h_Data, void * d_Data, const size_t SizeOfData, const eHostDeviceCopyOperation operation);

void DeviceFreeData(void *h_Data);

void CUDAFillPixels(sf::Uint8 *d_Pixels, const size_t ImgWidth, const size_t ImgHeight);

#endif // CUDA_KERNELS_H
