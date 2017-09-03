#if defined(__unix__) || defined(__unix)
#include <string.h>
#endif
#include <SFML/Graphics.hpp>
#include <iostream>

#include "Macros.h"
#include "cuda_kernels.h"

#include "cuda.h"
#include "cuda_runtime.h"

using namespace sf;

int main(int argc, char **argv)
{
    RenderWindow window(VideoMode(1024, 768), "Navier Stokes");
	Texture DynamicTexture;
	Sprite SpriteDynamicTexture;

	if (!DynamicTexture.create(PLOT_RESOLUTION, PLOT_RESOLUTION))
		return EXIT_FAILURE;

	SpriteDynamicTexture.setTexture(DynamicTexture);
	DynamicTexture.setSmooth(false);

	// Init Device
    //CUDADeviceQuery();
	CUDAInitDevice();

	// Host data buffers
	Uint64 PixelsBufferSize = PLOT_RESOLUTION * PLOT_RESOLUTION * 4 * sizeof(Uint8);
	Uint8* Pixels = new Uint8[PixelsBufferSize];	
	memset(Pixels, 10, PixelsBufferSize);

	// Device data buffers
	Uint8* d_Pixels; 
	//CreateAndSetDeviceData(d_Pixels, PixelsBufferSize);
	cudaMalloc((void **)&d_Pixels, PixelsBufferSize);
	cudaMemset(d_Pixels, 13, PixelsBufferSize);

	// Copy host to device
	HostDeviceCopyOperation(Pixels, d_Pixels, PixelsBufferSize, eHostDeviceCopyOperation::HostToDevice);

	CUDAFillPixels(d_Pixels, PLOT_RESOLUTION, PLOT_RESOLUTION);

	// Copy device to host
	std::cout << "Value: " << static_cast<size_t>(Pixels[0]) << std::endl;
	HostDeviceCopyOperation(Pixels, d_Pixels, PixelsBufferSize, eHostDeviceCopyOperation::DeviceToHost);
	std::cout << "Value: " << static_cast<size_t>(Pixels[0]) << std::endl;

	DeviceFreeData(d_Pixels);

    while (window.isOpen())
	{
        Event event;
		while (window.pollEvent(event))
		{
			// "close requested" event: we close the window
			if (Keyboard::isKeyPressed(Keyboard::Escape) || event.type == Event::Closed)
				window.close();
        }

		DynamicTexture.update(Pixels);
		window.draw(SpriteDynamicTexture);

		// end the current frame
		window.display();
    }

	delete[] Pixels;


    return 0;
}
