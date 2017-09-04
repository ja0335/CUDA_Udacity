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

extern int g_FilterWidth = 9;

int main(int argc, char **argv)
{
	for (int i = 0 ; i < argc; ++i)
	{
		const char * Param = argv[i];

		if (strcmp("-fw", argv[i]) == 0)
		{
			g_FilterWidth = i + 1 < argc ? atoi(argv[i + 1]) : g_FilterWidth;
		}
	}

    RenderWindow window(VideoMode(1024, 768), "Navier Stokes");

	Image SourceImage;

	if (!SourceImage.loadFromFile("../img/source.png"))
		return EXIT_FAILURE;
	
	Texture DynamicTexture;
	Sprite SpriteDynamicTexture;

	Uint64 ImageWidht = SourceImage.getSize().x;
	Uint64 ImageHeight = SourceImage.getSize().y;

	if (!DynamicTexture.create(ImageWidht, ImageHeight))
		return EXIT_FAILURE;

	SpriteDynamicTexture.setTexture(DynamicTexture);
	DynamicTexture.setSmooth(false);

	// Init Device
    CUDADeviceQuery();
	CUDAInitDevice();
	
	// Host data buffers
	Uint64 PixelsBufferSize = ImageWidht * ImageHeight * sizeof(uchar4);
	uchar4* Pixels = new uchar4[PixelsBufferSize];
	memcpy(Pixels, SourceImage.getPixelsPtr(), PixelsBufferSize);
	

	// Device data buffers
	uchar4* d_Pixels; cudaMalloc((void **)&d_Pixels, PixelsBufferSize);

	// Copy host to device
	HostDeviceCopyOperation(Pixels, d_Pixels, PixelsBufferSize, eHostDeviceCopyOperation::HostToDevice);

	GaussianBlur(d_Pixels, ImageWidht, ImageHeight);

	// Copy device to host
	HostDeviceCopyOperation(Pixels, d_Pixels, PixelsBufferSize, eHostDeviceCopyOperation::DeviceToHost);

	cudaFree(d_Pixels);

    while (window.isOpen())
	{
        Event event;
		while (window.pollEvent(event))
		{
			// "close requested" event: we close the window
			if (Keyboard::isKeyPressed(Keyboard::Escape) || event.type == Event::Closed)
				window.close();
        }

		window.clear(Color::Black);
		DynamicTexture.update(reinterpret_cast<Uint8*>(Pixels));
		SpriteDynamicTexture.setPosition(
			window.getSize().x *0.5f - DynamicTexture.getSize().x * 0.5f,
			window.getSize().y *0.5f - DynamicTexture.getSize().y * 0.5f);
		window.draw(SpriteDynamicTexture);

		// end the current frame
		window.display();
    }

	delete[] Pixels;


    return 0;
}
