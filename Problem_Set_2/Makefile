#all:
#	nvcc -g -std=c++11 main.cpp cuda_kernels.cu -lsfml-graphics -lsfml-window -lsfml-system -o App.out
CC=g++
NVCC=nvcc


all: App

App: main.o cuda_kernels.o
	$(NVCC) -g -std=c++11 --gpu-architecture=sm_50 main.o cuda_kernels.o -lsfml-graphics -lsfml-window -lsfml-system -o App.out -run

main.o: main.cpp
	$(NVCC) -c -std=c++11 --gpu-architecture=sm_50 main.cpp

cuda_kernels.o: cuda_kernels.cu
	$(NVCC) -c -std=c++11 --gpu-architecture=sm_50 cuda_kernels.cu -o cuda_kernels.o

clean:
	rm -rf *o App.out
