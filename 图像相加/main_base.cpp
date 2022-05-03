/*
 * @Author: kkchen
 * @Date: 2021-11-13 14:10:13
 * @LastEditors: kkchen
 * @LastEditTime: 2021-11-13 14:41:38
 * @Email: 1649490996@qq.com
 * @Description: 进行图像相减的动作,这里都将三通道图片转为gray后进行
 */
#include<cuda_runtime_api.h>
#include<opencv2/opencv.hpp>
#include<iostream>
#include<stdlib.h>

using namespace std;
using namespace cv;

void cpuProcess(uchar* ptr1, uchar* ptr2, const int width, const int height, const int step){
    for(int j = 0; j < height; j++){
        for(int i = 0; i < width; i++){
            long index = i + j * step;
            *(ptr1 + index) = *(ptr1 + index) / 2 + *(ptr2 + index) / 2;
        }
    }
}

__global__ gpuProcess(uchar* ptr1, uchar* ptr2, const int width, const int height, const int step){
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    long index = ix + iy + step;
    ptr1[index] = ptr1[index] / 2 + ptr2[index] / 2;
}



int main(int argc, char** argv){
    
    Mat img1 = imread(argv[1]);
    Mat img2 = imread(argv[2]);
    Mat gray1;
    Mat gray2;
    cvtColor(img1, gray1, COLOR_BGR2GRAY);
    cvtColor(img2, gray2, COLOR_BGR2GRAY);  

    resize(gray1, gray1, Size(1080, 720));
    resize(gray2, gray2, Size(1080, 720));

    Mat cpuPreMat1 = gray1.clone();
    Mat gpuPreMat1 = gray1.clone();

    Mat cpuPreMat2 = gray2.clone();
    Mat gpuPreMat2 = gray2.clone();
    
    int width = gray1.cols;
    int height = gray1.rows;
    int step = gray1.step;
    size_t byteSize = gray1.step * height;

    int blockXSize = atoi(argv[3]);
    int blockYSize = atoi(argv[4]);

    dim3 block(blockXSize, blockYSize);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    uchar* gpuPtr1;
    cudaMalloc((void**)& gpuPtr1, byteSize);
    cudaMemcpy(gpuPtr1, gpuPreMat1.data, byteSize, cudaMemcpyHostToDevice);

    uchar* gpuPtr2;
    cudaMalloc((void**)& gpuPtr2, byteSize);
    cudaMemcpy(gpuPtr2, gpuPreMat2.data, byteSize, cudaMemcpyHostToDevice);

    cpuProcess(cpuPreMat1, cpuPreMat2, width, height, step);
    imwrite("./cpuResult.jpg", cpuPreMat1);

    gpuProcess<<<grid, block>>>(gpuPreMat1, gpuPreMat2, width, height, step);
    cudaMemcpy(gpuPreMat1, gpuPtr1, byteSize, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    imwrite("./gpuResult.jpg",gpuPreMat1);

    imwrite("./yuanshipic1.jpg", gray1);
    imwrite("./yuanshipic2.jpg", gray2); 
    return 0;
}