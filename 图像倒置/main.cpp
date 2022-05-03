/*
 * @Author: your name
 * @Date: 2021-11-13 12:21:18
 * @LastEditors: kkchen
 * @LastEditTime: 2021-11-13 13:11:50
 * @Description: 用于图像倒置
 */
#include<cuda_runtime_api.h>
#include<opencv2/opencv.hpp>
#include<iostream>

using namespace std;
using namespace cv;


void Daozhi_Cpu(uchar* ptr, const int width, const int height, const int step){
    for(int j = 0; j < height; j++){
        for(int i = 0; i < width; i++){
            uchar* index = ptr + i + j * step;
            *index = 255 - *(index);
        }
    }
}


__global__ void Daozhi_Gpu(uchar* ptr, const int width, const int height, const int step){
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    int iy = threadIdx.y + blockDim.y * blockIdx.y;
    if( ix < width && iy < height){
        *(ptr + ix + iy * step) = 255 - *(ptr + ix + iy * step);
    }
}


int main(){

    Mat img = imread("./实验图片.jpg");
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    Mat CpuPreMat = gray.clone();

    Mat GpuPreMat = gray.clone();

    int step = gray.step;
    int height = gray.rows;
    int width = gray.cols;
    int ByteSize = step * height;

    uchar* gpuPtr;
    cudaMalloc((void**)&gpuPtr, ByteSize);

    cudaMemcpy(gpuPtr, GpuPreMat.data, ByteSize, cudaMemcpyHostToDevice);

    int uint = 2;
    dim3 block(uint, uint);
    dim3 grid((step + block.x - 1)/block.x, (height + block.y - 1)/block.y);

    Daozhi_Gpu<<<grid, block>>>(gpuPtr, width, height, step);
    
    cudaDeviceSynchronize();

    cudaMemcpy(GpuPreMat.data, gpuPtr, ByteSize, cudaMemcpyDeviceToHost);
    imwrite("yuanshi.jpg", gray);
    imwrite("GPUgray.jpg", GpuPreMat);

    Daozhi_Cpu(CpuPreMat.data, width, height, step);
    imwrite("CPUgray.jpg", CpuPreMat);
    return 0;
}