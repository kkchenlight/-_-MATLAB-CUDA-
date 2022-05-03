/*
 * @Author: kkchen
 * @Date: 2021-11-28 11:13:58
 * @LastEditors: kkchen
 * @LastEditTime: 2021-11-28 12:54:00
 * @Email: 1649490996@qq.com
 * @Description: 实现高斯滤波
 */

#include<opencv2/opencv.hpp>
#include<opencv2/imgproc.hpp>
#include<iostream>
#include<string>
#include<cuda_runtime.h>
#include<stdlib.h>


using namespace std;
using namespace cv;

void cpuProcess(uchar* img, int height, int width, int step){

    for(int j = 1; j < height - 1; j++){
        for(int i = 1; i < width - 1; i++){
            const int x_left = i - 1;
            const int x_right = i + 1;
            const int y_up = j - 1;
            const int y_down = j + 1;
            const int x = i;
            const int y = j;

            int tmp = 0;
            tmp = img[x_left + y_up * step] / 16 + 
                  img[x + y_up * step] / 8 + 
                  img[x_right + y_up * step] / 16 + 
                  img[x_left + y * step] / 8 +
                  img[x + y * step] / 4 + 
                  img[x_right + y * step] / 8 +
                  img[x_left + y_down * step] / 16+ 
                  img[x + y_down * step] / 8+ 
                  img[x_right + y_down * step] / 16;
            //tmp = tmp >> 4;
            cout << " tmp = " << tmp << endl;
            //tmp = min(tmp, 255);
            //tmp = max(0, tmp);
            img[x + y * step] = uchar(tmp);
        }
    }

}


__global__ void gpuProcess(uchar* img, int height, int width, int step){
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if(x > 0 && x < width - 1 && y > 0 && y < height - 1){
            const int x_left = x - 1;
            const int x_right = x + 1;
            const int y_up = y - 1;
            const int y_down = y + 1;

            int tmp = 0;
            //int tmp = 0;
            tmp = img[x_left + y_up * step] / 16 + 
                  img[x + y_up * step] / 8 + 
                  img[x_right + y_up * step] / 16 + 
                  img[x_left + y * step] / 8 +
                  img[x + y * step] / 4 + 
                  img[x_right + y * step] / 8 +
                  img[x_left + y_down * step] / 16+ 
                  img[x + y_down * step] / 8+ 
                  img[x_right + y_down * step] / 16;
            //tmp = tmp >> 4;
            //cout << " tmp = " << tmp << endl;
            //tmp = min(tmp, 255);
            //tmp = max(0, tmp);
            img[x + y * step] = uchar(tmp);
    }

}
int main(int argc, char** argv){
    
    string imgPath = string(argv[1]);
    int blockSize = atoi(argv[2]);

    //read the img and cvtColor to gray
    Mat img = imread(imgPath);
    cvtColor(img, img, COLOR_BGR2GRAY);
    Mat img2 = imread(imgPath);
    cvtColor(img2, img2, COLOR_BGR2GRAY);

    //imwrite the img for check
    imwrite("./ori.jpg", img);

    //caculate the width and the height 
    int width = img.cols;
    int height = img.rows;
    int step = img.step;
    size_t byteSize = width * height;

    //malloc the memory for gpu
    uchar* gpuPtr;
    cudaMalloc((void**)&gpuPtr, byteSize);

    //memcpy to gpu
    cudaMemcpy(gpuPtr, img.data, byteSize, cudaMemcpyHostToDevice);

    //start cpu process
    cpuProcess(img.data, height, width, step);

    //caculate the grid and block
    dim3 block(blockSize, blockSize);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    //start gpu process
    gpuProcess<<<grid, block>>>(gpuPtr, height, width, step);
    cudaDeviceSynchronize();
    

    //copy the gpu memory to cpu
    cudaMemcpy(img2.data, gpuPtr, byteSize, cudaMemcpyDeviceToHost);


    //imwrite the img for check
    imwrite("cpuResult.jpg", img);
    imwrite("gpuResult.jpg", img2);

    return 0;
}