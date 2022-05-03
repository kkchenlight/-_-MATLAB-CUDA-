/*
 * @Author: kkchen
 * @Date: 2021-11-27 16:23:41
 * @LastEditors: kkchen
 * @LastEditTime: 2021-11-27 17:09:23
 * @Email: 1649490996@qq.com
 * @Description: file content
 */
#include<opencv2/opencv.hpp>
#include<opencv2/imgproc.hpp>
#include<cuda_runtime.h>
#include<stdlib.h>
#include<string>
#include<iostream>

using namespace std;
using namespace cv;



void imgProcessCpu(uchar* img1, uchar* img2, int height, int width){
    uchar tmp;
    for(int j = 0; j < height; j++){
        for(int i = 0; i < width; i++){
            int index = i + j * width;
            tmp = img1[index] - img2[index] + 128;
            tmp = max(tmp, 255);
            tmp = min(0, tmp);
            img1[index] = tmp;
        }
    }
}

__global__ void imgProcessGpu(uchar* img1, uchar* img2, int height, int width){
    uchar tmp;
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    int iy = threadIdx.y + blockDim.y * blockIdx.y;
    if(ix < width && iy < height){
        int index = ix + iy * width;
            tmp = img1[index] - img2[index] + 128;
            tmp = max(tmp, 255);
            tmp = min(0, tmp);
            img1[index] = tmp;
    }
}



int mian(int argv, char** argc){
    
    cout << "starting >>>>>>>>>>>>>>>>> " << endl;
    string imgPath1 = string(argc[1]);
    string imgPath2 = string(argc[2]);
    int N = atoi(argc[3]);

    cout << "imgPath1 = " << imgPath1 << endl;
    cout << "imgPath2 = " << imgPath2 << endl;
    cout << "BlockSize = " << N << endl;

    //read the image and cvtColort to gray;
    Mat imgOri1 = imread(imgPath1);
    Mat imgOri2 = imread(imgPath2);
    cvtColor(imgOri1, imgOri1, COLOR_BGR2GRAY);
    cvtColor(imgOri2, imgOri2, COLOR_BGR2GRAY);

    //imwrite the original img for check
    imwrite("./ori1.jpg", imgOri1);
    imwrite("./ori2.jpg", imgOri2);

    cout << "imwrite Ori Done >>>>>>>>>>>>>>>>> " << endl;

    //caculate the height and width
    int imgHeight = imgOri1.rows;
    int imgWidth  = imgOri2.cols;

    //get the byteSize for malloc;
    size_t byteSize = imgHeight * imgWidth;

    //malloc the gpu memory 
    void* gpuPtr1;
    void* gpuPtr2;
    cudaMalloc((void**)&gpuPtr1, byteSize);
    cudaMalloc((void**)&gpuPtr2, byteSize);

    //copy the cpu memoryt to gpu memory
    cudaMemcpy(gpuPtr1, imgOri1.data, byteSize, cudaMemcpyHostToDevice);
    cudaMemcpy(gpuPtr2, imgOri2.data, byteSize, cudaMemcpyHostToDevice);

    cout << "memcpy to gpu Done >>>>>>>>>>>>>>>>> " << endl;

    // process use cpu
    imgProcessCpu(imgOri1.data, imgOri2.data, imgHeight, imgWidth);

    cout << "cpu process Done >>>>>>>>>>>>>>>>> " << endl;

    //caculate the  grid and block
    dim3 block(N, N);
    dim3 grid((imgWidth + block.x - 1) / block.x , (imgHeight + block.y - 1) / block.y);

    //process use gpu
    imgProcessGpu<<<grid,block>>>((uchar*)gpuPtr1, (uchar*)gpuPtr2, imgHeight, imgWidth);

    cout << "gpu process Done >>>>>>>>>>>>>>>>> " << endl;

    //cpy the gpu memory to cpu memory
    cudaMemcpy(imgOri2.data, gpuPtr1, byteSize, cudaMemcpyDeviceToHost);

    //imwrite the img for result check
    imwrite("./cpureslut.jpg", imgOri1);
    imwrite("./gpuresult.jpg", imgOri2);
    cout << "memcpy to cpu Done >>>>>>>>>>>>>>>>> " << endl;

    return 0;
}