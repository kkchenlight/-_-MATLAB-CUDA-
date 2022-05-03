/*
 * @Author: kkchen
 * @Date: 2021-11-27 17:16:08
 * @LastEditors: kkchen
 * @LastEditTime: 2021-11-28 11:09:05
 * @Email: 1649490996@qq.com
 * @Description: 均值滤波
 */
#include<opencv2/opencv.hpp>
#include<opencv2/imgproc.hpp>
#include<stdlib.h>
#include<string>
#include<iostream>
#include<cuda_runtime.h>
using namespace std;
using namespace cv;


void cpuProcess(uchar* img, int height, int width, int step){
    for(int j = 1; j < height - 1; j++){
        for(int i = 1; i < width - 1; i++){
            int2 index0{i - 1, j - 1};
            int2 index1{i, j - 1};
            int2 index2{i + 1, j - 1};
            int2 index3{i - 1, j};
            int2 index4{i, j};
            int2 index5{i + 1, j};
            int2 index6{i - 1, j + 1};
            int2 index7{i, j + 1};
            int2 index8{i + 1, j + 1};


            uchar tmp = (img[index0.x + index0.y * step] + img[index1.x + index1.y * step] +
            img[index2.x + index2.y * step] + img[index3.x + index3.y * step] + 
            img[index4.x + index4.y * step] + img[index5.x + index5.y * step] + 
            img[index6.x + index6.y * step] + img[index7.x + index7.y * step] + 
            img[index8.x + index8.y * step]) / 9;
            //cout << " img = " << tmp << endl;
            //printf("img = %d \n", tmp);

            tmp = min(tmp, 255);
            tmp = max(0, tmp);
            img[i + j * step] = tmp;
        }
    }
}

__global__ void gpuProcess(uchar* img, int height, int width, int step){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    //int2 index[9];
    if( i > 0 && i < width - 1 && j > 0 && j < height - 1){
        /*uchar tmp = 0;
        for(int i = -1; i < 2; i++){
            for(int j = -1; j < 2; j++){
                index[(j+1) + (i+1)*3].x = ix + i;
                index[(j+1) + (i+1)*3].y = iy + j;
            }
        }

        for(int i = 0; i < 9; i++){
            printf("index[%d].x = %d, index[%d].y = %d \n", i, index[i].x, i, index[i].y);
            tmp += img[index[i].x + index[i].y * step];
        }
        tmp /= 9;
        //printf("tmp = %d \n", tmp);
        tmp = min(tmp, 255);
        tmp = max(0, tmp);
        img[ix + iy * step] = tmp;
    }*/
            int2 index0{i - 1, j - 1};
            int2 index1{i, j - 1};
            int2 index2{i + 1, j - 1};
            int2 index3{i - 1, j};
            int2 index4{i, j};
            int2 index5{i + 1, j};
            int2 index6{i - 1, j + 1};
            int2 index7{i, j + 1};
            int2 index8{i + 1, j + 1};


            uchar tmp = (img[index0.x + index0.y * step] + img[index1.x + index1.y * step] +
            img[index2.x + index2.y * step] + img[index3.x + index3.y * step] + 
            img[index4.x + index4.y * step] + img[index5.x + index5.y * step] + 
            img[index6.x + index6.y * step] + img[index7.x + index7.y * step] + 
            img[index8.x + index8.y * step]);
            //cout << " img = " << tmp << endl;
            //printf("img = %d \n", tmp);

            tmp = min(tmp, 255);
            tmp = max(0, tmp);
            img[i + j * step] = tmp * 9;
}
}
int main(int argv, char** argc){

    cout << " start >>>>>>>>>>>>>>>>>> " << endl;

    
    string imgPath = string(argc[1]);
    int blockSize = atoi(argc[2]);

    //read the img and cvtColort to gray
    Mat img = imread(imgPath);
    cvtColor(img, img, COLOR_BGR2GRAY);
    Mat img2 = imread(imgPath);
    cvtColor(img2, img2, COLOR_BGR2GRAY);
    //imwrite the oriimg for check
    imwrite("./oriimg.jpg", img2);

    //caculate the height and width, byteSize
    int imgHeight = img.rows;
    int imgWidth = img.cols;
    int step = img.step;
    size_t byteSize = imgHeight * imgWidth;

    //malloc the Memory for gpu and cpu;
    uchar* gpuPtr;
    cudaMalloc((void**)&gpuPtr, byteSize);

    //copy the cpu Memory to gpu Memory
    cudaMemcpy(gpuPtr, img.data, byteSize, cudaMemcpyHostToDevice);

    //start cpu process
    cpuProcess(img.data, imgHeight, imgWidth, step);

    //caculate the block and grid
    dim3 block(blockSize, blockSize);
    dim3 grid((imgWidth + block.x - 1) / block.x, (imgHeight + block.y - 1) / block.y);
    
    //start gpu process
    gpuProcess<<<grid, block>>>(gpuPtr, imgHeight, imgWidth, step);

    //copy the gpu Memory to cpu
    cudaMemcpy(img2.data, gpuPtr, byteSize, cudaMemcpyDeviceToHost);

    //imwrite the img for check;
    imwrite("./cpuResult.jpg", img);
    imwrite("./gpuResult.jpg", img2);
}