/*
 * @Author: kkchen
 * @Date: 2021-12-11 17:55:23
 * @LastEditors: kkchen
 * @LastEditTime: 2021-12-18 17:00:55
 * @Email: 1649490996@qq.com
 * @Description: file content
 */
#include<string>
#include<stdlib.h>
#include<cuda_runtime.h>
#include<opencv2/opencv.hpp>
#include<device_atomic_functions.h>

texture<uchar, cudaTextureType2D, cudaReadModeNormalizedFloat> texRef;

using namespace std;
using namespace cv;

__global__ void getHistVec(uchar* img, int* histVec, int width, int height, int step){
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
   // printf("x = %d, y = %d, width = %d, height = %d, step = %d\n", x, y, width, height, step);
    if(x < width && y < height){
        int tid = x + y * step;
 //printf("tid = %d\n", tid);
        uchar value = img[tid];
        
       // printf("x = %d, y = %d, value = %d\n", x, y, value);
        atomicAdd(&(histVec[value]), 1); 
       // printf("histVec[%d] = %d \t", value, histVec[value]);
    }
}


__global__ void mapTheImage(uchar* img, int width, int height, int step){
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    int iy = threadIdx.y + blockDim.y * blockIdx.y;

    if(ix < width && iy < height){
        int index = ix + iy * step;
        img[index] = tex2D(texRef, int(img[index]), 0) * 255;
    }
}

int main(int argc, char** argv){

    string imgPath = string(argv[1]);
    int blockSize = atoi(argv[2]);

    Mat src = imread(imgPath);
    Mat graySrc;
    cvtColor(src, graySrc, COLOR_BGR2GRAY);

    //caculate height and width
    int height = graySrc.rows;
    int width = graySrc.cols;
    size_t size = height * width;

    //malloc the cudaMemory
    int* histCudaPtr;
    cudaMalloc(&histCudaPtr, 256 * sizeof(int));
    uchar* imgCudaPtr;
    cudaMalloc(&imgCudaPtr, size);

    int histCpuPtr[256] = {0};
    
    //imwrite jpg for check
    imwrite("./src.jpg", graySrc);

    //copy the Memory to cuda
    cudaMemcpy(imgCudaPtr, graySrc.data, size, cudaMemcpyHostToDevice);

    //caculate block and grid
    dim3 block(blockSize, blockSize);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1)/block.y);
    getHistVec<<<block, grid>>>(imgCudaPtr, histCudaPtr, width, height, graySrc.step);

    //copy the Memory to cpu
    cudaMemcpy(histCpuPtr, histCudaPtr, 256 * sizeof(int), cudaMemcpyDeviceToHost);

    printf("\n");

    for(int i = 0; i < 256; i++){
        printf("histCpuPtr[%d] = %d \t", i, histCpuPtr[i]);
    }


    //caculate hist
    int histMap[256] = {0};
    int sumHist = 0;
    for(int i = 0; i < 256; i++){
        sumHist += histCpuPtr[i];
        histMap[i] = sumHist;
    }

    uchar histMapUchar[256] = {0};
    for(int i = 0; i < 256; i++){
        histMapUchar[i] = (histMap[i] * 255) / sumHist;
    }


    //Malloc the cudaArray
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar>();
    cudaArray* cudaArrayPtr;
    cudaMallocArray(&cudaArrayPtr, &channelDesc, 256, 1);
    cudaMemcpyToArray(cudaArrayPtr, 0, 0, (void*)histMapUchar, 256 * sizeof(uchar), cudaMemcpyHostToDevice);

    //set the runtime properity of texture
    texRef.addressMode[0] = cudaAddressModeWrap;
    texRef.addressMode[1] = cudaAddressModeWrap;
    texRef.filterMode     = cudaFilterModeLinear;
    texRef.normalized     = false;

    //bind the cudaArray to texture
    cudaBindTextureToArray(&texRef, cudaArrayPtr, &channelDesc);


    //start the kernel
    mapTheImage<<<grid, block>>>(imgCudaPtr, width, height, graySrc.step);

    //copy the img to the host;
    cudaMemcpy(graySrc.data, imgCudaPtr, size, cudaMemcpyDeviceToHost);


    //imwrite the jpg for check
    cv::imwrite("./cudaProcess.jpg", graySrc); 


    cudaUnbindTexture(&texRef);
    cudaFreeArray(cudaArrayPtr);
    cudaFree(imgCudaPtr);
    cudaFree(histCudaPtr);
    return 0;
}