/*
 * @Author: kkchen
 * @Date: 2021-12-11 13:09:41
 * @LastEditors: kkchen
 * @LastEditTime: 2021-12-11 14:31:30
 * @Email: 1649490996@qq.com
 * @Description:训练纹理内存的使用
 */
#include<cuda_runtime.h>
#include<opencv2/opencv.hpp>
#include<stdlib.h>
#include<string>

using namespace std;
using namespace cv;



texture<uchar, cudaTextureType2D, cudaReadModeNormalizedFloat> texRef;


__global__ void resizeKernel(uchar* img, int2 inputSize, int2 outputSize){

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if(x < outputSize.x && y < outputSize.y){
        int index = x + y * outputSize.y;
        int xIn = float(inputSize.x) / float(outputSize.x) * x;
        int yIn = float(inputSize.y) / float(outputSize.y) * y;

        img[index] = tex2D(texRef, xIn, yIn) * 255;
        printf("img[%d] = %d", index, img[index]);
    }
}


int main(int argc, char** argv){
    
    string imgPath = string(argv[1]);
    int blockSize = atoi(argv[2]);

    Mat graySrc;
    Mat src = imread(imgPath);
    cvtColor(src, graySrc, COLOR_BGR2GRAY);

    

    //calculate the height and width

    int height = graySrc.rows;
    int width  = graySrc.cols;
    cv::Size inputSize = cv::Size(width, height);
    cv::Size outputSize = cv::Size(width/2, height/2);
    size_t byteSize = height * width;



    //malloc cuda array
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar>();

    cudaArray* cudaArrayPtr;
    cudaMallocArray(&cudaArrayPtr, &channelDesc, width, height);

    //copy image to cudaArray
    cudaMemcpyToArray(cudaArrayPtr, 0, 0, graySrc.data, byteSize, cudaMemcpyHostToDevice);


    // set the texRef for runtime
    texRef.addressMode[0] = cudaAddressModeWrap;
    texRef.addressMode[1] = cudaAddressModeWrap;
    texRef.filterMode     = cudaFilterModeLinear;
    texRef.normalized     = false;

    //Bind texture
    cudaBindTextureToArray(&texRef, cudaArrayPtr, &channelDesc);

    //malloc CudaMemory for outPut
    uchar* outputGpuPtr;
    cudaMalloc(&outputGpuPtr, byteSize / 4);

    /*void* outputCpuPtr;
    outputCpuPtr = malloc(byteSize);*/

    Mat outPutMat = cv::Mat(height/2, width/2, CV_8UC1);

    //set block and grid;
    dim3 block(blockSize, blockSize);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    
    int2 inputSizeint2 {inputSize.width, inputSize.height};
    int2 outputSizeint2 {outputSize.width, outputSize.height};
    resizeKernel<<<block, grid>>>(outputGpuPtr, inputSizeint2, outputSizeint2);

    //copy Mem from device to host
    cudaMemcpy(outPutMat.data, outputGpuPtr, byteSize/4, cudaMemcpyDeviceToHost);

    imwrite("./src.jpg", graySrc);
    imwrite("./resize.jpg", outPutMat);

    cudaUnbindTexture(texRef);
    cudaFree(outputGpuPtr);
    cudaFreeArray(cudaArrayPtr);
    return 0;
}
