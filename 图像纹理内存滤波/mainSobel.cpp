/*
 * @Author: kkchen
 * @Date: 2021-12-11 14:34:06
 * @LastEditors: kkchen
 * @LastEditTime: 2021-12-11 15:02:10
 * @Email: 1649490996@qq.com
 * @Description: file content
 */

#include<cuda_runtime.h>
#include<opencv2/opencv.hpp>
#include<string>
#include<stdlib.h>


using namespace std;
using namespace cv;
texture<uchar, cudaTextureType2D, cudaReadModeNormalizedFloat> texRef;


__global__ void kenelSobel(uchar* img, int width, int height){

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if( 0 < x && x < width && 0 < y && y < height){
        int index = x + y * width;
        img[index] = (8 * tex2D(texRef, x, y) - tex2D(texRef, x - 1, y - 1) - tex2D(texRef, x - 1, y)
        - tex2D(texRef, x - 1, y + 1) - tex2D(texRef, x, y -1 ) - tex2D(texRef, x , y + 1)
        - tex2D(texRef, x + 1, y - 1) - tex2D(texRef, x + 1, y) - tex2D(texRef, x + 1, y + 1)) * 255;
    }
}

int main(int argc, char** argv){

    string imgPath = string(argv[1]);
    int blockSize = atoi(argv[2]);


    Mat src = imread(imgPath);
    Mat srcGray;
    cvtColor(src, srcGray, COLOR_BGR2GRAY);

    //caculate the height and width
    int width = srcGray.cols;
    int height = srcGray.rows;
    size_t size = height * width;

    //malloc the cudaAarray
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar>();
    cudaArray* cudaArrayPtr;
    cudaMallocArray(&cudaArrayPtr, &channelDesc, width, height);
    cudaMemcpyToArray(cudaArrayPtr, 0, 0, srcGray.data, size, cudaMemcpyHostToDevice);
    
    //set texRef runtime Attributes
    texRef.addressMode[0] = cudaAddressModeWrap;
    texRef.addressMode[1] = cudaAddressModeWrap;
    texRef.filterMode     = cudaFilterModeLinear;
    texRef.normalized     = false;

    //combine the texture and array
    cudaBindTextureToArray(&texRef, cudaArrayPtr, &channelDesc);


    //prepare for output
    Mat output = Mat(height, width, CV_8UC1);
    memcpy(output.data, srcGray.data, size);

    uchar* outputGpuPtr;
    cudaMalloc(&outputGpuPtr, size);

    //caculate the block and grid
    dim3 block(blockSize, blockSize);
    dim3 grid((width + block.x - 1)/block.x, (block.y + height - 1)/block.y);

    //start the kernel;
    kenelSobel<<<block, grid>>>(outputGpuPtr, width, height);
    cudaDeviceSynchronize();

    //copy Mem to host;
    cudaMemcpy(output.data, outputGpuPtr, size, cudaMemcpyDeviceToHost);

    //free the memory
    cudaUnbindTexture(&texRef);

    cudaFreeArray(cudaArrayPtr);
    cudaFree(outputGpuPtr);

    //imwrite the image

    imwrite("./sobel.jpg", output);
    imwrite("./src.jpg", srcGray);

    return 0;
}