/*
 * @Author: kkchen
 * @Date: 2021-12-11 16:03:26
 * @LastEditors: kkchen
 * @LastEditTime: 2021-12-11 17:12:54
 * @Email: 1649490996@qq.com
 * @Description: 基于cpu和gpu实现直方图均衡化
 */


#include<opencv2/opencv.hpp>
#include<string>
#include<stdlib.h>
#include<vector>

using namespace std;
using namespace cv;

void getHistTable(uchar* img, size_t size, vector<uchar>& histVec){

    //clear Ori histVec
    for(int i = 0; i < histVec.size(); i++){
        histVec[i] = 0;
    }
    
    //sum the hist
    printf("size = %d\t histVec.size = %d\n", size, histVec.size());
    for(int i = 0; i < size; i++){
        histVec[img[i]] += 1;
        //printf("histVect[%d] = %d",img[i], histVec[img[i]]);     
    }
}

void mapHist(vector<uchar>& histVec){
    int sum = 0;

    vector<int> histSumVec;
    printf("histVec.size() = %d", histVec.size());
    for(int i = 0; i < histVec.size(); i++){
            sum += histVec[i];
          
            histSumVec.push_back(sum);
            printf("sum = %d, histSumVec[%d] = %d\t", sum, i, histVec[i]);
    }
    cout << "sum = " << sum << endl;

    for(int i = 0; i < histVec.size(); i++){
        histVec[i] = histSumVec[i] * 255 / sum;
        printf("histVec[%d] = %d \t", i, histSumVec[i]);
    }
}

void histImage(uchar* img, size_t size, vector<uchar>& histVec){
    for(int i = 0; i < size; i++){
        img[i] = histVec[img[i]];
    }
}

int main(int argc, char** argv){

    string imgPath = string(argv[1]);
    vector<uchar> histVec;
    for(int i = 0; i < 255;i++){
        histVec.push_back(0);
    }

    Mat src = imread(imgPath);
    Mat srcGray;

    cvtColor(src, srcGray, COLOR_BGR2GRAY);
    imwrite("./srcGray.jpg", srcGray);


    //caculate the width and height
    int height = srcGray.rows;
    int width = srcGray.cols;
    size_t size = width * height;
    
    //caculate the hist 
    std::cout << "histVec.size() = " << histVec.size() << std::endl;
    getHistTable(srcGray.data, size, histVec);

    //map the hist
    std::cout << "histVec.size() = " << histVec.size() << std::endl;
    mapHist(histVec);

    //transform the image
    histImage(srcGray.data, size, histVec);

    imwrite("./histImage.jpg", srcGray);

    return 0;
}