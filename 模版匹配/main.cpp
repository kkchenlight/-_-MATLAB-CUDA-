/*
 * @Author: kkchen
 * @Date: 2021-11-28 13:31:23
 * @LastEditors: kkchen
 * @LastEditTime: 2021-11-28 15:45:06
 * @Email: 1649490996@qq.com
 * @Description: file content
 */
#include<opencv2/opencv.hpp>
#include<opencv2/imgproc.hpp>
#include<string>
#include<iostream>
#include<stdlib.h>
#include<math.h>

using namespace std;
using namespace cv;

float getOnceMatch(uchar* img, uchar* imgTemp, Point start, Size imgTempSize, int imgStep){
    float score = 0;
    for(int j = start.y; j < start.y + imgTempSize.height; j++){
        for(int i = start.x; i < start.x + imgTempSize.width; i++){
            score += abs(img[i + j * imgStep] - imgTemp[i - start.x + (j - start.y) * imgTempSize.width]);
        }
    }
    return score; 
}


void cpuProcess(uchar* img, uchar* temp, float* result, Size imgSize, Size imgTempSize, int imgStep){
    const int imgHeight = imgSize.height;
    const int imgWidth  = imgSize.width;
    const int imgTempHeight = imgTempSize.height;
    const int imgTempWidth  = imgTempSize.width;
    const int resultHeight = imgHeight - imgTempHeight;
    const int resultWidth  = imgWidth - imgTempWidth;
    for(int j = 0; j < resultHeight; j++){
        for(int i = 0; i < resultWidth; i++){
            Point start = Point(i, j);
            result[i + j * resultWidth] = getOnceMatch(img, temp, start, imgTempSize, imgStep);
        }
    }
}


Point getMinLoc(float* result, int height, int width, int step){
    float minScore = 999999.0;
    Point loc{0, 0};
    for(int j = 0; j < height; j++){
        for(int i = 0; i < width; i++){
            if(minScore > result[i + j * step]){
                minScore = result[i + j * step];
                loc.x = i;
                loc.y = j;
            }
        }
    }
    return loc;
}

int main(int argc, char** argv){

    string imgPath = string(argv[1]);
    string imgPathTemp = string(argv[2]);
    Point location{0, 0};

    //read the img
    Mat imgori = imread(imgPath);

    Mat img = imread(imgPath);
    Mat imgTemp = imread(imgPathTemp);
    cvtColor(img, img, COLOR_BGR2GRAY);
    cvtColor(imgTemp, imgTemp, COLOR_BGR2GRAY);


    //get result Mat
    int resutlHeight = img.rows - imgTemp.rows + 1;
    int resultWidth  = img.cols - imgTemp.cols + 1;
    Mat result = Mat(resutlHeight, resutlHeight, CV_32FC1);


    //get Size
    cv::Size imgSize = cv::Size(img.cols, img.rows);
    cv::Size imgTempSize = cv::Size(imgTemp.cols, imgTemp.rows);
    int imgStep = img.step;

    //process use cpu
    cpuProcess(img.data, imgTemp.data, (float*)result.data, imgSize, imgTempSize, imgStep);

    //get the location
    location = getMinLoc((float*)result.data, result.rows, result.cols, result.step);

    if(location.x == 0 && location.y == 0){
        cout << "WARNING: WE GOT NO MATCH >>>>>>>>>>>>>>> " << endl;
    }else{
        cout << "MATCHED: x = " << location.x << " y = " << location.y << endl;
    }
    
    rectangle(imgori, cv::Point(location.x, location.y), Point(location.x + imgTemp.cols, location.y + imgTemp.rows), Scalar(0, 255, 0), 2, 8, 0);
    
    imwrite("./resultshoudong.jpg", imgori);
    return 0;
}

