/*

Copyright (c) 2015 Davide Ghilardi

See the file LICENSE for copying permission.

*/

#include <iostream>
#include <vector>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <math.h>
#include <cmath>

#define IMAGENAME "../img/building.jpg"
#define HOUGH_COLS 180

using namespace std;
using namespace cv;

void drawLine(Mat &destination, int r, int theta, int height){
    float m = -1/tan(theta/180.0*M_PI);
    float q = (r-height)/sin(theta/180.0*M_PI);
    int y;
    for(int x=0; x<destination.cols; x++){
        y=round(m*x+q);
        if(y<destination.rows && y>=0){
            destination.at<uchar>(y,x)=255;
        }
    }
}

void antiHoughTransform(Mat &hough, Mat &destination, Size destSize, float houghThreshold){
    destination.create(destSize, CV_8UC1);
    destination.setTo(Scalar(0));
    for(int x=0; x<hough.cols; x++){
        for(int y=0; y<hough.rows; y++){
            if(hough.at<float>(y,x)>houghThreshold){
                drawLine(destination, y, x, hough.rows/2);
            }
        }
    }
}

int main()
{
    Mat source = imread(IMAGENAME,CV_LOAD_IMAGE_GRAYSCALE);
    if(source.empty()){
        cerr << "ERROR! image not loaded"<<endl;
        return -1;
    }
    Mat workImg;
    Canny(source, workImg, 80, 800);

    int houghRows = sqrt(pow(workImg.rows, 2)+pow(workImg.cols, 2));
    int r;
    Mat houghSpace(2*houghRows, HOUGH_COLS, CV_32FC1, Scalar(0));
    for(int x=0; x<workImg.cols; x++){
        for(int y=0; y<workImg.rows; y++){
            if(workImg.at<uchar>(y,x)==255){
                for(int theta=0; theta<houghSpace.cols; theta++){
                    r = (round(x*cos(theta/180.0*M_PI)+y*sin(theta/180.0*M_PI)))+houghRows;
                    //if(r>houghRows) cout << r <<": x->"<<x<<"  y->"<<y<<"  theta:"<<theta<< endl;
                    if (r>=0 && r<houghSpace.rows) houghSpace.at<float>(r, theta) += 1.0;
                }
            }
        }
    }
    cout << houghRows << endl;
    //cout << houghSpace << endl;

    Mat antiTransform;
    float maxValue = *(max_element(houghSpace.begin<float>(), houghSpace.end<float>()));
    antiHoughTransform(houghSpace, antiTransform, workImg.size(), maxValue-200);


    float minValue = *(min_element(houghSpace.begin<float>(), houghSpace.end<float>()));


    Mat result;
    houghSpace.convertTo(result, CV_8UC1, 255.0/(maxValue-minValue), -minValue*255.0/(maxValue-minValue));

    cvNamedWindow("Source",2);
    imshow("Source",source);

    cvNamedWindow("Canny",2);
    imshow("Canny",workImg);

    cvNamedWindow("Result",2);
    imshow("Result",result);

    cvNamedWindow("Anti-transform",2);
    imshow("Anti-transform", antiTransform);
    waitKey();
    return 0;
}

