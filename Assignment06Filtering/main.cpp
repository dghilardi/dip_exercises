/*

Copyright (c) 2015 Davide Ghilardi

See the file LICENSE for copying permission.

*/

#include <iostream>
#include <vector>
#include "cv.h"
#include "highgui.h"
#include <math.h>
#include <cmath>

#define GAUSS_IMAGENAME "./lenaGaussian.bmp"
#define SALT_PEPPER_IMAGE "./SaltAndPepperNoise.jpg"
#define NO_NOISE_IMAGE "./ippi.jpg"
#define FILTERROWS 10
#define FILTERCOLS 10

using namespace cv;
using namespace std;
using namespace cv::flann;

void medianFilter(Mat *image, Mat *dest, int filterSize){
    uchar resultPixel;
    int imgX, imgY;
    vector<uchar> elemList;
    for(int xi=0; xi<image->cols; xi++){
        for(int yi=0; yi<image->rows; yi++){
            resultPixel=0;
            elemList.clear();
            for(int xk=0; xk<filterSize; xk++){
                for(int yk=0; yk<filterSize; yk++){
                    imgX=abs(xi+xk-ceil(filterSize/2));
                    imgY=abs(yi+yk-ceil(filterSize/2));
                    if(imgX > image->cols) imgX=image->cols - (imgX-image->cols);
                    if(imgY > image->rows) imgY=image->rows - (imgY-image->rows);
                    elemList.push_back(image->at<uchar>(imgY,imgX));
                }
            }
            sort(elemList.begin(), elemList.end());
            resultPixel=(uchar)elemList[floor(elemList.size()/2)];
            dest->at<uchar>(yi,xi)=resultPixel;
        }
    }

}

void convolution(Mat *kernel, Mat *image, Mat *dest){
    float kernelValue;
    uchar resultPixel;
    int imgX, imgY;
    Mat matTmp(FILTERROWS, FILTERCOLS, CV_8UC1, Scalar(0));
    for(int xi=0; xi<image->cols; xi++){
        for(int yi=0; yi<image->rows; yi++){
            resultPixel=0;
            for(int xk=0; xk<kernel->cols; xk++){
                for(int yk=0; yk<kernel->rows; yk++){
                    kernelValue=kernel->at<float>(yk,xk);
                    imgX=abs(xi+xk-ceil(kernel->cols/2));
                    imgY=abs(yi+yk-ceil(kernel->rows/2));
                    if(imgX > image->cols) imgX=image->cols - (imgX-image->cols);
                    if(imgY > image->rows) imgY=image->rows - (imgY-image->rows);
                    matTmp.at<uchar>(xk,yk)=(uchar)image->at<uchar>(imgY,imgX)*kernelValue;
                    resultPixel+=matTmp.at<uchar>(xk,yk);
                }
            }
            resultPixel=(uchar)resultPixel;
            dest->at<uchar>(yi,xi)=resultPixel;
        }
    }
}


void createFilter(Mat *filter, int mux, int muy, int varx, int vary)
{
    int x, y;
    double sum=0, exponent;
    for(int i=0; i<filter->rows; i++){
        for(int j=0; j<FILTERCOLS; j++){
            x = j-ceil(FILTERCOLS/2);
            y = -i+ceil(FILTERROWS/2);
            exponent = -(varx*pow(x-mux,2)+vary*pow(y-muy,2))/(double)(2*varx*vary);
            //cout << "exponent: " << exponent << endl;
            filter->at<float>(i,j)=pow(M_E,exponent);
            sum+=filter->at<float>(i,j);
        }
    }
    for(int i=0; i<FILTERROWS; i++){
        for(int j=0; j<FILTERCOLS; j++){
            filter->at<float>(i,j)/=sum;
            //filter.at<float>(i,j)=1-filter.at<float>(i,j);
        }
    }
}

void imageSubtraction(Mat *minuend, Mat *subtrahend, Mat *difference){
    for(int x=0; x<minuend->cols; x++){
        for(int y=0; y<minuend->rows; y++){
            difference->at<uchar>(y,x) = minuend->at<uchar>(y,x)-subtrahend->at<uchar>(y,x);
        }
    }
}

int main(int argc, char *argv[]){

    int selection;
    cout << "Image to load:
" << "1. No noise 
2. Salt and Pepper noise 
3. Gaussian noise
-> ";
    cin >> selection;
    string imageName;
    switch (selection){
        case 1:
            imageName=NO_NOISE_IMAGE;
            break;
        case 2:
            imageName=SALT_PEPPER_IMAGE;
            break;
        case 3:
            imageName=GAUSS_IMAGENAME;
            break;
        default:
            cerr << "Invalid selection";
            return 1;
            break;
    }

    //Load the image
    Mat source = imread(imageName,0);
    if(source.empty()){
        cerr << "ERROR! image not loaded"<<endl;
        return -1;
    }

    cout << "Insert:" << endl << "1. Highpass filter 
2. Lowpass filter 
3. Median filter
"<<"-> ";
    cin >> selection;

    Mat filter(FILTERROWS, FILTERCOLS, CV_32FC1, Scalar(0));
    Mat result(source.rows, source.cols, CV_8UC1, Scalar(0));
    Mat lowPassed(source.rows, source.cols, CV_32FC1, Scalar(0));
    int mux=0, muy=0, varx=10, vary=10;
    switch (selection){
        case 1:
            createFilter(&filter, mux, muy, varx, vary);
            convolution(&filter, &source, &lowPassed);
            imageSubtraction(&source, &lowPassed, &result);
            break;
        case 2:
            createFilter(&filter, mux, muy, varx, vary);
            convolution(&filter, &source, &result);
            break;
        case 3:
            medianFilter(&source, &result, 3);
            break;
        default:
            cerr << "Invalid Selection" << endl;
            return 1;
            break;

    }

    cout << filter << endl;

    cvNamedWindow("Source",2);
    imshow("Source",source);
    cvNamedWindow("Result",2);
    imshow("Result",result);
    waitKey();
    return 0;
}
