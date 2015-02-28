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

#define GAUSS_IMAGENAME "../img/lenaGaussian.jpg"
#define SALT_PEPPER_IMAGE "../img/SaltAndPepperNoise.jpg"
#define NO_NOISE_IMAGE "../img/ippi.jpg"
#define FILTERROWS 201
#define FILTERCOLS 201

using namespace cv;
using namespace std;

void createLowPass(Mat *filter){
    int cols = filter->cols;
    int rows = filter->rows;
    for(int x=0; x<filter->cols; x++){
        for(int y=0; y<filter->rows; y++){
            if(!(pow(x-cols/2,2)+pow(y-rows/2,2)<=pow(cols/2.5,2))){
            //int a = abs(x-cols/2);
            //if(!(pow(a-cols/2,2)+pow(y-rows/2,2)<=pow(cols/2.5,2))){
                filter->at<float>(y,x)=1;
            }

        }
    }
}

void prePost(Mat *toPrePost){
    int cols = toPrePost->cols;
    int rows = toPrePost->rows;
    for(int x=0; x<cols; x++){
        for(int y=0; y<rows; y++){
            toPrePost->at<float>(y,x)*=pow(-1, x+y);
        }
    }

}

void post(Mat *toPrePost){
    int cols = toPrePost->cols;
    int rows = toPrePost->rows;
    for(int x=0; x<cols; x++){
        for(int y=0; y<rows; y++){
            toPrePost->at<float>(y,x)=abs(toPrePost->at<float>(y,x));
        }
    }

}

void pointProduct(Mat *factor1, Mat *factor2, Mat *product){
    if(!(factor1->cols==factor2->cols && product->cols==factor2->cols &&factor1->rows==factor2->rows && product->rows==factor2->rows)){
        cerr << "ERROR! dimension must agree!" << endl;
        cerr << "factor 1: " << factor1->rows << " - " << factor1->cols << endl;
        cerr << "factor 2: " << factor2->rows << " - " << factor2->cols << endl;
        cerr << "product: " << product->rows << " - " << product->cols << endl;

    }
    int channels = factor1->channels();

    for(int i=0; i<factor1->rows; i++){
        for(int j=0; j<factor1->cols; j++){
            for(int k=0; k<channels; k++){
                //cout << "a" << endl;
                product->at<float>(i,j*channels+k) = factor1->at<float>(i,j*channels+k)*factor2->at<float>(i,j*channels+k);
                //cout << product->at<float>(i,j*channels+k) << "=" << factor1->at<float>(i,j*channels+k) << "*" <<factor2->at<float>(i,j*channels+k)<< endl;
            }
        }
    }
}

int main()
{
    string imageName = NO_NOISE_IMAGE;
    Mat source = imread(imageName,CV_LOAD_IMAGE_GRAYSCALE);
    if(source.empty()){
        cerr << "ERROR! image not loaded"<<endl;
        return -1;
    }
    Mat sourceFreqFloatC2, sourceFloatC1;
    source.convertTo(sourceFreqFloatC2, CV_32F);
    source.convertTo(sourceFloatC1, CV_32F);

    Mat lowPass(FILTERROWS+source.rows-1, FILTERCOLS+source.cols-1, CV_32F, Scalar(0));
    createLowPass(&lowPass);

    cout << sourceFreqFloatC2.cols << " - " << sourceFreqFloatC2.rows << endl;
    copyMakeBorder(sourceFreqFloatC2, sourceFreqFloatC2, 0, FILTERROWS-1, 0, FILTERCOLS-1, 0);
    prePost(&sourceFreqFloatC2);

    dft(sourceFreqFloatC2, sourceFreqFloatC2, DFT_REAL_OUTPUT);
    //idft(sourceFreqFloatC2, sourceFreqFloatC2, DFT_SCALE);
    Mat product(FILTERROWS+source.rows-1, FILTERCOLS+source.cols-1, CV_32F, Scalar(0));

    pointProduct(&sourceFreqFloatC2, &lowPass, &product);

    Mat result(FILTERROWS+source.rows-1, FILTERCOLS+source.cols-1, CV_32F, Scalar(0));
    //product = sourceFreqFloatC2;
    idft(product, result, DFT_SCALE /*| DFT_REAL_OUTPUT*/);
    prePost(&result);
    //cout << result << endl;

    cout << result.channels() << product.channels() << endl;
    result.convertTo(result, CV_8UC1);
    sourceFreqFloatC2.convertTo(sourceFreqFloatC2, CV_8UC1);
    sourceFreqFloatC2.convertTo(sourceFreqFloatC2, CV_8UC1);

    product.convertTo(product, CV_8UC1);
    cvNamedWindow("Freq Image",2);
    imshow("Freq Image",product);
    //vector<Mat> planes;
    //split(result, planes);
    //cout << result.channels() << endl;
    //planes[0].convertTo(planes[0], CV_8UC1);

    cvNamedWindow("Result",2);
    imshow("Result",result);
    cvNamedWindow("Filter (freq)",2);
    imshow("Filter (freq)",lowPass);
    cvNamedWindow("Source",2);
    imshow("Source",source);
    waitKey();
    return 0;
}

