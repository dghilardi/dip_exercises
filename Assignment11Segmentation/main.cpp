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
#include <list>

#define IMAGENAME "../img/manichini.jpg"

using namespace std;
using namespace cv;

typedef struct QuadNode
{
    Rect *region;
    struct QuadNode *childs[4];
} QuadNode;

void binarization(Mat &source, Mat &destination, int threshold){
    destination.create(source.size(), CV_8UC1);
    destination.setTo(0);
    for(int x=0; x<source.cols; x++){
        for(int y=0; y<source.rows; y++){
            if(source.at<uchar>(y,x)>threshold) destination.at<uchar>(y,x) = 255;
        }
    }
}

uchar regionMean(Mat &source, Mat &mask, uchar region){
    int sum = 0;
    int count=0;
    for(int x=0; x<source.cols; x++){
        for(int y=0; y<source.rows; y++){
            if(mask.at<uchar>(y,x)==region){
                sum+=source.at<uchar>(y,x);
                count++;
            }
        }
    }
    uchar result;
    count == 0 ? result=0 : result = sum/count;
    return result;
}

float regionWeightedVariance(Mat &source, Mat &mask, uchar mean, uchar region){
    float sum = 0;
    int count=0;
    for(int x=0; x<source.cols; x++){
        for(int y=0; y<source.rows; y++){
            if(mask.at<uchar>(y,x)==region){
                sum+=pow((int)source.at<uchar>(y,x) - (int)mean,2);
                count++;
            }
        }
    }
    float result;
    count == 0 ? result=0 : result = sum;
    return result;
}
int findOptimalThreshold(Mat source)
{
    Mat resultColorBased;
    float wvar0, wvar255, rho=-1;
    uchar mean0, mean255;
    int optimalThreshold;
    for(int i=0; i<256; i++){
        binarization(source, resultColorBased, i);

        mean255 = regionMean(source, resultColorBased, 255);
        wvar255 = regionWeightedVariance(source, resultColorBased, mean255, 255);
        mean0 = regionMean(source, resultColorBased, 0);
        wvar0 = regionWeightedVariance(source, resultColorBased, mean0, 0);
        if(wvar0+wvar255 < rho || rho==-1){
            rho = wvar0 + wvar255;
            optimalThreshold = i;
        }
        cout << i*100/255 << "%" << endl;
    }

    return optimalThreshold;
}

bool findFreePoint(Mat &matrix, Point &result){
    for(int i=0; i<matrix.rows; i++){
        for(int j=0; j<matrix.cols; j++)
            if(matrix.at<short>(i,j)==-1){
                result.x = j;
                result.y = i;
                return true;
            }
    }
    return false;
}

bool pointBelongsToMatrix(int x, int y, Mat &matrix){
    return (x>=0) && (y>=0) && (x<matrix.cols) && (y<matrix.rows);
}

void grow(Mat &source, Mat &destination, Point &startingPoint, int threshold){
    short value = source.at<uchar>(startingPoint);
    list<Point *> toAnalyze;
    toAnalyze.push_back(new Point(startingPoint));
    destination.at<short>(startingPoint) = value;
    Point *actualItem;
    while(toAnalyze.size()>0){
        actualItem = toAnalyze.front();
        for(int dx=-1; dx<=1; dx++){
            for(int dy=-1; dy<=1; dy++){
                Point *toAdd = new Point;
                toAdd->x=actualItem->x+dx;
                toAdd->y=actualItem->y+dy;
                if(pointBelongsToMatrix(actualItem->x+dx, actualItem->y+dy, source) && !(dx==0 && dy==0)){
                    if(abs((int)source.at<uchar>(*toAdd)-value)<threshold &&
                            destination.at<short>(actualItem->y+dy,actualItem->x+dx)==-1){
                        destination.at<short>(actualItem->y+dy,actualItem->x+dx) = value;
                        toAnalyze.push_back(toAdd);
                    }
                }
            }
        }
        delete actualItem;
        toAnalyze.pop_front();
    }

}

void regionGrowing(Mat &source, Mat &destination, int threshold){
    destination.create(source.size(), CV_16SC1);
    destination.setTo(Scalar(-1));
    Point freePoint;
    while(findFreePoint(destination, freePoint)){
        grow(source, destination, freePoint, threshold);
    }
}

void splitImage(Mat *image, Mat *destination, QuadNode *node, int threshold){
    Rect *childs[4];
    Mat *imageRegions[4], *dstRegions[4];

    int remainingRows = image->rows-image->rows/2, remainingCols=image->cols-image->cols/2;
    childs[0] = new Rect(0,0,image->cols/2, image->rows/2);
    childs[1] = new Rect(image->cols/2,0,remainingCols, image->rows/2);
    childs[2] = new Rect(0,image->rows/2,image->cols/2, remainingRows);
    childs[3] = new Rect(image->cols/2,image->rows/2,remainingCols, remainingRows);

    for(int i=0; i<4; i++){
        imageRegions[i] = new Mat(*image, *childs[i]);
        dstRegions[i] = new Mat(*destination, *childs[i]);
    }
    cout << "[" << node->region->height << " x " << node->region->width <<"]"<< endl;
    double wholeMean = mean(*image).val[0];
    Scalar regionMean, regionVar;
    cout << "whole Mean: " << wholeMean << endl;
    for(int i=0; i<4; i++){
        meanStdDev(*imageRegions[i], regionMean, regionVar);
        if(abs(regionVar.val[0])<threshold){
            node->childs[i]=NULL;
            dstRegions[i]->setTo(Scalar(round(regionMean.val[0])));
            delete childs[i];
        }else{
            node->childs[i] = new QuadNode;
            node->childs[i]->region = childs[i];
            splitImage(imageRegions[i], dstRegions[i], node->childs[i], threshold);
        }
    }
    for(int i=0; i<4; i++){
        delete imageRegions[i];
        delete dstRegions[i];
    }
}

int main()
{
    Mat source = imread(IMAGENAME,CV_LOAD_IMAGE_GRAYSCALE);
    if(source.empty()){
        cerr << "ERROR! image not loaded" << endl;
        return -1;
    }
    Mat resultColorBased;
    Mat resultRegionGrowing;
    int optimalThreshold = findOptimalThreshold(source);
    cout << optimalThreshold << endl;
    binarization(source, resultColorBased, optimalThreshold);
    regionGrowing(source, resultRegionGrowing, 30);

    Mat convertedResult;
    resultRegionGrowing.convertTo(convertedResult, CV_8UC1);

    QuadNode *root = new QuadNode;
    Rect *mainRegion = new Rect(0,0,source.cols, source.rows);
    root->region = mainRegion;
    Mat splitImagedst(source.size(), CV_8UC1, Scalar(0));
    splitImage(&source, &splitImagedst, root, 25);


    imwrite("../img/result_binarization_man.png", resultColorBased);
    cvNamedWindow("Result Binarization",2);
    imshow("Result Binarization",resultColorBased);
    cvNamedWindow("Result Region-Growing",2);
    imshow("Result Region-Growing",convertedResult);
    cvNamedWindow("Result Split",2);
    imshow("Result Split",splitImagedst);
    cvNamedWindow("Source",2);
    imshow("Source",source);
    waitKey();
    return 0;
}

