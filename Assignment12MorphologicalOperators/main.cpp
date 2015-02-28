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

#define IMAGENAME "../img/result_binarization_man.png"
#define PAR_EROSION 0
#define PAR_DILATION 1

using namespace std;
using namespace cv;

const Point *kernel1[] = {
    new Point(-1,0),
    new Point(0,1),
    new Point(1,0),
    new Point(0,-1),
    NULL
};

const Point *kernel2[] = {
    new Point(-1,-1),
    new Point(-1,0),
    new Point(-1,1),
    new Point(0,1),
    new Point(1,1),
    new Point(1,0),
    new Point(1,-1),
    new Point(0,-1),
    NULL
};

void morphologicalOperation(Mat &source, Mat &destination, int type, const Point *kernel[]){
    if(source.cols!=destination.cols || source.rows!=destination.rows){
        destination.create(source.size(), CV_8UC1);
    }
    int i;
    Point actualPoint;
    uchar tmpValue;
    for(actualPoint.x=0; actualPoint.x<source.cols; actualPoint.x++){
        for(actualPoint.y=0; actualPoint.y<source.rows; actualPoint.y++){
          i=0;
          destination.at<uchar>(actualPoint) = type*255;
          while(kernel[i]!=NULL){
              if(actualPoint.x>=0 && actualPoint.y>=0 && actualPoint.x<source.cols && actualPoint.y<source.rows){
                  tmpValue = source.at<uchar>(actualPoint+*kernel[i]);
                  if(((bool)tmpValue)!=((bool)type)){
                      // if tmpValue and type aren't both set or unset then the result will be the opposite of type
                      destination.at<uchar>(actualPoint) = ((uchar)!((bool)type))*255;
                      break;
                  }
              }
              i++;
          }

        }
    }
}

void openingClosing(Mat &source, Mat &destination, const Point *kernel[], bool isclosing,int niter_dil=1, int niter_er=1){
    Mat tmp1;
    Mat tmp2;
    source.copyTo(tmp1);
    int iter1, iter2, niter1, niter2;
    if(isclosing){
        iter1 = PAR_DILATION;
        iter2 = PAR_EROSION;
        niter1 = niter_dil;
        niter2 = niter_er;
    }else{
        iter1 = PAR_EROSION;
        iter2 = PAR_DILATION;
        niter1 = niter_er;
        niter2 = niter_dil;
    }
    for(int i=0; i<niter1; i++){
        if(i%2==0) morphologicalOperation(tmp1, tmp2, iter1, kernel);
        else morphologicalOperation(tmp2, tmp1, iter1, kernel);
    }
    if((niter1-1)%2==0) tmp2.copyTo(tmp1);
    for(int i=0; i<niter2; i++){
        if(i%2==0) morphologicalOperation(tmp1, tmp2, iter2, kernel);
        else morphologicalOperation(tmp2, tmp1, iter2, kernel);
    }
    if((niter2-1)%2==0) tmp2.copyTo(destination);
    else tmp1.copyTo(destination);
}


int main()
{
    Mat source = imread(IMAGENAME,CV_LOAD_IMAGE_GRAYSCALE);
    if(source.empty()){
        cerr << "ERROR! image not loaded" << endl;
        return -1;
    }

    Mat destination;
    destination.create(source.size(), CV_8UC1);
    destination.setTo(Scalar(0));
    openingClosing(source, destination, kernel2, true, 0, 1);
    Mat contour = destination - source;

    Mat skeleton(source.size(), CV_8UC1, Scalar(0));

    Mat temp, eroded;
    bool done=false;
    int count=0;
    do
    {
      morphologicalOperation(source, eroded, PAR_DILATION, kernel1);
      morphologicalOperation(eroded, temp, PAR_EROSION, kernel1);
      subtract(source, temp, temp);
      bitwise_or(skeleton, temp, skeleton);
      eroded.copyTo(source);

      done = (countNonZero(source) == 0);
      count++;
    } while (!done);

    imwrite("../img/manichini_skel.png", skeleton);
    imwrite("../img/manichini_cont.png", contour);
    cvNamedWindow("source",2);
    imshow("source",contour);
    cvNamedWindow("skeleton",2);
    imshow("skeleton", skeleton);
    waitKey();
    return 0;
}

