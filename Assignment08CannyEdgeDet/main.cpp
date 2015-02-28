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

#define IMAGENAME "../img/a992.jpg"
#define HIGH_THRESHOLD 200
#define LOW_THRESHOLD 80

using namespace cv;
using namespace std;

short hgradient[3][3] = {{-1, 0, 1},{-2, 0, 2},{-1, 0, 1}};
short vgradient[3][3] = {{-1, -2, -1},{0, 0, 0},{1, 2, 1}};

short getPixelValue(Mat &source, int y, int x){
    if(x>=0 && y>=0 && x<source.cols*source.channels() && y<source.rows) return source.at<short>(y,x);
    else return 0;
}

short getPointGradient(Mat &source, int x, int y, bool horizontal){
    short result=0;
    short (*gradient)[3][3]; // pointer to array
    if(horizontal) gradient = &hgradient;
    else gradient = &vgradient;
    for(int dx=-1; dx<=1; dx++){
        for(int dy=-1; dy<=1; dy++){
            if(x+dx>=0 && y+dy>=0 && x+dx<source.cols && y+dy<source.rows){
                result+=(*gradient)[dx+1][dy+1]*(short)source.at<uchar>(y+dy,x+dx);
            }
        }
    }
    return result;
}

void getGradient(Mat &source, Mat &gradient, bool horizontal){
    gradient.create(source.size(), CV_16SC1);
    for(int x=0; x<source.cols; x++){
        for(int y=0; y<source.rows; y++){
            gradient.at<short>(y,x) = getPointGradient(source, x, y, horizontal);
        }
    }
}

void normalize(Mat &source){
    short max= *max_element(source.begin<short>(), source.end<short>());
    int nchannels = source.channels();
    cout << max << endl;
    for(int x=0; x<source.cols; x++){
        for(int y=0; y<source.rows; y++){
            for(int i=0; i<nchannels; i++){
                source.at<short>(y,nchannels*x+i)=255*(source.at<short>(y,nchannels*x+i))/max;
                //source.at<short>(y,nchannels*x+i)+=127;
            }

        }
    }
}

void computeGradient(Mat &image, Mat &gradient){
    gradient.create(image.size(), CV_16SC3);

    Mat hGradient;
    Mat vGradient;
    getGradient(image, hGradient,  true);
    getGradient(image, vGradient, false);

    short mod, gx, gy, region;
    float phase;
    for(int x=0; x<image.cols; x++){
        for(int y=0; y<image.rows; y++){
            gx = hGradient.at<short>(y,x);
            gy = vGradient.at<short>(y,x);
            mod = sqrt(pow(gx,2)+pow(gy,2));
            phase = atan2(gy, gx);
            region = floor(phase/M_PI*8);
            if(region == 0 || region == 7){ //yellow (red+green)
                gradient.at<short>(y,3*x+2) = mod;
                gradient.at<short>(y,3*x+1) = mod;
            }
            else if(region == 1 || region == 2) gradient.at<short>(y,3*x+1) = mod; //green
            else if(region == 3 || region == 4) gradient.at<short>(y,3*x) = mod; //blue
            else if(region == 5 || region == 6) gradient.at<short>(y,3*x+2) = mod; //red
        }
    }
}

bool isEdge(Mat &image, int x, int y){
    short red   = image.at<short>(y,3*x+2);
    short green = image.at<short>(y,3*x+1);
    short blue  = image.at<short>(y,3*x);
    /*
    if(red!=0 && green!=0){ // yellow
        if(getPixelValue(image,y,3*(x-1)+2)<=red && getPixelValue(image,y,3*(x+1)+2)<=red) return true;
    }else if(red!=0){ //red
        if(getPixelValue(image,y+1,3*(x+1)+2)<=red && getPixelValue(image,y-1,3*(x-1)+2)<=red) return true;
    }else if(green!=0){ //green
        if(getPixelValue(image,y+1,3*(x-1)+1)<=green && getPixelValue(image,y-1,3*(x+1)+1)<=green) return true;
    }else if(blue!=0){ //blue
        if(getPixelValue(image,y+1,3*x+2)<=blue && getPixelValue(image,y-1,3*x+2)<=blue) return true;
    }
    return false;
    */
    if(red!=0 && green!=0){ // yellow
        if(getPixelValue(image,y+1,3*x+2)<=red && getPixelValue(image,y-1,3*x+2)<=red){return true;}
        else return false;
    }else if(red!=0){ //red
        if(getPixelValue(image,y+1,3*(x-1)+2)<=red && getPixelValue(image,y-1,3*(x+1)+2)<=red) return true;
        else return false;
    }else if(green!=0){ //green
        if(getPixelValue(image,y+1,3*(x+1)+1)<=green && getPixelValue(image,y-1,3*(x-1)+1)<=green) return true;
        else return false;
    }else if(blue!=0){ //blue
        if(getPixelValue(image,y,3*(x-1))<=blue && getPixelValue(image,y,3*(x+1))<=blue) return true;
        else return false;
    }
    return false;


}

void nonEdgeSuppression(Mat &image){
    int num=0;
    for(int x=0; x<image.cols; x++){
        for(int y=0; y<image.rows; y++){
            if(!isEdge(image, x, y)){
                if(image.at<short>(y,3*x) + image.at<short>(y,3*x+1) + image.at<short>(y,3*x+2)!= 0){
                    image.at<short>(y,3*x)  = 0;
                    image.at<short>(y,3*x+1)= 0;
                    image.at<short>(y,3*x+2)= 0;
                }
            }
        }
    }
    cout << "Azzerati: " << num << endl;
}

bool nearEdges(Mat &image, int x, int y){
    for(int dx=-1; dx<=1; dx++){
        for(int dy=-1; dy<1; dy++){
            if(image.at<uchar>(y+dy,x+dx)==255) return true;
        }
    }
    return false;
}

void drawEdges(Mat &gradient, Mat &destination){
    short limit;
    for(int x=0; x<gradient.cols; x++){
        for(int y=0; y<gradient.rows; y++){
            if(nearEdges(destination, x, y)) limit = LOW_THRESHOLD;
            else limit = HIGH_THRESHOLD;
            if(abs(gradient.at<short>(y,x))>limit) destination.at<uchar>(y,x)=255;
        }
    }
}

short getDirection(Mat &gradient, int x, int y){
    short red   = gradient.at<short>(y,3*x+2);
    short green = gradient.at<short>(y,3*x+1);
    short blue  = gradient.at<short>(y,3*x);

    if(red!=0 && green!=0){ // yellow
        if(getPixelValue(gradient,y,3*(x-1)+2)<=red && getPixelValue(gradient,y,3*(x+1)+2)<=red) return 1;
    }else if(red!=0){ //red
        if(getPixelValue(gradient,y+1,3*(x+1)+2)<=red && getPixelValue(gradient,y-1,3*(x-1)+2)<=red) return 2;
    }else if(green!=0){ //green
        if(getPixelValue(gradient,y+1,3*(x-1)+1)<=green && getPixelValue(gradient,y-1,3*(x+1)+1)<=green) return 3;
    }else if(blue!=0){ //blue
        if(getPixelValue(gradient,y+1,3*x+2)<=blue && getPixelValue(gradient,y-1,3*x+2)<=blue) return 4;
    }
    return -1;
}

short getMagnitude(Mat &gradient, int x, int y){
    return max(max(gradient.at<short>(y,3*x),gradient.at<short>(y,3*x+1)),gradient.at<short>(y,3*x+2));
}

bool computeEdges(Mat &gradient, Mat &gradientEdges, Mat &destination){
    destination.create(gradient.size(), CV_8UC1);
    short magnitude, direction, pixel1Direction, pixel2Direction;;
    bool pixel1Taken, pixel2Taken, changed=false;
    for(int x=0; x<gradient.cols; x++){
        for(int y=0; y<gradient.rows; y++){
            magnitude = getMagnitude(gradientEdges, x, y);
            if(magnitude>=HIGH_THRESHOLD){
                if(destination.at<uchar>(y,x)!=255){
                    destination.at<uchar>(y,x)=255;
                    changed = true;
                }
            }else if(magnitude>=LOW_THRESHOLD){
                direction = getDirection(gradientEdges, x, y);
                switch(direction){
                case 1: //yellow (0°)
                    pixel1Direction=getDirection(gradientEdges, x+1, y);
                    pixel2Direction=getDirection(gradientEdges, x-1, y);
                    pixel1Taken=destination.at<uchar>(y, x+1) == 255;
                    pixel2Taken=destination.at<uchar>(y, x-1) == 255;
                    break;
                case 2: //red (135°)
                    pixel1Direction=getDirection(gradientEdges, x+1, y-1);
                    pixel2Direction=getDirection(gradientEdges, x-1, y+1);
                    pixel1Taken=destination.at<uchar>(y-1, x+1) == 255;
                    pixel2Taken=destination.at<uchar>(y+1, x-1) == 255;
                    break;
                case 3: //green (45°)
                    pixel1Direction=getDirection(gradientEdges, x+1, y+1);
                    pixel2Direction=getDirection(gradientEdges, x-1, y-1);
                    pixel1Taken=destination.at<uchar>(y+1, x+1) == 255;
                    pixel2Taken=destination.at<uchar>(y-1, x-1) == 255;
                    break;
                case 4: //blue (90°)
                    pixel1Direction=getDirection(gradientEdges, x, y+1);
                    pixel2Direction=getDirection(gradientEdges, x, y-1);
                    pixel1Taken=destination.at<uchar>(y+1, x) == 255;
                    pixel2Taken=destination.at<uchar>(y-1, x) == 255;
                    break;
                }
                if((pixel1Direction==direction && pixel1Taken) ||
                   (pixel2Direction==direction && pixel2Taken)){
                    if(destination.at<uchar>(y,x)!=255){
                        destination.at<uchar>(y,x) = 255;
                        changed=true;
                    }
                }

            }
        }
    }
    return changed;
}

int main()
{
    Mat source = imread(IMAGENAME,CV_LOAD_IMAGE_GRAYSCALE);
    if(source.empty()){
        cerr << "ERROR! image not loaded"<<endl;
        return -1;
    }
    Mat workImg;
    GaussianBlur(source, workImg, cv::Size(5, 5), 1.4);

    Mat gradient;
    computeGradient(workImg, gradient);

    Mat maxSuppression;
    gradient.copyTo(maxSuppression);
    nonEdgeSuppression(maxSuppression);

    Mat edges(source.size(), CV_8UC1, Scalar(0));
    //drawEdges(hGradient, edges);

    normalize(gradient);

    Mat resultGradient_edge;
    maxSuppression.convertTo(resultGradient_edge, CV_8UC3);

    Mat resultGradient;
    gradient.convertTo(resultGradient, CV_8UC3);
    bool changed;
    int niter = 0;
    do{
        niter++;
        changed=computeEdges(gradient, maxSuppression, edges);
        cout << "iteration: " << niter << endl;
    }while(changed);
    cvNamedWindow("Gradient-Edge",2);
    imshow("Gradient-Edge",resultGradient_edge);

    cvNamedWindow("Gradient",2);
    imshow("Gradient",resultGradient);

    cvNamedWindow("Edges",2);
    imshow("Edges",edges);

    cvNamedWindow("Source",2);
    imshow("Source",source);

    cvNamedWindow("Smoothed",2);
    imshow("Smoothed",workImg);
    waitKey();
    return 0;
}
