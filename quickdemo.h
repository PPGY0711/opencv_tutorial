//
// Created by Guanyan Peng on 2022/5/3.
//
#include "opencv2/opencv.hpp"
using namespace cv;
#ifndef QUICKDEMO_H
#define QUICKDEMO_H
class QuickDemo{
public:
    void colorSpaceDemo(Mat& image);
    void matCreationDemo(Mat& image);
    void pixelVisitDemo(Mat& image);
    void pixelOperatorsDemo(Mat& image);
    void trackBarDemo(Mat& image);
    void keyDemo(Mat& image);
    void colorStyleDemo(Mat& image);
    void bitwiseDemo(Mat& image);
    void channelsDemo(Mat& image);
    void inRangeDemo(Mat& image);
    void pixelStatisticDemo(Mat& image);
    void drawingDemo(Mat& image);
    void randomDrawingDemo();
    void polylineDrawingDemo();
    void mouseDrawingDemo(Mat& image);
    void normDemo(Mat& image);
    void resizeDemo(Mat& image);
    void flipDemo(Mat& image);
    void rotateDemo(Mat& image);
    void videoDemo();
    void showHistogramDemo(Mat& image);
    void showHistogram2DDemo(Mat& image);
};
#endif //QUICKDEMO_H
