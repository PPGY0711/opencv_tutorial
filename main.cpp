//
// Created by Guanyan Peng on 2022/5/3.
// Last Update date: 2022/5/5
//

#include "opencv2/opencv.hpp"
#include "quickdemo.h"
using namespace cv;
using namespace std;


// day1：图像读取与显示
//int main(int argc, char** argv){
//    Mat img = imread("../img/29.bmp", IMREAD_GRAYSCALE); // 加载成灰度图像
//    if(img.empty()){
//        cout << "could not load image" << endl;
//        return -1;
//    }
//    namedWindow("input", WINDOW_AUTOSIZE);
//    imshow("input", img);
//    waitKey(0);
//    destroyAllWindows();
//    return 0;
//}

// day2：图像色彩空间转换
//int main(int argc, char** argv){
//    Mat img = imread("../avatar.jpeg");
//    if(img.empty()){
//        cout << "could not load image" << endl;
//        return -1;
//    }
//    namedWindow("input");
//    imshow("input", img);
//    QuickDemo qd;
//    qd.colorSpaceDemo(img);
//    waitKey(0);
//    destroyAllWindows();
//    return 0;
//}

// day3：图像对象Mat的创建与使用
//int main(int argc, char** argv){
//    Mat img = imread("../avatar.jpeg");
//    if(img.empty()) {
//        cout << "could not load image" << endl;
//        return -1;
//    }
//    imshow("input", img);
//    QuickDemo qd;
//    qd.matCreationDemo(img);
//    waitKey(0);
//    destroyAllWindows();
//    return 0;
//}

// day4：图像像素的访问
//int main(int argc, char** argv){
//    Mat img = imread("../test/gray.png");
//    if(img.empty()) {
//        cout << "could not load image" << endl;
//        return -1;
//    }
//    imshow("input", img);
//    QuickDemo qd;
//    qd.pixelVisitDemo(img);
//    waitKey(0);
//    destroyAllWindows();
//    return 0;
//}

// day5：图像像素的算术操作
//int main(int argc, char** argv){
//    Mat img = imread("../avatar.jpeg");
//    if(img.empty()) {
//        cout << "could not load image" << endl;
//        return -1;
//    }
//    imshow("input", img);
//    QuickDemo qd;
//    qd.pixelOperatorsDemo(img);
//    waitKey(0);
//    destroyAllWindows();
//    return 0;
//}


// day6：trackBar使用
//int main(int argc, char** argv){
//    Mat img = imread("../avatar.jpeg");
//    if(img.empty()) {
//        cout << "could not load image" << endl;
//        return -1;
//    }
//    imshow("input", img);
//    QuickDemo qd;
//    qd.trackBarDemo(img);
//    waitKey(0);
//    destroyAllWindows();
//    return 0;
//}

// day7：键盘响应监听
//int main(int argc, char** argv){
//    Mat img = imread("../test/test.jpeg");
//    if(img.empty()) {
//        cout << "could not load image" << endl;
//        return -1;
//    }
//    imshow("input", img);
//    QuickDemo qd;
//    qd.colorStyleDemo(img);
//    waitKey(0);
//    destroyAllWindows();
//    return 0;
//}

// day8：ColorMap
//int main(int argc, char** argv){
//    Mat img = imread("../test/test.jpeg");
//    if(img.empty()) {
//        cout << "could not load image" << endl;
//        return -1;
//    }
//    imshow("input", img);
//    QuickDemo qd;
//    qd.colorStyleDemo(img);
//    waitKey(0);
//    destroyAllWindows();
//    return 0;
//}

// day9：图像像素逻辑操作
//int main(int argc, char** argv){
//    Mat img = imread("../test/test.jpeg");
//    if(img.empty()) {
//        cout << "could not load image" << endl;
//        return -1;
//    }
//    imshow("input", img);
//    QuickDemo qd;
//    qd.bitwiseDemo(img);
//    waitKey(0);
//    destroyAllWindows();
//    return 0;
//}

// day10：图像通道分离与合并
//int main(int argc, char** argv){
//    Mat img = imread("../avatar.jpeg");
//    if(img.empty()) {
//        cout << "could not load image" << endl;
//        return -1;
//    }
//    imshow("input", img);
//    QuickDemo qd;
//    qd.channelsDemo(img);
//    waitKey(0);
//    destroyAllWindows();
//    return 0;
//}

// day11：图像色彩空间转换
//int main(int argc, char** argv){
//    Mat img = imread("../test/green.jpeg");
//    if(img.empty()) {
//        cout << "could not load image" << endl;
//        return -1;
//    }
//    imshow("input", img);
//    QuickDemo qd;
//    qd.inRangeDemo(img);
//    waitKey(0);
//    destroyAllWindows();
//    return 0;
//}

// day12：图像像素统计
//int main(int argc, char** argv){
//    Mat img = imread("../avatar.jpeg");
//    if(img.empty()) {
//        cout << "could not load image" << endl;
//        return -1;
//    }
//    imshow("input", img);
//    QuickDemo qd;
//    qd.pixelStatisticDemo(img);
//    waitKey(0);
//    destroyAllWindows();
//    return 0;
//}

// day13：几何形状绘制
//int main(int argc, char** argv){
//    Mat img = imread("../avatar.jpeg");
//    if(img.empty()) {
//        cout << "could not load image" << endl;
//        return -1;
//    }
//    imshow("input", img);
//    QuickDemo qd;
//    qd.drawingDemo(img);
//    waitKey(0);
//    destroyAllWindows();
//    return 0;
//}

// day14：随机几何形状绘制
//int main(int argc, char** argv){
//    QuickDemo qd;
//    qd.randomDrawingDemo();
//    waitKey(0);
//    destroyAllWindows();
//    return 0;
//}

// day15：多边形填充与绘制
//int main(int argc, char** argv){
//    QuickDemo qd;
//    qd.polylineDrawingDemo();
//    waitKey(0);
//    destroyAllWindows();
//    return 0;
//}

// day16：鼠标操作与响应
//int main(int argc, char** argv){
//    Mat img = imread("../test/test.jpeg");
//    if(img.empty()) {
//    cout << "could not load image" << endl;
//    return -1;
//    }
//    QuickDemo qd;
//    qd.mouseDrawingDemo(img);
//    waitKey(0);
//    destroyAllWindows();
//    return 0;
//}

// day17：图像像素类型转换与归一化
//int main(int argc, char** argv){
//    Mat img = imread("../test/test.jpeg");
//    if(img.empty()) {
//    cout << "could not load image" << endl;
//    return -1;
//    }
//    QuickDemo qd;
//    qd.normDemo(img);
//    waitKey(0);
//    destroyAllWindows();
//    return 0;
//}

// day18：图像放缩与插值
//int main(int argc, char** argv){
//    Mat img = imread("../test/test.jpeg");
//    if(img.empty()) {
//        cout << "could not load image" << endl;
//        return -1;
//    }
//    QuickDemo qd;
//    qd.resizeDemo(img);
//    waitKey(0);
//    destroyAllWindows();
//    return 0;
//}

// day19：图像翻转+旋转
//int main(int argc, char** argv){
//    Mat img = imread("../test/test.jpeg");
//    if(img.empty()) {
//        cout << "could not load image" << endl;
//        return -1;
//    }
//    imshow("origin", img);
//    QuickDemo qd;
//    qd.flipDemo(img);
//    qd.rotateDemo(img);
//    waitKey(0);
//    destroyAllWindows();
//    return 0;
//}

// day20：视频操作
//int main(int argc, char** argv){
//    Mat img = imread("../test/test.jpeg");
//    if(img.empty()) {
//        cout << "could not load image" << endl;
//        return -1;
//    }
//
//    imshow("origin", img);
//    QuickDemo qd;
//    qd.videoDemo();
//    waitKey(0);
//    destroyAllWindows();
//    return 0;
//}

// day21：图像直方图、2D直方图
//int main(int argc, char** argv){
//    Mat img = imread("../avatar.jpeg");
//    if(img.empty()) {
//        cout << "could not load image" << endl;
//        return -1;
//    }
//
//    imshow("origin", img);
//    QuickDemo qd;
//    qd.showHistogramDemo(img);
//    qd.showHistogram2DDemo(img);
//    waitKey(0);
//    destroyAllWindows();
//    return 0;
//}

// day22：直方图均衡化
//int main(int argc, char** argv){
//    Mat img = imread("../test/test.jpeg");
//    if(img.empty()) {
//        cout << "could not load image" << endl;
//        return -1;
//    }
//
//    imshow("origin", img);
//    QuickDemo qd;
//    qd.histogramEqualizationDemo(img);
//    waitKey(0);
//    destroyAllWindows();
//    return 0;
//}

// day23:图像卷积操作
//int main(int argc, char** argv){
//    Mat img = imread("../test/test.jpeg");
//    if(img.empty()) {
//        cout << "could not load image" << endl;
//        return -1;
//    }
//
//    imshow("origin", img);
//    QuickDemo qd;
//    qd.blurDemo(img);
//    waitKey(0);
//    destroyAllWindows();
//    return 0;
//}

// day24:高斯模糊
//int main(int argc, char** argv){
//    Mat img = imread("../avatar.jpeg");
//    if(img.empty()) {
//        cout << "could not load image" << endl;
//        return -1;
//    }
//
//    imshow("origin", img);
//    QuickDemo qd;
//    qd.gaussianBlueDemo(img);
//    qd.biFilterDemo(img);
//    waitKey(0);
//    destroyAllWindows();
//    return 0;
//}

// day25:人脸检测Demo（借助OpenCV4的dnn模块）
int main(int argc, char** argv){
    QuickDemo qd;
    qd.faceDetectionDemo();
    waitKey(0);
    destroyAllWindows();
    return 0;
}