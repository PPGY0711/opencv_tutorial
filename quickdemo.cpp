//
// Created by Guanyan Peng on 2022/5/3.
// Last Update date: 2022/5/5
//

#include "quickdemo.h"
#include <opencv2/dnn.hpp>

void QuickDemo::colorSpaceDemo(Mat &image) {
    Mat gray,hsv;
    cvtColor(image, hsv, COLOR_BGR2HSV);
    // H 0~180（颜色）,S（饱和度）,V(亮度）
    cvtColor(image, gray, COLOR_BGR2GRAY);
    imshow("HSV", hsv);
    imshow("GRAY", gray);
//    imwrite("../test/hsv.png", hsv);
//    imwrite("../test/gray.png", gray);
}

void QuickDemo::matCreationDemo(Mat &image) {
    Mat dst_cl, dst_cp;
    dst_cl = image.clone();
    image.copyTo(dst_cp);
    // 创建空白图像，有zeros()和ones()两种创建方式，
    // 注意ones在创建单通道图的时候正确，但是在创建多通道图的时候只会给第一个通道赋值为1，其余通道赋值为0
    Mat m3 = Mat::zeros(Size(8,8), CV_8UC1);
    std::cout<< "width: " << m3.cols << ", height: " << m3.rows << ", channels: " << m3.channels() <<std::endl;
    std::cout << m3 << std::endl;
    std:: cout << "-----------" << std::endl;
    // 注意这个m4通道数是3，那么一个像素点包含3个通道的值，即一个像素点有3个值来描述
    Mat m4 = Mat::zeros(Size(8,8), CV_8UC3);
    std::cout<< "width: " << m4.cols << ", height: " << m4.rows << ", channels: " << m4.channels() << std::endl;
    std::cout << m4 << std::endl;
    // width: 8, height: 8, channels: 3
    Mat m5 = Mat::ones(Size(8,8), CV_8UC3);
    std::cout<< "width: " << m5.cols << ", height: " << m5.rows << ", channels: " << m5.channels() << std::endl;
    std::cout << m5 << std::endl;
    // width: 8, height: 8, channels: 3

    // 创建图像并给每个通道赋值
    Mat m6 = Mat::zeros(Size(64,64), CV_8UC3);
    m6 = Scalar(255,128,129); // 给不同通道赋值，比如3通道则分别对应B\G\R
    std::cout<< "width: " << m6.cols << ", height: " << m6.rows << ", channels: " << m6.channels() << std::endl;
    std::cout << m6 << std::endl;
    imshow("m6", m6);

    Mat m7 = m6;
    m7 = Scalar(0,255,255);
    imshow("m6 after change", m6);
    imshow("m7", m6);

    Mat m8 = m6.clone();
    m8 = Scalar(255,0,0);
    imshow("m8", m8);
    imshow("m6 after clone", m6);
}


void QuickDemo::pixelVisitDemo(Mat &image) {
    int w = image.cols;
    int h = image.rows;
    int c = image.channels();
    if(c == 1) std::cout << "灰度图" << std::endl;
    else std::cout << "彩色图,且通道数为：" << c << std::endl;
    // 在++操作的时候使用++i比i++少一次底层的拷贝
//    for(int row = 0; row < h; ++row){
//        uchar* currentRow = image.ptr<uchar>(row);
//        for(int col = 0; col < w; ++col){
//            if(c == 1){//单通道灰度图像
//                int pv = *currentRow;
//                *currentRow++ = 255-pv; // 先赋值后++
//            }else if(c == 3){//彩色图像
//                *currentRow++ = 255-*currentRow; // 先赋值后++
//                *currentRow++ = 255-*currentRow; // 先赋值后++
//                *currentRow++ = 255-*currentRow; // 先赋值后++
//            }
//        }
//    }
    for(int row = 0; row < h; ++row){
        uchar* currentRow = image.ptr<uchar>(row);
        for(int col = 0; col < w; ++col){
            if(c == 1){//单通道灰度图像
                // 原来是存的uchar类型，这里会进行隐式转换
                int pv = image.at<uchar>(row,col);
                // 黑白转换,即反相
                image.at<uchar>(row,col) = 255-pv;
                // 这里如果用1000-pv，那么得到的值会溢出，后面还会有Mat中存储类型的转换操作要学
            }else if(c == 3){//彩色图像
                // Vec3b中每一个通道都存的uchar
                Vec3b bgr = image.at<Vec3b>(row,col);
                image.at<Vec3b>(row,col)[0] = 255-bgr[0];
                image.at<Vec3b>(row,col)[1] = 255-bgr[1];
                image.at<Vec3b>(row,col)[2] = 255-bgr[2];

            }
        }
    }
    imshow("pixel read and write", image);
}

void QuickDemo::pixelOperatorsDemo(Mat &image) {
    Mat dst;
    // 使用标量 直接做简单的加减乘除
    // 加减
    dst = image - Scalar (50,50,50);
    imshow("+ or - operator", dst);
    // 乘除
    dst = image / Scalar (2,2,2);
    imshow("/ operator", dst);
    Mat m = Mat::zeros(image.size(), image.type());
    m = Scalar(50,50,50);
    multiply(m, image, dst);
    imshow("* operator", dst);

    dst = image * 2; // 不能Scalar(2,2,2)，会报错
    imshow("* operator 2", dst);

    // 自己实现add
    dst = Mat::zeros(image.size(), image.type());
    int w = image.cols, h = image.rows, c = image.channels();
    for(int row = 0; row < h; ++row){
        for(int col = 0; col < w; ++col){
            Vec3b p1 = image.at<Vec3b>(row,col);
            Vec3b p2 = m.at<Vec3b>(row,col);
            dst.at<Vec3b>(row,col)[0] = saturate_cast<uchar>(p1[0] + p2[0]);
            dst.at<Vec3b>(row,col)[1] = saturate_cast<uchar>(p1[1] + p2[1]);
            dst.at<Vec3b>(row,col)[2] = saturate_cast<uchar>(p1[2] + p2[2]);
        }
    }
    imshow("custom add func", dst);
    m = Scalar(5,5,5);
    // 调用opencv的库函数
    add(image,m,dst);
    imshow("official add func", dst);
    subtract(image,m,dst);
    imshow("official subtract func", dst);
    divide(image,m,dst);
    imshow("official divide func", dst);
    multiply(image,m,dst);
    imshow("official multiply func", dst);
}


static void on_lightness(int pos, void* userdata){
    Mat image = *((Mat*)userdata);
//    std::cout << "pos value: " << pos << std::endl;
    Mat m = Mat::zeros(image.size(), image.type());
    Mat dst = Mat::zeros(image.size(), image.type());
    // 融合两张图
    addWeighted(image, 1.0, m, 0.0, pos, dst);
    //变亮
    //add(image, m, dst);
    //变暗
    //subtract(image, m, dst);
    imshow("Lightness&Contrast change", dst);
}

static void on_contrast(int pos, void* userdata){
    Mat image = *((Mat*)userdata);
//    std::cout << "pos value: " << pos << std::endl;
    Mat m = Mat::zeros(image.size(), image.type());
    Mat dst = Mat::zeros(image.size(), image.type());
    double contrast = pos/100.0;
    // 融合两张图
    addWeighted(image, contrast, m, 0.0, 0, dst);
    //变亮
    //add(image, m, dst);
    //变暗
    //subtract(image, m, dst);
    imshow("Lightness&Contrast change", dst);
}

void QuickDemo::trackBarDemo(Mat& image){
    namedWindow("Lightness&Contrast change", WINDOW_AUTOSIZE);
    int max_value = 100;
    int lightness = 0;
    createTrackbar("Lightness Value Bar", "Lightness&Contrast change", nullptr, 100, on_lightness, &image);
    setTrackbarPos("Lightness Value Bar", "Lightness&Contrast change", lightness);
    //参数依次是 bar的名称，window名称，跟踪的值的地址（因为要随时调整），取值范围（最大值），回调函数callback, 传入的userdata（void *）
    //之前传&lightness会报错，可能是opencv版本问题，因此这里根据错误提示传入了空指针，值的跟踪通过on_track的两个传入参数来获取
    //on_track(50,&image); //本来这种写法，可能是编译器认为不安全，并且这样写的程序会报段错误，所以采用了void* userdata直接在create的时候传参（图片数据）。
    createTrackbar("Contrast Value Bar", "Lightness&Contrast change", nullptr, 200, on_contrast, &image);
    setTrackbarPos("Contrast Value Bar", "Lightness&Contrast change", 100);
    imshow("Lightness&Contrast change", image);
}

void QuickDemo::keyDemo(Mat &image) {
    Mat dst = Mat::zeros(image.size(), image.type());
    while(true){
        // 在循环中不断监听键盘操作
        // 每次等待100毫秒再做响应，这里是为了演示把间隔时间设置比较长，在做视频分析的时候就要注意设置一个小的值，比如waitKey(1)
        int c = waitKey(100);
        if(c == 13|| c==27){
            //13是回车，27是esc
            break;
        }
        if(c == 49 || c == 50 || c == 51){
            //Key #1
            std::cout << "you input key: " << (char)c << std::endl;
            if(c == 49){
                // 按1变灰度
                cvtColor(image, dst, COLOR_BGR2GRAY);
            }else if(c == 50){
                // 按2变hsv
                cvtColor(image, dst, COLOR_BGR2HSV);
            }else{
                // 按3提高亮度
                // 注意这一步在add的时候没有写成add(image,dst,dst)，避免在切换过程中dst的变化导致计算出错。
                add(image, Scalar(50,50,50), dst);
            }
            imshow("Key Demo", dst);
        }
    }
}

void QuickDemo::colorStyleDemo(Mat &image) {
    Mat dst = Mat::zeros(image.size(), image.type());
    int colorMap[]={
        COLORMAP_AUTUMN,
        COLORMAP_BONE,
        COLORMAP_JET,
        COLORMAP_WINTER,
        COLORMAP_RAINBOW,
        COLORMAP_OCEAN,
        COLORMAP_SUMMER,
        COLORMAP_SPRING,
        COLORMAP_COOL,
        COLORMAP_HSV,
        COLORMAP_PINK,
        COLORMAP_HOT,
        COLORMAP_PARULA,
        COLORMAP_MAGMA,
        COLORMAP_INFERNO,
        COLORMAP_PLASMA,
        COLORMAP_VIRIDIS,
        COLORMAP_CIVIDIS,
        COLORMAP_TWILIGHT,
        COLORMAP_TWILIGHT_SHIFTED,
        COLORMAP_TURBO,
        COLORMAP_DEEPGREEN
    };
    int index = 0;
    while(true){
//        std::cout << "here" << std::endl;
        int c = waitKey(1000);
        if(c==27 || c==13) break;
        // 可以给灰度图像填伪彩色
        applyColorMap(image, dst, colorMap[index]);
        index = (index+1)%21;
        imshow("颜色风格", dst);
    }
}

void QuickDemo::bitwiseDemo(Mat &image) {
    Mat m1 = Mat::zeros(Size(256,256), CV_8UC3);
    Mat m2 = Mat::zeros(Size(256,256), CV_8UC3);
    // rect(画矩形的背景图，Rect(左上角的坐标，宽度，高度），矩形颜色，
    // 线宽（小于0表示填充，大于0表示边框线宽绘制）,线绘制方式(因为直线所经过的点坐标有时候并不是整数，绘制的时候会产生锯齿
    // 因此有不同的绘制方式，比如说不管锯齿的话，有400和800的两种绘制方式，常用的有LINE_8，而如果管锯齿的影响的话，希望这个锯齿和周围的颜色有点交融，就是LINE_AA，即反锯齿的绘制模式),
    rectangle(m1, Rect(100,100,80,80), Scalar(255,255,0), -1, LINE_8,0);
    rectangle(m2, Rect(150,150,80,80), Scalar(0,255,255), -1, LINE_8,0);
    imshow("m1",m1);
    imshow("m2",m2);

    Mat dst;
    bitwise_and(m1,m2,dst);
    imshow("bitwise_and", dst);
    bitwise_or(m1,m2,dst);
    imshow("bitwise_or", dst);
    bitwise_xor(m1,m2,dst);
    imshow("bitwise_xor", dst);
    bitwise_not(m1,dst);
    imshow("bitwise_not", dst);
    Mat m3 = ~m1;
    imshow("~m1", m3);
}

void QuickDemo::channelsDemo(Mat &image) {
    std::vector<Mat> mv;
    std::vector<std::string> names = {"blue","green","red"};

    for(int i = 0; i < 3; i++){
        split(image, mv);
        Mat dst;
        mv[i] = 0;
        merge(mv,dst);
        imshow("eliminate: "+names[i], dst);
    }
    split(image, mv);
    Mat dst;
    mv[0] = 0;
    merge(mv, dst);
    int from_to[] = {0,2,1,1,2,0};
    mixChannels(&image, 1, &dst, 1, from_to, 3);
    imshow("通道混合", dst);
}

void QuickDemo::inRangeDemo(Mat &image) {
    Mat hsv;
    cvtColor(image, hsv, COLOR_BGR2HSV);
    Mat mask;
    // 中间两个颜色控制的是颜色转换范围，在HSV色彩空间中颜色是由H和S来控制的，更容易提取一些
    inRange(hsv, Scalar(35, 43, 46), Scalar(77, 255, 255), mask);
    imshow("Mask", mask);
    Mat redBack = Mat::zeros(image.size(), image.type());
    redBack = Scalar(40, 40, 200);
    bitwise_not(mask, mask); // 让人所在的像素不为0
    // 利用掩码图做不规则形状的提取
    image.copyTo(redBack, mask); // 如果设置了mask，那么在copyTo的时候只有在mask中不为0的像素点对应的位置会被copy到图像m上。
    imshow("after copy according to mask", redBack);
}

void QuickDemo::pixelStatisticDemo(Mat &image) {
    std::vector<Mat> mv;
    split(image, mv);
    double minv, maxv;
    Point minLoc, maxLoc;
    int cnt = 0;
    for(auto& m:mv){
        // 要求是单通道图的统计
        minMaxLoc(m, &minv, &maxv, &minLoc, &maxLoc, Mat());
        std::cout << "min value: " << minv << ", max value: " << maxv << std::endl;
        std::cout << "min point: " << minLoc << "and min point pixel: " << static_cast<int>(m.at<uchar>(minLoc.x, minLoc.y)) << ", max point: " << maxLoc << "and max point pixel: " << static_cast<int>(m.at<uchar>(maxLoc.x,maxLoc.y)) << std::endl;
        imshow(std::to_string(cnt+1), m);
        cnt++;
    }

    Mat mean, stddev;
    meanStdDev(image, mean, stddev, Mat());
    for(int i = 0; i < mean.rows; i++){
        std::cout << "mean[" << i << "]: " << mean.at<double>(i,0) << ", stddev[" << i << "]: " << stddev.at<double>(i,0) << std::endl;
    }
}

void QuickDemo::drawingDemo(Mat &image) {
    Rect rect;
    rect.x = 100;
    rect.y = 100;
    rect.width = 250;
    rect.height = 300;
    Mat bg = Mat::zeros(image.size(), image.type());
    rectangle(bg, rect, Scalar(0,0,255), -1, LINE_8, 0);
    circle(bg, Point(350,400), 50, Scalar(255,0,0), -1, LINE_8, 0);
    line(bg, Point(100,100), Point(350,400), Scalar(0,255,0), 2, LINE_AA,0);
    RotatedRect rrt;
    rrt.center = Point(200,200);
    // 通过Size规定椭圆的长短轴长度
    rrt.size = Size(200,300);
    rrt.angle = 30; // height顺时针转
    ellipse(bg, rrt, Scalar(0,255,255), 2, LINE_AA);
    Mat dst;
    // 图片叠加，通过设置不同的权重可以创建类似半透明的效果
    addWeighted(image,0.7,bg,0.3, 0, dst);
    imshow("drawing addWeighted", dst);
}

void QuickDemo::randomDrawingDemo() {
    Mat canvas = Mat::zeros(Size(512,512), CV_8UC3);
    int w = canvas.cols, h = canvas.rows;
    RNG rng(12345);
    while(true){
        int c = waitKey( 100);
        if(c == 27 || c == 13){
            break;
        }
        Point p1,p2;
        p1.x = rng.uniform(0, h);
        p1.y = rng.uniform(0, w);
        p2.x = rng.uniform(0, h);
        p2.y = rng.uniform(0, w);
        int b = rng.uniform(0,255);
        int g = rng.uniform(0,255);
        int r = rng.uniform(0,255);
        canvas = Scalar(0,0,0);
        line(canvas, p1, p2, Scalar(b,g,r), 1, LINE_AA, 0);
        imshow("random Line", canvas);
    }
}

void QuickDemo::polylineDrawingDemo() {
    Mat canvas = Mat::zeros(Size(512,512), CV_8UC3);
    std::vector<Point> pts;
    Point p1(100,100);
    Point p2(350,100);
    Point p3(450,280);
    Point p4(320,450);
    Point p5(80,400);
    pts.emplace_back(p1);
    pts.emplace_back(p2);
    pts.emplace_back(p3);
    pts.emplace_back(p4);
    pts.emplace_back(p5);

    // 填充多边形
    //fillPoly(canvas, pts, Scalar(255,0,0),LINE_8,0);
    // 这个函数不能进行填充，只能根据vector点集进行多边形绘制，thickness必须大于0
    //polylines(canvas, pts, true, Scalar(0,0,255),2, LINE_AA,0);

    // drawContours()绘制或者填充轮廓，可以绘制或填充多个轮廓
    std::vector<std::vector<Point>> contours;
    contours.emplace_back(pts);
    // contourIndex指定填充哪一个轮廓，若为-1则一次性填充所有的轮廓
    // thickness同样是大于0为线宽，小于0为填充
    drawContours(canvas, contours, 0, Scalar(255,0,0), -1);
    imshow("polyline drawing", canvas);
}


Point sp(-1,-1), ep(-1,-1);
Mat temp;
static void on_mouse_draw(int event, int x, int y, int flags, void* userdata){
    Mat image = *(Mat*)userdata;
    if(event == EVENT_LBUTTONDOWN) {
        // 左键按下
        sp.x = x;
        sp.y = y;
        temp.copyTo(image);
        std::cout << "start point: " << sp << std::endl;
    }
    else if(event == EVENT_LBUTTONUP){
        ep.x = min(x, image.cols);
        ep.y = min(y, image.rows);
        int dx = std::abs(ep.x-sp.x), dy = std::abs(ep.y-sp.y);
        Rect box(std::min(sp.x, std::min(image.rows, ep.x)), std::min(sp.y,std::min(image.cols, ep.y)), dx, dy);
        rectangle(image, box, Scalar(0,0,255), 2, 8, 0);
        // 绘制圆
//        int cx = (sp.x+ep.x)>>1, cy = (sp.y+ep.y)>>1;
//        double radius = std::sqrt(dx*dx+dy*dy)/2;
        temp.copyTo(image);
        rectangle(image, box, Scalar(0,0,255), 2, 8, 0);
//        circle(image, Point(cx,cy), (int)radius, Scalar(255,0,0), 2, LINE_8,0);
        imshow("mouse drawing", image);
        imshow("RoI Area", temp(box));
        // reset为下一次绘制做好准备
        sp.x = -1;
        sp.y = -1;
    }
    else if(event == EVENT_MOUSEMOVE){
        if(sp.x > 0 && sp.y > 0){
            ep.x = min(x, image.cols);
            ep.y = min(y, image.rows);
            int dx = std::abs(ep.x-sp.x), dy = std::abs(ep.y-sp.y);
            Rect box(std::min(sp.x, ep.x), std::min(sp.y, ep.y), dx, dy);
//            int cx = (sp.x+ep.x)>>1, cy = (sp.y+ep.y)>>1;
//            double radius = std::sqrt(dx*dx+dy*dy)/2;
//            imshow("mouse drawing", image);
            temp.copyTo(image);
//            circle(image, Point(cx,cy), (int)radius, Scalar(255,0,0), 2, LINE_8,0);
            rectangle(image, box, Scalar(0,0,255), 2, 8, 0);
            imshow("mouse drawing", image);
        }
    }
}

void QuickDemo::mouseDrawingDemo(Mat &image) {
    namedWindow("mouse drawing", WINDOW_AUTOSIZE);
    setMouseCallback("mouse drawing", on_mouse_draw, &image);
    imshow("mouse drawing", image);
    temp = image.clone();
}

void QuickDemo::normDemo(Mat &image) {
    Mat dst;
    // 把image的数据类型转换成浮点类型
    std::cout << image.type() << std::endl; // CV_8UC3
    // 正常显示
    imshow("original", image);
    image.convertTo(dst, CV_32F);
    // 如果是浮点数显示的话，值必须在0~1之间，因为显示的时候会把浮点数*255，所以必须归一化
    // imwrite不支持浮点数数据
    // 不能正常显示
    imshow("before normalize", dst);
    std::cout << dst.type() << std::endl; // CV_32F
    // 用转了浮点数的图片进行归一化计算
    normalize(dst, dst, 1.0, 0, NORM_MINMAX);
    std::cout << dst.type() << std::endl; // CV_32F
    // 正常显示
    imshow("normalize", dst);

    Mat ct;
    multiply(dst, 255, ct); // 乘Scalar(255,255,255)也行
    // 不能正常显示
    imshow("multiply 255", ct);
    ct.convertTo(ct, CV_8UC3);
    // 正常显示
    imshow("convert to 8UC3", ct);
}

void QuickDemo::resizeDemo(Mat &image) {
    Mat zin,zout;
    int h = image.rows, w = image.cols;
    resize(image, zout, Size(w/2,h/2),0,0, INTER_LINEAR);
    imshow("zoom out", zout);
    resize(image, zin, Size(w*2,h*2),0,0, INTER_LINEAR);
    imshow("zoom in", zin);
}

void QuickDemo::flipDemo(Mat &image) {
    Mat dst;
    flip(image, dst, 0); // 以x为轴翻转,上下
    imshow("flip x-axis", dst);
    flip(image, dst, 1); // 以y为轴翻转，左右
    imshow("flip y-axis", dst);
    flip(image, dst, -1); // 以z为轴翻转，旋转180°（原点对称）
    imshow("flip z-axis", dst);
}

void QuickDemo::rotateDemo(Mat &image) {
    Mat dst,M;
    int w = image.cols, h = image.rows;
    M = getRotationMatrix2D(Point(w/2,h/2), 45, 1.0); //scale参数可以控制放缩倍数
    double cos = abs(M.at<double>(0,0)), sin = abs(M.at<double>(0,1));
    int nw = cos*w+sin*h, nh = sin*w+cos*h;
    // 求取旋转之后图片的实际大小（视野里如果包含所有图像的话应该是多大）
    // 宽高有变化，中心会偏移，这里是调整中心的位置
    M.at<double>(0,2) += (nw/2-w/2);
    M.at<double>(1,2) += (nh/2-h/2);
    // 这里修改了旋转之后产生的黑边背景色为蓝色
    warpAffine(image, dst, M, Size(nw,nh), INTER_LINEAR, 0, Scalar(255,0,0));
    imshow("rotate", dst);
}

void QuickDemo::videoDemo() {
//    VideoCapture capture(0);
    VideoCapture capture("/Users/pengguanyan/Desktop/myown/练习曲/彩虹练习3.mp4");
    int frameWidth = capture.get(CAP_PROP_FRAME_WIDTH);
    int frameHeight = capture.get(CAP_PROP_FRAME_HEIGHT);
    int count = capture.get(CAP_PROP_FRAME_COUNT);
    double fps = capture.get(CAP_PROP_FPS);
    std::cout << "frame width: " << frameWidth << ", frameHeight: " << frameHeight << ", count: " << count << ", fps: " << fps << std::endl;
    VideoWriter writer("../test/test.mp4",capture.get(CAP_PROP_FOURCC), fps, Size(frameWidth, frameHeight), true);

    Mat frame;
    while(true){
        capture.read(frame);
        if(frame.empty()){
            break;
        }
        // TODO: do sth.
        flip(frame, frame, 1);
        colorSpaceDemo(frame);
        // 保存视频（没有音频），实测保存下来的视频帧率一样的并不是卡顿的
        writer.write(frame);
        imshow("frame", frame);
        int c = waitKey(10);
        if(c == 13 || c == 27) break;
    }
    // release资源
    capture.release();
}

static void histogramCalculation(Mat& image, Mat& histImage){
    std::vector<Mat> bgr_plane;
    split(image, bgr_plane);
    const int channels[1] = {0};
    const int bins[1] = {256};
    // 这个区间可能是左闭右开的
    float hranges[2] = {0, 256};
    const float* ranges[1] = {hranges};
    Mat b_hist,g_hist,r_hist;
    // 计算B、G、R三通道的直方图
    // calcHist这个函数可以用于多张图片的直方图计算，只是这里的nimages, channels, dims等参数要修改
    calcHist(&bgr_plane[0], 1, 0,Mat(), b_hist, 1, bins, ranges);
    calcHist(&bgr_plane[1], 1, 0,Mat(), g_hist, 1, bins, ranges);
    calcHist(&bgr_plane[2], 1, 0,Mat(), r_hist, 1, bins, ranges);
    // b_hist/g_hist/r_hist都是一维的，长度为bins的数量
    std::cout << b_hist.size()  << std::endl;
    // 显示直方图
    int hist_w = 512, hist_h = 400;
    int bin_w = cvRound((double)(hist_w/bins[0]));
    histImage = Mat::zeros(hist_h, hist_w, CV_8UC3);
    // 归一化直方图数据(因为有的像素值可能频率非常高，这里归一化是为了不超过画布许可的范围，使用到的就是范围归一化）
    normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    // 绘制直方图曲线（感觉更像是折线）
    for(int i = 1; i < bins[0]; i++){
        // 这里用hist_h减去b_hist.at<float>(i)是做了一个坐标位置的转换，假设b_hist.at<float>(i)比较小，那么直方图的高度也是小的，
        // 但是画布上y值小的显示在高处（是反的），所以要用总的高度减去该值得到在画布上的坐标
        line(histImage, Point(bin_w*(i-1), hist_h- cvRound(b_hist.at<float>(i-1))),
             Point(bin_w*i, hist_h- cvRound(b_hist.at<float>(i))), Scalar(255,0,0),2,LINE_8,0);
        line(histImage, Point(bin_w*(i-1), hist_h- cvRound(g_hist.at<float>(i-1))),
             Point(bin_w*i, hist_h- cvRound(g_hist.at<float>(i))), Scalar(0,255,0),2,LINE_8,0);
        line(histImage, Point(bin_w*(i-1), hist_h- cvRound(r_hist.at<float>(i-1))),
             Point(bin_w*i, hist_h- cvRound(r_hist.at<float>(i))), Scalar(0,0,255),2,LINE_8,0);
    }
}

void QuickDemo::showHistogramDemo(Mat &image) {
    Mat histImage;
    histogramCalculation(image, histImage);
    // 显示直方图
    namedWindow("Histogram", WINDOW_AUTOSIZE);
    imshow("Histogram", histImage);
}

void QuickDemo::showHistogram2DDemo(Mat &image) {
    // 2D直方图
    Mat hsv, hs_hist;
    cvtColor(image, hsv, COLOR_BGR2HSV);
    int hbins = 30, sbins = 32; // hbins小于range的时候，会有多个像素值落进一个区间
    int hist_bins[] = {hbins, sbins};
    float h_range[] = {0,180};
    float s_range[] = {0,256};
    const float* hs_ranges[]  = {h_range, s_range};
    int hs_channels[] = {0,1};
    calcHist(&hsv, 1, hs_channels, Mat(), hs_hist, 2, hist_bins, hs_ranges, true, false);
    std::cout << hs_hist.size() << std::endl; //[32 x 30]
    double maxVal = 0;
    minMaxLoc(hs_hist, 0, &maxVal, 0,0);
    int scale = 10;
    // 这里初始化zeros的时候容易h和w搞反
    Mat hist2dImage = Mat::zeros(sbins*scale, hbins*scale, CV_8UC3);
    for(int s = 0; s < hbins; s++){
        for(int h = 0; h < sbins; h++){
            float binVal = hs_hist.at<float>(s,h);
            int intensity = cvRound(binVal*255/maxVal);
            rectangle(hist2dImage, Point(s*scale, h*scale),Point((s+1)*scale-1, (h+1)*scale-1),
                      Scalar::all(intensity),
                      -1);
        }
    }
    applyColorMap(hist2dImage, hist2dImage, COLORMAP_BONE);
    imshow("H-S Histogram", hist2dImage);
    imwrite("../test/hist2d.png", hist2dImage);
}

static Mat claheDeal(Mat &src, double ClipLimit = 40.0, int TilesGridSize = 8)
{
    Mat ycrcb = src.clone();
    std::vector<cv::Mat> channels;
    //YCrCb色彩空间：Y为颜色的亮度（luma）成分、而CB和CR则为蓝色和红色的浓度偏移量成分。
    //只有Y成分的图基本等同于彩色图像的灰度图。
    cv::cvtColor(ycrcb, ycrcb, cv::COLOR_BGR2YCrCb);
    cv::split(ycrcb, channels);
    cv::Mat clahe_img;
    //opencv源码中原型：  createCLAHE(double clipLimit = 40.0, Size tileGridSize = Size(8, 8));
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();

    clahe->setClipLimit(ClipLimit);
    clahe->setTilesGridSize(Size(TilesGridSize, TilesGridSize));
    clahe->apply(channels[0], clahe_img);
    channels[0].release();
    clahe_img.copyTo(channels[0]);
    cv::merge(channels, ycrcb);
    cv::cvtColor(ycrcb, ycrcb, cv::COLOR_YCrCb2BGR);
    return ycrcb;
}

void QuickDemo::histogramEqualizationDemo(Mat &image) {
    // 对灰度图像进行直方图均衡化
    Mat gray,dst;
    cvtColor(image, gray, COLOR_BGR2GRAY);
    equalizeHist(gray, dst);
    imshow("gray", gray);
    imshow("equalize hist", dst);

    // 对彩色图像进行直方图均衡化，通过图像色彩格式转换来实现，Y分量基本就是图像的灰阶版本
    Mat ycrcb, eq, histEq, histOri;
    histogramCalculation(image, histOri);
    imshow("Histogram Original image", histOri);
    std::vector<Mat> yuv;
    cvtColor(image, ycrcb, COLOR_BGR2YCrCb);
    split(ycrcb, yuv);
    // equalize Y channel
    equalizeHist(yuv[0],yuv[0]);
    // merge channels
    merge(yuv, eq);
    cvtColor(eq, eq, COLOR_YCrCb2BGR);
    imshow("Equalized Y", eq);
    histogramCalculation(eq, histEq);
    imshow("Equalized color image histogram", histEq);

    // 局部直方图均衡化
    double clipLimit = 4.0;
    int tilesGridSize = 8;
    Mat claheRes = claheDeal(image, clipLimit, tilesGridSize);
    imshow("c4.0 t8 CLAHE", claheRes);

    clipLimit = 8.0;
    Mat claheRes2 = claheDeal(image, clipLimit, tilesGridSize);
    imshow("c8.0 t8 CLAHE", claheRes2);

    clipLimit = 16.0;
    Mat claheRes3 = claheDeal(image, clipLimit, tilesGridSize);
    imshow("c16.0 t8 CLAHE", claheRes3);
}

void QuickDemo::blurDemo(Mat &image) {
    Mat dst;
    // 注意这里处理边缘是按照默认方式进行处理的,blur是一种平滑处理、空间滤波方式，可以支持一维、二维卷积
    blur(image, dst, Size(15,15), Point(-1,-1));
    imshow("blur", dst);
}

void QuickDemo::gaussianBlueDemo(Mat &image) {
    Mat dst;
    GaussianBlur(image, dst, Size(5,5), 15);
    imshow("Gaussian blur", dst);
}

void QuickDemo::biFilterDemo(Mat &image) {
    Mat dst;
    bilateralFilter(image, dst, 0, 100, 10);
    imshow("Gaussian blur", dst);
}

void QuickDemo::faceDetectionDemo() {
    // 效果很垃圾的感觉
    std::string rootDir = "../face_detector/";
    cv::dnn::Net net = cv::dnn::readNetFromTensorflow(rootDir + "opencv_face_detector_uint8.pb",rootDir + "opencv_face_detector.pbtxt");
    // 用摄像头拍自己
//    VideoCapture capture(0);
    VideoCapture capture(rootDir + "face_detector.avi");
    Mat frame;
    while(true){
        capture.read(frame);
        if(frame.empty()){
            break;
        }
        /*
         * scalefactor：图像像素数值normalize到0-1.0
         * Size，mean都是和模型有关的
         * SwapRB：是否要交换RB
         */
        Mat blob = dnn::blobFromImage(frame, 1.0, Size(300,300), Scalar(104,177,123), false, false);
        net.setInput(blob); // 出来的格式是NCHW
        Mat probs = net.forward(); // 4个维度，第一维是图像编号，img的index；第二个维度对应该img的批次，即batch index；第三个维度表示有多少个框；第四个维度表示每个框有7个值（7列）？

        // 解析结果
        Mat detectionMat(probs.size[2],probs.size[3], CV_32F, probs.ptr<float>());
        for(int i = 0; i < detectionMat.rows; i++){
            //解析检测到的框
            float confidence = detectionMat.at<float>(i,2);
            // 若置信度大于0.5则认为检测到了人脸
            if(confidence > 0.5){
                // 解析矩形框的坐标，预测出来的值是0~1的，必须乘以宽度才是正确的像素坐标
                int x1 = static_cast<int>(detectionMat.at<float>(i,3)*frame.cols);
                int y1 = static_cast<int>(detectionMat.at<float>(i,4)*frame.cols);
                int x2 = static_cast<int>(detectionMat.at<float>(i,5)*frame.cols);
                int y2 = static_cast<int>(detectionMat.at<float>(i,6)*frame.cols);
                std::cout << x1 << "," << y1 << ", " << x2 << ", " << y2 << std::endl;
                Rect box(x1,y1,x2-x1,y2-y1);
                rectangle(frame, box, Scalar(0,0,255),2,LINE_8, 0);
                imshow("face detection", frame);
            }
        }
        int c = waitKey(100);
        if(c == 27 || c == 13){
            break;
        }
    }
}
