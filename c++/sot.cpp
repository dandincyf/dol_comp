/*精跟踪：边缘质心法（270）*/
#include"opencv2/opencv.hpp"  
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>
using namespace cv;
#include <iostream>  
#include <opencv2/imgproc/types_c.h>
#include <chrono>
using namespace std::chrono;
using namespace std;

Mat newRingStrel(int ro, int ri);

//鼠标事件
Point initialPoint;
bool pointSelected = false;

float myDistance(const Point& p1, const Point& p2)
{
    return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
}

void showImg(Mat frame, chrono::steady_clock::time_point start)
{
    auto end = high_resolution_clock::now();
    duration<double, milli> ms_double = end - start;
    cout << "it took " << ms_double.count() << " ms" << endl;

    namedWindow("result", 0);
    resizeWindow("result", 640, 512);
    imshow("result", frame);

    waitKey(1);
}

//鼠标事件回调函数：选择跟踪目标
void onMouse(int event, int x, int y, int flags, void* param)
{
    switch (event)
    {
    case EVENT_LBUTTONDOWN:
        initialPoint = Point(x, y);
        pointSelected = true;
        break;
    }
}

int main()
{
    //j20240419_1_dark
    //j20240419_6_light
    VideoCapture video("I:/dolphin_dataset/video.mp4", IMREAD_GRAYSCALE);
    if (!video.isOpened())
        return -1;

    int frameCount = video.get(CAP_PROP_FRAME_COUNT);//获取帧数
    double FPS = video.get(CAP_PROP_FPS);//获取FPS
    Mat frame;//存储帧
    Mat result;//存储结果图像
    bool lightFlag = true;

    //Scharr滤波器相关变量
    Mat g_scharrGradient_X, g_scharrGradient_Y;
    Mat g_scharrAbsGradient_X, g_scharrAbsGradient_Y;
    Mat scharrImage;

    Mat binaryImg;

    int num = 0;

    for (int i = 0; i < frameCount; i++)
    {
        video >> frame;//读帧进frame
        namedWindow("00", 0);
        resizeWindow("00", 640, 512);
        imshow("00", frame);

        cvtColor(frame, frame, COLOR_BGR2GRAY);
        blur(frame, frame, Size(3, 3));
        //medianBlur(frame, frame, 3);

        //腐蚀
        //Mat b2 = Mat::ones(4, 4, CV_8UC1);
        //erode(frame, frame, b2);

        auto start = high_resolution_clock::now();
        cout << "min= " << (num / 20) / 60 << ",sec=" << (num / 20) % 60 << endl;
        num++;

        if (frame.empty())//对帧进行异常检测  
        {
            cout << "frame is empty!" << endl;
            break;
        }

        if (lightFlag == false)
        {
            frame = Scalar::all(255) - frame;
        }

        Rect roi_frame(frame.cols / 4, frame.rows / 4, frame.cols / 2, frame.rows / 2);
        Mat frame_littel = frame(roi_frame).clone();


        // 求 X方向梯度
        Scharr(frame_littel, g_scharrGradient_X, CV_16S, 1, 0, 1, 0, BORDER_DEFAULT);
        convertScaleAbs(g_scharrGradient_X, g_scharrAbsGradient_X);//计算绝对值，并将结果转换成8位
        // 求Y方向梯度
        Scharr(frame_littel, g_scharrGradient_Y, CV_16S, 0, 1, 1, 0, BORDER_DEFAULT);
        convertScaleAbs(g_scharrGradient_Y, g_scharrAbsGradient_Y);//计算绝对值，并将结果转换成8位
        // 合并梯度
        addWeighted(g_scharrAbsGradient_X, 0.5, g_scharrAbsGradient_Y, 0.5, 0, scharrImage);


        namedWindow("11", 0);
        resizeWindow("11", 640, 512);
        imshow("11", scharrImage);

        //用topN求threshold
        //int histSize = 256;
        //float range[] = { 0,256 };
        //const float* histRange = { range };
        //bool uniform = true, accumulate = false;
        //Mat hist;
        //calcHist(&scharrImage, 1, 0, Mat(), hist, 1, &histSize, &histRange,
        //    uniform, accumulate);
        //int max_n = 5000;
        //int sum = 0;
        //int light_threshold = -1;
        //for (int i = histSize - 1; i >= 0; i--)
        //{
        //    sum += hist.at<float>(i);
        //    if (sum > max_n)
        //    {
        //        light_threshold = i;
        //        break;
        //    }
        //}
        //cout << "Light threshod: " << light_threshold << endl;
        //threshold(scharrImage, binaryImg, light_threshold, 255, THRESH_BINARY);
        threshold(scharrImage, binaryImg, 30, 255, THRESH_BINARY);

        namedWindow("binaryImg", 0);
        resizeWindow("binaryImg", 640, 512);
        imshow("binaryImg", binaryImg);
        //imwrite("H:/data/FN20240708-/J20240711/binaryImg/binaryImg" + to_string(i) + ".jpg", binaryImg);


        Mat_<uchar>::iterator it;
        double totalWeightX = 0;
        double totalWeightY = 0;
        double totalWeight = 0;

        const int step = 1; // 每隔4个像素采样一次
        for (int y = 0; y < binaryImg.rows; y += step) {
            for (int x = 0; x < binaryImg.cols; x += step) {
                double weight = static_cast<double>(binaryImg.at<uchar>(y, x)); // 像素的灰度值作为权重
                totalWeightX += x * weight;
                totalWeightY += y * weight;
                totalWeight += weight;
            }
        }
        cout << "total weight = " << totalWeight << endl;

        // 计算加权平均位置
        float centerX = static_cast<float>(totalWeightX) / totalWeight + frame.cols / 4;
        float centerY = static_cast<float>(totalWeightY) / totalWeight + frame.rows / 4;
        rectangle(frame, Rect(max((int)(centerX - 30), 0), max((int)(centerY - 30), 0), 60, 60),
            Scalar(0, 0, 255), 2, 8, 0);
        cout << "全图质心位置：(" << centerX << ", " << centerY << ")" << endl;

        // 显示效果图窗口
        showImg(frame, start);
    }


    return 0;
}
