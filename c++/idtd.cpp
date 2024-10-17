/*红外弱小目标检测：检测单目标*/
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

//newTopHat参数
int ro = 11;
int ri = 10;
Mat delta_b = newRingStrel(ro, ri);
Mat bb = Mat::ones(2 * ri + 1, 2 * ri + 1, CV_8UC1);

Mat newRingStrel(int ro, int ri)
{
    int d = 2 * ro + 1;
    Mat se = getStructuringElement(MORPH_RECT, Size(d, d));
    int start_index = ro + 1 - ri;
    int end_index = ro + 1 + ri;
    Rect roi(start_index, start_index, end_index - start_index, end_index - start_index);
    se(roi) = Scalar(0);
    return se;
}

Mat MNWTH(Mat img, Mat delta_b, Mat bb)
{
    Mat img_d, img_e, out;
    Mat img_d2;
    Mat temp, binaryImg, img_f;
    auto start1 = high_resolution_clock::now();
    img_f = img.clone();
    threshold(img, binaryImg, 100, 255, THRESH_BINARY);

    dilate(img, img_d, delta_b);

    auto end1 = high_resolution_clock::now();
    duration<double, milli> ms_double1 = end1 - start1;

    auto start2 = high_resolution_clock::now();

    erode(img_d, img_e, bb);

    auto end2 = high_resolution_clock::now();
    duration<double, milli> ms_double2 = end2 - start2;

    out = img - img_e;

    out.setTo(0, out < 0);
    return out;
}

Mat MoveDetect(Mat frame)
{

    Mat gray;
    cvtColor(frame, gray, CV_BGR2GRAY);

    auto start1 = high_resolution_clock::now();

    Mat result = MNWTH(gray, delta_b, bb);

    auto end1 = high_resolution_clock::now();
    duration<double, milli> ms_double1 = end1 - start1;

    return result;//返回result  
}

float myDistance(const Point& p1, const Point& p2)
{
    return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
}

void showImg(Mat frame, chrono::steady_clock::time_point start)
{
    auto end = high_resolution_clock::now();
    duration<double, milli> ms_double = end - start;
    imshow("frame", frame);
    waitKey(1);
}


int main()
{
    VideoCapture video("D:/dataset/corseTrack/104658.mp4", IMREAD_GRAYSCALE);
    if (!video.isOpened())
        return -1;

    int frameCount = video.get(CAP_PROP_FRAME_COUNT);//获取帧数  
    double FPS = video.get(CAP_PROP_FPS);//获取FPS  
    Mat frame;//存储帧  
    Mat result;//存储结果图像
    bool complexFlag = false;//进入复杂场景，如强层云
    Mat binaryImg;

    int minValue = 5;//最低亮度阈值
    float complex_bg = 200.0;//复杂背景阈值

    int num = 0;

    for (int i = 0; i < frameCount; i++)
    {
        video >> frame;//读帧进frame
        auto start = high_resolution_clock::now();
        //cout << "num == " << num << endl;
        num++;

        if (frame.empty())//对帧进行异常检测  
        {
            cout << "frame is empty!" << endl;
            break;
        }

        //无论目标是否丢失，均需执行检测算法
        if (complexFlag == false)//WTH算法
        {
            result = MoveDetect(frame);
        }
        namedWindow("nobinary,frame", 0);
        resizeWindow("nobinary,frame", 640, 480);
        imshow("nobinary,frame", result);
        threshold(result, binaryImg, 30, 255, THRESH_BINARY);
        namedWindow("binaryImg", 0);
        resizeWindow("binaryImg", 640, 480);
        imshow("binaryImg", binaryImg);

        //获取所有目标
        vector<Point> currentPoints;
        vector<Point> currentScale;
        Mat labels, stats, centroids;
        int num_labels = connectedComponentsWithStats(binaryImg, labels, stats, centroids, 4);
        for (int i = 1; i < num_labels; i++)
        {
            int area = stats.at<int>(i, CC_STAT_AREA);
            int x = stats.at<int>(i, CC_STAT_LEFT) + stats.at<int>(i, CC_STAT_WIDTH) / 2;
            int y = stats.at<int>(i, CC_STAT_TOP) + stats.at<int>(i, CC_STAT_HEIGHT) / 2;
            int w = stats.at<int>(i, CC_STAT_WIDTH);
            int h = stats.at<int>(i, CC_STAT_HEIGHT);
            currentPoints.push_back(Point(x, y));
            currentScale.push_back(Point(w, h));
        }

        namedWindow("frame", 0);
        resizeWindow("frame", 640, 480);

        //获取最显著目标，仅用于首帧
        double minVal, maxVal;
        Point minLoc, maxLoc;
        minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);
        float x = maxLoc.x;
        float y = maxLoc.y;
        float w = 20;
        float h = 20;

        rectangle(frame, Rect(x - w / 2, y - h / 2, w, h), Scalar(0, 255, 0), 2);//在result上绘制矩形框
        showImg(frame, start);

    }

    return 0;

}
