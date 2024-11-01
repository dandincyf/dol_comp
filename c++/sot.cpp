/*�����٣���Ե���ķ���270��*/
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

//����¼�
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

//����¼��ص�������ѡ�����Ŀ��
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

    int frameCount = video.get(CAP_PROP_FRAME_COUNT);//��ȡ֡��
    double FPS = video.get(CAP_PROP_FPS);//��ȡFPS
    Mat frame;//�洢֡
    Mat result;//�洢���ͼ��
    bool lightFlag = true;

    //Scharr�˲�����ر���
    Mat g_scharrGradient_X, g_scharrGradient_Y;
    Mat g_scharrAbsGradient_X, g_scharrAbsGradient_Y;
    Mat scharrImage;

    Mat binaryImg;

    int num = 0;

    for (int i = 0; i < frameCount; i++)
    {
        video >> frame;//��֡��frame
        namedWindow("00", 0);
        resizeWindow("00", 640, 512);
        imshow("00", frame);

        cvtColor(frame, frame, COLOR_BGR2GRAY);
        blur(frame, frame, Size(3, 3));
        //medianBlur(frame, frame, 3);

        //��ʴ
        //Mat b2 = Mat::ones(4, 4, CV_8UC1);
        //erode(frame, frame, b2);

        auto start = high_resolution_clock::now();
        cout << "min= " << (num / 20) / 60 << ",sec=" << (num / 20) % 60 << endl;
        num++;

        if (frame.empty())//��֡�����쳣���  
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


        // �� X�����ݶ�
        Scharr(frame_littel, g_scharrGradient_X, CV_16S, 1, 0, 1, 0, BORDER_DEFAULT);
        convertScaleAbs(g_scharrGradient_X, g_scharrAbsGradient_X);//�������ֵ���������ת����8λ
        // ��Y�����ݶ�
        Scharr(frame_littel, g_scharrGradient_Y, CV_16S, 0, 1, 1, 0, BORDER_DEFAULT);
        convertScaleAbs(g_scharrGradient_Y, g_scharrAbsGradient_Y);//�������ֵ���������ת����8λ
        // �ϲ��ݶ�
        addWeighted(g_scharrAbsGradient_X, 0.5, g_scharrAbsGradient_Y, 0.5, 0, scharrImage);


        namedWindow("11", 0);
        resizeWindow("11", 640, 512);
        imshow("11", scharrImage);

        //��topN��threshold
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

        const int step = 1; // ÿ��4�����ز���һ��
        for (int y = 0; y < binaryImg.rows; y += step) {
            for (int x = 0; x < binaryImg.cols; x += step) {
                double weight = static_cast<double>(binaryImg.at<uchar>(y, x)); // ���صĻҶ�ֵ��ΪȨ��
                totalWeightX += x * weight;
                totalWeightY += y * weight;
                totalWeight += weight;
            }
        }
        cout << "total weight = " << totalWeight << endl;

        // �����Ȩƽ��λ��
        float centerX = static_cast<float>(totalWeightX) / totalWeight + frame.cols / 4;
        float centerY = static_cast<float>(totalWeightY) / totalWeight + frame.rows / 4;
        rectangle(frame, Rect(max((int)(centerX - 30), 0), max((int)(centerY - 30), 0), 60, 60),
            Scalar(0, 0, 255), 2, 8, 0);
        cout << "ȫͼ����λ�ã�(" << centerX << ", " << centerY << ")" << endl;

        // ��ʾЧ��ͼ����
        showImg(frame, start);
    }


    return 0;
}
