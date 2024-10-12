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

//newTopHat����
int ro = 11;
int ri = 10;
Mat delta_b = newRingStrel(ro, ri);
Mat bb = Mat::ones(2 * ri + 1, 2 * ri + 1, CV_8UC1);

//����¼�
Point initialPoint;
bool pointSelected = false;

Mat newRingStrel(int ro, int ri)
{
    int d = 2 * ro + 1;
    Mat se = getStructuringElement(MORPH_RECT, Size(d, d));
    //Mat se = Mat::ones(d, d, CV_8UC1);
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
    //cout << "dilate took " << ms_double1.count() << " ms" << endl;


    auto start2 = high_resolution_clock::now();

    erode(img_d, img_e, bb);

    auto end2 = high_resolution_clock::now();
    duration<double, milli> ms_double2 = end2 - start2;
    //cout << "erode took " << ms_double2.count() << " ms" << endl;

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
    //cout << "detect took " << ms_double1.count() << " ms" << endl;

    return result;//����result  
}

//Mat LCMDetect(Mat im)
//{
//    Size size = im.size();
//    int n = size.width;//cols
//    int m = size.height;
//    double  T0 = 0.0014;
//    int d = 2 * r + 1;
//    Mat local_region(d, d, CV_32F, Scalar(0));
//    Mat P(m, n, CV_32F, Scalar(0));
//
//
//    for (int i = r; i < m - r; i = i + 2) {
//        for (int j = r; j < n - r; j += 2) {
//            Range rows(i - r, i + r + 1);
//            Range cols(j - r, j + r + 1);
//            local_region = im(rows, cols);
//            P.at<float>(i, j) = im.at<uchar>(i, j) / sum(local_region)[0];
//        }
//    }
//
//    float threshold = 1 / pow((2 * r + 1), 2) + T0;
//    vector<Point> coords = getCoordsGreaterThanValue(P, threshold);
//
//    while (coords.size() < 100) {
//        T0 = T0 - 0.0001;
//        threshold = 1 / pow((2 * r + 1), 2) + T0;
//        coords = getCoordsGreaterThanValue(P, threshold);
//        continue;
//    }
//    // �ҵ���������ֵ����Сֵ
//    double minVal, MAX_G;
//    Point minLoc, maxLoc;
//    minMaxLoc(P, &minVal, &MAX_G, &minLoc, &maxLoc);
//
//    Mat f1(m, n, CV_32F, Scalar(0));
//    if (coords.size() > 0) {
//        for (const auto& coord : coords) {
//            f1.at<float>(coord.x, coord.y) = floor(float(P.at<float>(coord.x, coord.y) * float(255)) / float(MAX_G));
//        }
//    }
//
//    Mat f2(m, n, CV_32F, Scalar(0));
//    Mat mean, stddev;
//    meanStdDev(im, mean, stddev);
//    double threshold1 = mean.at<double>(0) + stddev.at<double>(0) * 5;
//    threshold1 = min(int(threshold1), 248);
//
//    for (int i = 0; i < m; ++i) {
//        for (int j = 0; j < n; ++j) {
//            if (f1.at<float>(i, j) > threshold1) {
//                f2.at<float>(i, j) = f1.at<float>(i, j);
//            }
//        }
//    }
//    return f2;//����result
//    //Mat resultImg;
//    //return resultImg;
//}

float myDistance(const Point& p1, const Point& p2)
{
    return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
}

void showImg(Mat frame, chrono::steady_clock::time_point start)
{
    auto end = high_resolution_clock::now();
    duration<double, milli> ms_double = end - start;
    //cout << "it took " << ms_double.count() << " ms" << endl;

    //namedWindow("frame", 0);
    //resizeWindow("frame", 640, 480);
    imshow("frame", frame);

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
    VideoCapture video("D:/dataset/idtd/104658.mp4", IMREAD_GRAYSCALE);
    if (!video.isOpened())
        return -1;

    int frameCount = video.get(CAP_PROP_FRAME_COUNT);//��ȡ֡��  
    double FPS = video.get(CAP_PROP_FPS);//��ȡFPS  
    Mat frame;//�洢֡  
    Mat result;//�洢���ͼ��

    bool missFlag = false;
    bool jumpFlag = false;
    bool firstFlag = true;
    bool twiceFlag = true;
    bool complexFlag = false;//���븴�ӳ�������ǿ����
    Mat binaryImg;

    float x_lastlast, y_lastlast;//xy����֡
    float x_last, y_last;//xy��֡
    float x_pre, y_pre;//xyԤ��ֵ
    int detectTimes = 0;//miss�������⵽��Զ�ĵط���Ŀ��
    //ֻ���ۼƼ�⵽5�ζ�����Ͻ�������Ϊ���¼�⵽��Ŀ��
    float x_jump, y_jump;//miss���⵽��Զ��Ŀ�꣬��¼����Ϊ��һ֡��ֵ

    int minValue = 40;//���������ֵ
    int jumpDistance = 50;//������ֵ
    int missJumpDistance = 20;//��ʧ�������жϵ�������ֵ
    int w_lcm = 50;//LCM�㷨��������
    int h_lcm = 50;
    int pixel_count = w_lcm * h_lcm;
    float complex_bg = 200.0;//���ӱ�����ֵ

    int num = 0;

    for (int i = 0; i < frameCount; i++)
    {
        video >> frame;//��֡��frame
        auto start = high_resolution_clock::now();
        //cout << "num == " << num << endl;
        num++;

        if (frame.empty())//��֡�����쳣���  
        {
            cout << "frame is empty!" << endl;
            break;
        }

        //����Ŀ���Ƿ�ʧ������ִ�м���㷨
        if (complexFlag == false)//WTH�㷨
        {
            result = MoveDetect(frame);
        }
        else//LCM�㷨������
        {
            /*Rect lcmRect(max(int(x_last - w_lcm / 2), 0),
                max(int(y_last - h_lcm / 2), 0), min(w_lcm, int(frame.cols - x_last)), min(h_lcm, int(frame.rows - y_last)));
            lcmRoi = frame(lcmRect).clone();
            Mat lcmResult = LCMDetect(lcmRoi);
            result.create(frame.size(), CV_8U);
            result.setTo(Scalar(0));
            lcmResult.copyTo(result(lcmRect));*/
        }
        namedWindow("nobinary,frame", 0);
        resizeWindow("nobinary,frame", 640, 480);
        imshow("nobinary,frame", result);
        threshold(result, binaryImg, 30, 255, THRESH_BINARY);
        namedWindow("binaryImg", 0);
        resizeWindow("binaryImg", 640, 480);
        imshow("binaryImg", binaryImg);

        //��ȡ����Ŀ��
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
            //rectangle(frame, Point(x - w/2 - 5, y - h/2 - 5), Point(x + w/2 + 5, y + h/2 + 5), Scalar(255, 0, 0), 2);
        }

        namedWindow("frame", 0);
        resizeWindow("frame", 640, 480);
        setMouseCallback("frame", onMouse);

        //��ȡ������Ŀ�꣬��������֡
        double minVal, maxVal;
        Point minLoc, maxLoc;
        minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);
        float x0 = maxLoc.x;
        float y0 = maxLoc.y;
        float w0 = 20;
        float h0 = 20;

        float x = 0;
        float y = 0;
        float w = 30;
        float h = 30;
        uchar pixel = frame.at<uchar>(y, x);//bug����������ĳ�uintֱ������ڴ������vs���������������־����޷���������

        if (num_labels == 1)
        {
            pixel = 0;
        }

        //�����㷨 s0,s1,s2

        //�û���ʼ����������ȼ� s0
        if (pointSelected)
        {
            pointSelected = false;

            missFlag = false;
            jumpFlag = false;
            firstFlag = false;
            twiceFlag = false;
            detectTimes = 0;

            x = initialPoint.x;
            y = initialPoint.y;
            x_last = x;
            y_last = y;
            x_lastlast = x;
            y_lastlast = y;
            rectangle(frame, Rect(x - (w0 + w) / 2, y - (h0 + h) / 2, w + w0, h + h0), Scalar(0, 255, 0), 2);//��result�ϻ��ƾ��ο�

            cout << "initial positon" << endl;
            showImg(frame, start);

            continue;
        }

        //���Ŀ��δ��ʧ s1
        if (missFlag == false)
        {
            //��Ч��һ֡����һ��֡���轫missFlag��Ϊtrue
            if (firstFlag == true) //s11
            {
                if (maxVal > minValue) //s111
                {
                    rectangle(frame, Rect(x0 - w0 / 2, y0 - h0 / 2, w0, h0), Scalar(0, 255, 0), 2);//��result�ϻ��ƾ��ο�
                    firstFlag = false;
                    x_last = x0;
                    y_last = y0;
                    cout << "s111, 1st frame" << endl;
                    showImg(frame, start);

                    continue;
                }
                else //s112
                {
                    cout << "s112, 1st frame doesn't have target" << endl;
                    showImg(frame, start);
                    continue;
                }
            }

            //��������Ч�ڶ�֡
            if (firstFlag == false && twiceFlag == true) //s12
            {
                //������Ŀ����������ĵ���ΪĿ�꣬����xy
                if (!currentPoints.empty())
                {
                    float minDistance = 999999;
                    float temp;
                    for (int i = 0; i < currentPoints.size(); i++)
                    {
                        temp = myDistance(Point(x_last, y_last), currentPoints[i]);
                        if (temp < minDistance)
                        {
                            minDistance = temp;
                            x = currentPoints[i].x;
                            y = currentPoints[i].y;
                            w = currentScale[i].x;
                            h = currentScale[i].y;
                        }
                    }
                }

                //�е���Ŀ���㷨
                if (w > 300 && h > 300)
                {
                    cout << "big target! exit!" << endl;
                    break;
                }

                //�����ж�
                if (myDistance(Point(x, y), Point(x_last, y_last)) > jumpDistance)
                {
                    jumpFlag = true;
                }
                else
                {
                    jumpFlag = false;
                }

                if (pixel > minValue && jumpFlag == false) //s121
                {
                    rectangle(frame, Rect(x - (w0 + w) / 2, y - (h0 + h) / 2, w + w0, h + h0), Scalar(0, 255, 0), 2);//��result�ϻ��ƾ��ο�
                    twiceFlag = false;
                    x_lastlast = x_last;
                    y_lastlast = y_last;
                    x_last = x;
                    y_last = y;
                    cout << "s121, 2nd frame" << endl;
                    showImg(frame, start);
                    continue;
                }
                else //s122
                {
                    cout << "s122, 2nd frame doesn't have target" << endl;
                    showImg(frame, start);
                    continue;
                }
            }

            //��������֡
            else //s13
            {
                x_pre = x_last /*+ (x_last - x_lastlast)*/;
                y_pre = y_last /*+ (y_last - y_lastlast)*/;

                //������Ŀ����������ĵ���ΪĿ�꣬����xy
                if (!currentPoints.empty())
                {
                    float minDistance = 999999;
                    float temp;
                    for (int i = 0; i < currentPoints.size(); i++)
                    {
                        temp = myDistance(Point(x_last, y_last), currentPoints[i]);
                        if (temp < minDistance)
                        {
                            minDistance = temp;
                            x = currentPoints[i].x;
                            y = currentPoints[i].y;
                            w = currentScale[i].x;
                            h = currentScale[i].y;
                        }
                    }
                }

                //�е���Ŀ���㷨
                //if (w > 10 && h > 10)
                //{
                //    cout << "big target! exit!" << endl;
                //    continue;
                //}

                //�е�LCM�㷨
                //���Ż�
                //Rect lcmRect(max(int(x_last - w_lcm / 2), 0),
                //    max(int(y_last - h_lcm / 2), 0), min(w_lcm, int(frame.cols - x_last)), min(h_lcm, int(frame.rows - y_last)));
                //lcmRoi = frame(lcmRect).clone();
                //if (sum(lcmRoi)[0] / pixel_count > complex_bg)
                //{
                //    cout << "LCM run!" << endl;
                //    complexFlag = true;
                //    continue;
                //}


                //�����ж�
                if (myDistance(Point(x, y), Point(x_last, y_last)) > jumpDistance)
                {
                    jumpFlag = true;
                }
                else
                {
                    jumpFlag = false;
                }

                //����Ҫ�����������жϣ�ֻҪmiss��˵������ʿ���������
                if (pixel > minValue && jumpFlag == false) //�ҵ�Ŀ�� s131
                {
                    rectangle(frame, Rect(x - (w0 + w) / 2, y - (h0 + h) / 2, w + w0, h + h0), Scalar(0, 255, 0), 2);//��result�ϻ��ƾ��ο�
                    twiceFlag = false;
                    x_lastlast = x_last;
                    y_lastlast = y_last;
                    x_last = x;
                    y_last = y;
                    cout << "s131,nomiss,tracking,pixel=" << unsigned(pixel) << ",x=" << x << ",y=" << y
                        << ",num_labels=" << num_labels << ",(w,h)=" << "(" << w << "," << h << ")" << endl;
                    showImg(frame, start);
                    continue;
                }
                else //δ�ҵ�Ŀ�� s132
                {
                    missFlag = true;
                    x_lastlast = x_last;
                    y_lastlast = y_last;
                    x_last = x_pre;
                    y_last = y_pre;
                    rectangle(frame, Rect(x_pre - w0 / 2, y_pre - h0 / 2, w0, h0), Scalar(0, 255, 0), 2);//��result�ϻ��ƾ��ο�

                    cout << "s132, nomiss, track miss, pixel=" << pixel << ",jumpFlag=" << jumpFlag << endl;
                    showImg(frame, start);
                    continue;
                }
            }
        }

        //���Ŀ�궪ʧ s2
        else if (missFlag == true)
        {
            x_pre = x_last /*+ (x_last - x_lastlast)*/;
            y_pre = y_last /*+ (y_last - y_lastlast)*/;

            //������Ŀ����������ĵ���ΪĿ�꣬����xy
            if (!currentPoints.empty())
            {
                float minDistance = 999999;
                float temp;
                for (int i = 0; i < currentPoints.size(); i++)
                {
                    temp = myDistance(Point(x_last, y_last), currentPoints[i]);
                    if (temp < minDistance)
                    {
                        minDistance = temp;
                        x = currentPoints[i].x;
                        y = currentPoints[i].y;
                        w = currentScale[i].x;
                        h = currentScale[i].y;
                    }
                }
            }


            //�����ж�
            if (myDistance(Point(x, y), Point(x_pre, y_pre)) > jumpDistance)
            {
                jumpFlag = true;
            }
            else
            {
                jumpFlag = false;
            }

            if (pixel > minValue && jumpFlag == false) //s21
            {
                detectTimes = 0;
                rectangle(frame, Rect(x - (w0 + w) / 2, y - (h0 + h) / 2, w + w0, h + h0), Scalar(0, 255, 0), 2);//��result�ϻ��ƾ��ο�
                missFlag = false;
                x_lastlast = x_last;
                y_lastlast = y_last;
                x_last = x;
                y_last = y;
                cout << "s21,miss,no jump,find again" << endl;
                showImg(frame, start);
                continue;
            }
            else if (pixel > minValue && jumpFlag == true) //s22
            {
                if (detectTimes == 0) //s221
                {
                    detectTimes += 1;
                    x_jump = x;
                    y_jump = y;
                    rectangle(frame, Rect(x_pre - w0 / 2, y_pre - h0 / 2, w0, h0), Scalar(0, 255, 0), 2);//��result�ϻ��ƾ��ο�
                    cout << "s221,miss,jump,find 1st time" << endl;
                    showImg(frame, start);
                    continue;
                }
                else if (detectTimes > 0 && detectTimes < 5) //s222
                {
                    if (myDistance(Point(x, y), Point(x_jump, y_jump)) < missJumpDistance) //s2221
                    {
                        detectTimes += 1;
                        x_jump = x;
                        y_jump = y;
                        rectangle(frame, Rect(x_pre - w0 / 2, y_pre - h0 / 2, w0, h0), Scalar(0, 255, 0), 2);//��result�ϻ��ƾ��ο�
                        x_lastlast = x_last;
                        y_lastlast = y_last;
                        x_last = x_pre;
                        y_last = y_pre;
                        cout << "s2221,miss,find " << detectTimes << " times" << endl;
                        showImg(frame, start);
                        continue;
                    }
                    else //s2222
                    {
                        detectTimes = 0;
                        rectangle(frame, Rect(x_pre - w0 / 2, y_pre - h0 / 2, w0, h0), Scalar(0, 255, 0), 2);//��result�ϻ��ƾ��ο�
                        x_lastlast = x_last;
                        y_lastlast = y_last;
                        x_last = x_pre;
                        y_last = y_pre;
                        cout << "s2222,miss,jump,not same target,detectTimes reset to 0" << endl;
                        showImg(frame, start);
                        continue;
                    }
                }
                else if (detectTimes >= 5) //s223
                {
                    detectTimes = 0;
                    missFlag = false;
                    x_last = x;
                    y_last = y;
                    x_lastlast = x;
                    y_lastlast = y;
                    rectangle(frame, Rect(x - (w0 + w) / 2, y - (h0 + h) / 2, w + w0, h + h0), Scalar(0, 255, 0), 2);//��result�ϻ��ƾ��ο�
                    cout << "s223,miss,find again" << endl;
                    showImg(frame, start);
                    continue;
                }
                else //s224
                {
                    cout << "224! miss, other situation unknown" << endl;
                    showImg(frame, start);
                    continue;
                }
            }
            else //s23
            {
                cout << "23! miss, pixel < 40, don't find target" << endl;
                detectTimes = 0;
                missFlag = true;
                x_lastlast = x_last;
                y_lastlast = y_last;
                x_last = x_pre;
                y_last = y_pre;
                rectangle(frame, Rect(x_pre - w0 / 2, y_pre - h0 / 2, w0, h0), Scalar(0, 255, 0), 2);//��result�ϻ��ƾ��ο�
                showImg(frame, start);
                continue;
            }

        }


    }

    return 0;

}
