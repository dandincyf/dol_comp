import cv2
import numpy as np
import time
import math

def my_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def show_img(frame, start, algorithm_name):
    end = time.time()
    ms_double = (end - start) * 1000
    fps = 1000 / ms_double if ms_double > 0 else 0
    print(f"it took {ms_double:.2f} ms")

    # 在图像上显示 FPS 和算法名称
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Algorithm: {algorithm_name}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.namedWindow("result", 0)
    cv2.resizeWindow("result", 640, 512)
    cv2.imshow("result", frame)
    cv2.waitKey(1)

def process_image(frame):
    # 转换为灰度图像并模糊处理
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.blur(gray_frame, (3, 3))

    # 腐蚀操作
    b2 = np.ones((4, 4), np.uint8)
    gray_frame = cv2.erode(gray_frame, b2)

    # 裁剪ROI区域
    roi_frame = gray_frame[gray_frame.shape[0] // 4: 3 * gray_frame.shape[0] // 4,
                           gray_frame.shape[1] // 4: 3 * gray_frame.shape[1] // 4]

    # 计算Scharr梯度
    scharr_grad_x = cv2.Scharr(roi_frame, cv2.CV_16S, 1, 0)
    scharr_grad_y = cv2.Scharr(roi_frame, cv2.CV_16S, 0, 1)
    abs_grad_x = cv2.convertScaleAbs(scharr_grad_x)
    abs_grad_y = cv2.convertScaleAbs(scharr_grad_y)
    scharr_image = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    # 二值化处理
    _, binary_img = cv2.threshold(scharr_image, 30, 255, cv2.THRESH_BINARY)

    # 计算质心位置
    y_indices, x_indices = np.where(binary_img > 0)
    if len(x_indices) > 0:
        weights = binary_img[y_indices, x_indices].astype(float)
        total_weight = np.sum(weights)
        total_weight_x = np.sum(x_indices * weights)
        total_weight_y = np.sum(y_indices * weights)

        # 计算加权平均位置
        center_x = total_weight_x / total_weight + frame.shape[1] / 4
        center_y = total_weight_y / total_weight + frame.shape[0] / 4

        return int(center_x), int(center_y)  # 返回质心坐标 (x, y)

    return None  # 如果没有检测到对象，返回 None

def main():
    video_path = r"F:\丰宁外场数据\精跟0328\j105441.mp4"
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print(f"Failed to open video: {video_path}")
        return -1

    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取帧数
    num = 0

    try:
        for i in range(frame_count):
            ret, frame = video.read()
            if not ret:
                print("Failed to read frame!")
                break

            start = time.time()  # 开始计时

            centroid = process_image(frame)  # 处理图像，获取质心坐标
            if centroid:
                x, y = centroid
                cv2.rectangle(frame, (max(x - 30, 0), max(y - 30, 0)),
                              (min(x + 30, frame.shape[1]), min(y + 30, frame.shape[0])),
                              (0, 0, 255), 2)
                print(f"全图质心位置：({x:.2f}, {y:.2f})")

            # 显示效果图
            show_img(frame, start, algorithm_name="sot")

    finally:
        # 确保在任何情况下都能释放资源
        video.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
