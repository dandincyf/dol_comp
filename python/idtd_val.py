import cv2
import numpy as np
import os
import time
import json
import sys
import idtd
from idtd_lyy import process_image


def calculate_time_score(time_in_ms):
    """根据处理时间计算得分"""
    if time_in_ms <= 2:
        return 90 + 10 * (2 - time_in_ms) / 2
    elif time_in_ms > 2 and time_in_ms <= 6:
        return 60 + 30 * (6 - time_in_ms) / 4
    elif time_in_ms > 6 and time_in_ms <= 9:
        return 60 * (9 - time_in_ms) / 3
    else:
        return 0


def calculate_acc_score(pixels):
    """根据像素差计算得分"""
    if pixels > 20:
        return 0
    else:
        return 100 * (20 - pixels) / 20


def calculate_center_from_gt(label_file):
    """读取真值标签文件并计算框的中心坐标"""
    with open(label_file, 'r') as f:
        line = f.readline().strip()
        if not line:
            raise ValueError(f"Ground truth file {label_file} is empty.")

        category, x, y, w, h = map(float, line.split())
        return x + w / 2, y + h / 2


def calculate_pixel_difference(pred_center, gt_center):
    """计算预测标签和真值标签中心坐标的像素差值"""
    return np.linalg.norm(np.array(pred_center) - np.array(gt_center))


def draw_boxes(image, gt_center, pred_center, elapsed_time, time_score, pixel_difference, acc_score):
    """在图像上绘制框和中心点"""
    box_size = (40, 30)

    # 真实值框和中心点
    gt_x, gt_y = int(gt_center[0] - box_size[0] / 2), int(gt_center[1] - box_size[1] / 2)
    cv2.rectangle(image, (gt_x, gt_y), (gt_x + box_size[0], gt_y + box_size[1]), (0, 255, 0), 2)  # 绿色
    cv2.circle(image, (int(gt_center[0]), int(gt_center[1])), 5, (0, 255, 0), -1)

    # 预测值框和中心点
    pred_x, pred_y = int(pred_center[0] - box_size[0] / 2), int(pred_center[1] - box_size[1] / 2)
    cv2.rectangle(image, (pred_x, pred_y), (pred_x + box_size[0], pred_y + box_size[1]), (255, 0, 0), 2)  # 红色
    cv2.circle(image, (int(pred_center[0]), int(pred_center[1])), 5, (255, 0, 0), -1)

    # 绘制每一行信息
    lines = [
        f"Time: {elapsed_time:.2f} ms, Time Score: {time_score:.2f}",
        f"Pixel Diff: {pixel_difference:.2f}, Acc Score: {acc_score:.2f}",
        f"Total Score: {0.3 * time_score + 0.7 * acc_score:.2f}"
    ]

    # Y 坐标初始位置
    y_offset = 30

    for line in lines:
        cv2.putText(image, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 20  # 增加 Y 坐标，以便每行之间有一定间距


def main():
    input_folder = r"I:\dolphin_dataset\pre\process_data\2and3\images"  # 测试图片文件夹
    gt_folder = r"I:\dolphin_dataset\pre\process_data\2and3\labels"  # 真值标签的txt文件夹路径
    center_folder = "C/"  # 中心坐标结果保存文件夹
    result_log = "D/sot-log.txt"  # 保存计算的日志文件路径
    output_image_folder = "output_image/"  # 保存每帧图像的文件夹
    output_video_path = "output_video.avi"  # 视频输出路径

    # 创建输出文件夹
    os.makedirs(center_folder, exist_ok=True)
    os.makedirs("D", exist_ok=True)
    os.makedirs(output_image_folder, exist_ok=True)

    log_entries = []
    total_time_score = 0
    total_acc_score = 0
    total_score = 0
    count = 0

    # # 视频写入对象
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = None

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)

            if image is None:
                print(f"Failed to read image: {filename}")
                continue

            start_time = time.time()
            center = process_image(image)

            if center is not None:
                center_file_path = os.path.join(center_folder, f"{os.path.splitext(filename)[0]}.txt")
                with open(center_file_path, 'w') as f:
                    f.write(f"{center[0]:.2f} {center[1]:.2f}\n")

                elapsed_time = (time.time() - start_time) * 1000
                time_score = calculate_time_score(elapsed_time)

                gt_file = os.path.join(gt_folder, f"{os.path.splitext(filename)[0]}.txt")
                if os.path.exists(gt_file):
                    try:
                        gt_center = calculate_center_from_gt(gt_file)
                    except ValueError as e:
                        print("error!!!")
                        print(e)  # 输出错误信息，跳过该文件
                        continue

                    pixel_difference = calculate_pixel_difference(center, gt_center)
                    acc_score = calculate_acc_score(pixel_difference)

                    log_entry = {
                        "filename": filename,
                        "pixel_difference": pixel_difference,
                        "acc_score": acc_score,
                        "time_score": time_score,
                        "score": 0.3 * time_score + 0.7 * acc_score
                    }
                    log_entries.append(log_entry)

                    # 累加分数
                    total_time_score += time_score
                    total_acc_score += acc_score
                    total_score += log_entry["score"]
                    count += 1

                    print(f"现在是第 {count} 张图片")

                    # 绘制框和中心点
                    draw_boxes(image, gt_center, center, elapsed_time, time_score, pixel_difference,
                               acc_score)

                    # 保存图像
                    output_image_path = os.path.join(output_image_folder, filename)
                    cv2.imwrite(output_image_path, image)

                    # # 初始化视频写入对象
                    # if out is None:
                    #     out = cv2.VideoWriter(output_video_path, fourcc, 30, (image.shape[1], image.shape[0]))
                    #
                    # # 写入视频帧
                    # out.write(image)
                    # print(f"已经写入 {count} 帧视频")

                else:
                    print(f"Ground truth file for {filename} not found!")

    # # 释放视频写入对象
    # if out is not None:
    #     out.release()
    #     print("结束视频写入")


    # 计算平均分
    if count > 0:
        avg_time_score = total_time_score / count
        avg_acc_score = total_acc_score / count
        avg_score = total_score / count
        print(f"Average time_score: {avg_time_score:.2f}")
        print(f"Average acc_score: {avg_acc_score:.2f}")
        print(f"Average score: {avg_score:.2f}")
    else:
        print("No valid entries to calculate averages.")

    with open(result_log, "w") as log_file:
        json.dump(log_entries, log_file, indent=4)


if __name__ == "__main__":
    main()
