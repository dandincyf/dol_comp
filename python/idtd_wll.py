import cv2
import numpy as np


def newRingStrel(ro, ri):
    d = 2 * ro + 1
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (d, d))
    start_index = ro + 1 - ri
    end_index = ro + 1 + ri
    se[start_index:end_index, start_index:end_index] = 0  # 设置内部区域为零
    return se


def MNWTH(img, delta_b, bb):
    img_d = cv2.dilate(img, delta_b)  # 执行膨胀
    img_e = cv2.erode(img_d, bb)  # 执行腐蚀
    out = cv2.subtract(img, img_e)  # 从原始图像减去腐蚀后的图像
    out[out < 0] = 0  # 将负值设置为零
    return out


def process_image(image):
    # 参数设置
    ro = 11
    ri = 10
    delta_b = newRingStrel(ro, ri)
    bb = np.ones((2 * ri + 1, 2 * ri + 1), dtype=np.uint8)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
    result = MNWTH(gray, delta_b, bb)  # 检测运动

    _, binaryImg = cv2.threshold(result, 30, 255, cv2.THRESH_BINARY)  # 二值化

    # 获取连通组件
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binaryImg, connectivity=4)

    # 找到最大的连通区域的质心
    if num_labels > 1:
        largest_component_index = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # 找到最大的连通区域
        centroid = centroids[largest_component_index]  # 获取质心坐标
        return int(centroid[0]), int(centroid[1])  # 返回质心坐标 (x, y)

    return None  # 如果没有检测到对象，返回 None


if __name__ == "__main__":
    # 示例用法
    image_path = r"I:\wll\images\15648.bmp"  # 替换为你的图像路径
    image = cv2.imread(image_path)

    if image is not None:
        centroid = process_image(image)
        if centroid:
            print(f"质心坐标: {centroid}")
        else:
            print("没有检测到对象。")
    else:
        print("无法读取图像。")
