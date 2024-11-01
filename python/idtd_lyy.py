import cv2
import numpy as np

def new_ring_strel(ro, ri):
    d = 2 * ro + 1
    se = np.ones((d, d), dtype=np.uint8)
    start_index = ro + 1 - ri
    end_index = ro + 1 + ri
    se[start_index:end_index, start_index:end_index] = 0
    return se

def mnwth(img, delta_b, bb):
    img_d = cv2.dilate(img, delta_b)
    img_e = cv2.erode(img_d, bb)
    out = cv2.subtract(img, img_e)
    out[out < 0] = 0
    return out

def smooth_image(frame):
    return cv2.GaussianBlur(frame, (7, 7), 0)

def move_detect(frame):
    ro = 11
    ri = 10
    delta_b = new_ring_strel(ro, ri)
    bb = np.ones((2 * ri + 1, 2 * ri + 1), dtype=np.uint8)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    result = mnwth(gray, delta_b, bb)
    return result

def get_subpixel_centroid_using_minmax(result, frame):
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    cx, cy = max_loc
    search_window = 3
    region = result[cy - search_window:cy + search_window + 1, cx - search_window:cx + search_window + 1]

    total = region.sum()
    if total == 0:
        return (float('nan'), float('nan'))

    h, w = region.shape
    y_indices, x_indices = np.indices((h, w))
    cx = (region * x_indices).sum() / total + (cx - search_window)
    cy = (region * y_indices).sum() / total + (cy - search_window)

    return (cx, cy)

def get_subpixel_centroid_using_gradient(result, frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    std_dev = cv2.meanStdDev(gradient_magnitude)[1][0][0]
    threshold = std_dev * 0.5
    candidates = np.where(gradient_magnitude > threshold)

    cx, cy = float('nan'), float('nan')

    if len(candidates[0]) > 0:
        weights = gradient_magnitude[candidates]
        cx = np.average(candidates[1], weights=weights)
        cy = np.average(candidates[0], weights=weights)
    return (cx, cy)

def process_image(frame):
    smoothed_frame = smooth_image(frame)
    result = move_detect(smoothed_frame)

    centroid = get_subpixel_centroid_using_minmax(result, frame)
    cx, cy = centroid

    if np.isnan(cx) or (cx == 0 and cy == 0):
        cx, cy = get_subpixel_centroid_using_gradient(result, frame)

    return (cx, cy)
