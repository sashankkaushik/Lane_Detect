import cv2
import numpy as np


def region_of_interest(img, roi_vertices):
    mask = np.zeros_like(img)

    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, roi_vertices, ignore_mask_color)

    masked_image = cv2.bitwise_and(img, mask)
    assert isinstance(masked_image, object)
    return masked_image


def get_slope(x1, y1, x2, y2):
    return (y2 - y1) / (x2 - x1)


def draw_lines(img, lines, color=(128, 0, 128), thickness=3):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


image = cv2.imread("/Users/sashankkaushik/Downloads/StraightRoad.jpeg")
# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
assert isinstance(gray_image, object)
# Convert to HSV wrt yellow lane lines and white lane lines separately and combine the two to form a mask
img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower_yellow = np.array([20, 100, 100], dtype='uint8')
upper_yellow = np.array([30, 255, 255], dtype='uint8')
mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
mask_white = cv2.inRange(gray_image, 240, 255)
mask_yw = cv2.bitwise_or(mask_white, mask_yellow)
mask_yw_image = cv2.bitwise_and(gray_image, mask_yw)
# Apply a Gaussian Blur to the HSV image
kernel_size = 5
gauss_gray = cv2.GaussianBlur(mask_yw_image, (kernel_size, kernel_size), 0)
# Apply Canny edge detection
low_threshold = 50
high_threshold = 150
canny_edges = cv2.Canny(gauss_gray, low_threshold, high_threshold)
# Isolate Region of Interest using arbitrary vertices based on test image
imshape = image.shape
lower_left = [imshape[1] / 9, imshape[0]]
lower_right = [imshape[1] - imshape[1] / 9, imshape[0]]
top_left = [imshape[1] / 2 - imshape[1] / 8, imshape[0] / 2 + imshape[0] / 10]
top_right = [imshape[1] / 2 + imshape[1] / 8, imshape[0] / 2 + imshape[0] / 10]
vertices = [np.array([lower_left, top_left, top_right, lower_right], dtype=np.int32)]
roi_image = region_of_interest(canny_edges, vertices)
# Initialise Hough line params
rho = 4
theta = np.pi / 180

threshold = 30
min_line_len = 100
max_line_gap = 180
# Display image with Hough lines superimposed
line_image = hough_lines(roi_image, rho, theta, threshold, min_line_len, max_line_gap)
result = cv2.addWeighted(image, 0.8, line_image, 1, 0)

cv2.imshow('Lane_image', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
