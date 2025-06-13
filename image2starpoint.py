import cv2 as cv
import numpy as np
import random
import os

def generate_star_constellation_bgr(image_path, starnum):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img_gray = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    _, thresh = cv.threshold(img_gray, 100, 255, cv.THRESH_BINARY_INV)
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("Can't find the outline.")

    contour = max(contours, key=cv.contourArea)
    points = contour.squeeze()
    if len(points.shape) != 2 or points.shape[0] < 5:
        raise ValueError("Not enough feature points")

    sampled_points = random.sample(list(points), min(starnum, len(points)))
    corners = cv.goodFeaturesToTrack(img_gray, maxCorners=20, qualityLevel=0.01, minDistance=10)
    corner_points = [tuple(pt.ravel()) for pt in corners] if corners is not None else []

    height, width = img_gray.shape
    star_img = np.zeros((height, width, 3), dtype=np.uint8)

    for pt in sampled_points:
        pt_tuple = tuple(pt)
        is_corner = any(np.linalg.norm(np.array(pt_tuple) - np.array(c)) < 5 for c in corner_points)

        if is_corner:
            size = 4
            color_bgr = (255, 255, 255)  # White
        else:
            size = random.choice([1, 2, 3])
            if size == 1:
                color_bgr = (180, 100, 230)  # purple
            elif size == 2:
                color_bgr = (200, 150, 230)
            else:
                color_bgr = (230, 200, 255)  # Nearly white purple

        cv.circle(star_img, pt_tuple, size, color_bgr, -1, lineType=cv.LINE_AA)

    return star_img

# test
if __name__ == "__main__":
    img = generate_star_constellation_bgr("./yossi40100logo.png",300)
    cv.imshow("Stars", img)
    cv.waitKey(0)
    cv.destroyAllWindows()
