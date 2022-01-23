from typing import List

import cv2
import numpy as np

from ...utils.geometry import merge_rectangles, square_box_rectangles
from ...utils.rectangle import Rectangle

# GAUSIAN_BLUR_SIZE = (5, 5)
GAUSIAN_BLUR_SIZE = (21, 21)
MIN_CONTOUR_AREA_FACTOR = 0.0005
MAX_CONTOUR_AREA_FACTOR = 0.2
THRESHOLD_MINVALUE = 20
THRESHOLD_MAXVALUE = 255
DEFAULT_COUNTOURS_SIZE = 2


def contours_to_rectangles(contours) -> List[Rectangle]:
    bounding_rectangles = [cv2.boundingRect(contour) for contour in contours]
    return [Rectangle(x, y, x + w, y + h) for x, y, w, h in bounding_rectangles]


def filter_motion_areas(
    contours,
    min_contour_area: int,
    max_contour_area: int,
    size: int = DEFAULT_COUNTOURS_SIZE,
) -> List[Rectangle]:
    return list(
        sorted(
            [
                rectangle
                for rectangle in contours_to_rectangles(contours)
                if rectangle.area > min_contour_area
                and rectangle.area < max_contour_area
            ],
            key=lambda rectangle: rectangle.area,
            reverse=True,
        )
    )[:size]


def filter_motion_areas_by_contours(
    contours,
    min_contour_area: int,
    max_contour_area: int,
    size: int = DEFAULT_COUNTOURS_SIZE,
) -> List[Rectangle]:
    return contours_to_rectangles(
        [
            contour
            for contour, _ in (
                sorted(
                    [
                        (contour, contour_area)
                        for contour, contour_area in [
                            (contour, cv2.contourArea(contour)) for contour in contours
                        ]
                        if contour_area > min_contour_area
                        and contour_area < max_contour_area
                    ],
                    key=lambda v: v[1],
                    reverse=True,
                )
            )
        ][:size]
    )


def detect_motion(previous_img: np.ndarray, current_img: np.ndarray) -> List[Rectangle]:
    (height, width, _) = current_img.shape
    total_area = height * width

    min_contour_area = int(total_area * MIN_CONTOUR_AREA_FACTOR)
    max_contour_area = int(total_area * MAX_CONTOUR_AREA_FACTOR)

    diff = cv2.absdiff(previous_img, current_img)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    diff_blur = cv2.GaussianBlur(diff_gray, GAUSIAN_BLUR_SIZE, 0)
    _, thresh_bin = cv2.threshold(
        diff_blur, THRESHOLD_MINVALUE, THRESHOLD_MAXVALUE, cv2.THRESH_BINARY
    )
    # thresh_bin = cv2.dilate(thresh_bin, None, iterations=3)
    # thresh_bin = cv2.erode(thresh_bin, None, iterations=3)
    contours, _ = cv2.findContours(thresh_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # rectangles = filter_motion_areas(contours, min_contour_area, max_contour_area)
    rectangles = filter_motion_areas_by_contours(
        contours, min_contour_area, max_contour_area
    )

    rectangles = merge_rectangles(rectangles)

    dimensions = [
        square_box_rectangles(rectangle, (height, width)) for rectangle in rectangles
    ]

    return dimensions
