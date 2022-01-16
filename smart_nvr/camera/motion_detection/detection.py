from typing import List, Tuple

import cv2
import numpy as np

# GAUSIAN_BLUR_SIZE = (5, 5)
GAUSIAN_BLUR_SIZE = (21, 21)
MIN_CONTOUR_AREA_FACTOR = 0.0005
MAX_CONTOUR_AREA_FACTOR = 0.2
THRESHOLD_MINVALUE = 20
THRESHOLD_MAXVALUE = 255
DEFAULT_COUNTOURS_SIZE = 4


def contours_to_rectangles(contours) -> List[Tuple[int, int, int, int]]:
    return [cv2.boundingRect(contour) for contour in contours]


def filter_motion_areas(
    contours,
    min_contour_area: int,
    max_contour_area: int,
    size: int = DEFAULT_COUNTOURS_SIZE,
) -> List[Tuple[int, int, int, int]]:
    return list(
        sorted(
            [
                rectangle
                for rectangle in contours_to_rectangles(contours)
                if rectangle[2] * rectangle[3] > min_contour_area
                and rectangle[2] * rectangle[3] < max_contour_area
            ],
            key=lambda rectangle: rectangle[2] * rectangle[3],
            reverse=True,
        )
    )[:size]


def filter_motion_areas_by_contours(
    contours,
    min_contour_area: int,
    max_contour_area: int,
    size: int = DEFAULT_COUNTOURS_SIZE,
) -> List[Tuple[int, int, int, int]]:
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


def rectangle_to_dimensions(
    rectangle: Tuple[int, int, int, int],
    shape: Tuple[int, int],  # height, width
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    img_height, img_width = shape

    x, y, rec_width, rec_height = rectangle

    if rec_width < 300 and rec_height < 300:
        size = 300
    elif rec_width < 600 and rec_height < 600:
        size = 600
    elif rec_width < 900 and rec_height < 900:
        size = 900

    topleft_x = max(0, x - int((size - rec_width) / 2))
    topleft_y = max(0, y - int((size - rec_height) / 2))

    bottomright_x = min(img_width - 1, (x + rec_width) + int((size - rec_width) / 2))
    bottomright_y = min(img_height - 1, (y + rec_height) + int((size - rec_height) / 2))

    topleft_x = max(0, bottomright_x - size)
    topleft_y = max(0, bottomright_y - size)

    bottomright_x = topleft_x + size
    bottomright_y = topleft_y + size

    return ((topleft_y, bottomright_y), (topleft_x, bottomright_x))


def detect_motion(
    previous_img: np.ndarray, current_img: np.ndarray
) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
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

    return [
        rectangle_to_dimensions(rectangle, (height, width)) for rectangle in rectangles
    ]
