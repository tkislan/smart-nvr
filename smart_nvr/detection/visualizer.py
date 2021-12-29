import cv2
import numpy as np

from ..detection.detection_types import Detection

BOXES_COLOR = (0, 255, 0)
BOXES_THICKNESS = 2
FONT_FACE = cv2.FONT_HERSHEY_DUPLEX
FONT_SCALE = 0.5
FONT_COLOR = (0, 0, 0)
FONT_THICKNESS = 1


class Visualizer:
    @staticmethod
    def draw_detection(image_np: np.ndarray, detection: Detection):
        cv2.rectangle(
            image_np,
            (int(detection.xmin), int(detection.ymin)),
            (int(detection.xmax), int(detection.ymax)),
            BOXES_COLOR,
            BOXES_THICKNESS,
        )

        confidence_percentage = "{0:.0%}".format(detection.confidence)
        label = "{}: {}".format(detection.name, confidence_percentage)

        cv2.putText(
            image_np,
            label,
            (int(detection.xmin), int(detection.ymin)),
            FONT_FACE,
            FONT_SCALE,
            FONT_COLOR,
            FONT_THICKNESS,
        )
