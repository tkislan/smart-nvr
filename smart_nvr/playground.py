import queue
import threading
import time
from fractions import Fraction
from typing import List

import av
import cv2
import numpy as np

# import torch
from smart_nvr.camera.feed_multiplexer import CameraFeedMultiplexer
from smart_nvr.camera.image import (
    CameraImageContainer,
    DetectionCameraImageContainer,
    get_split_image_dimensions,
    split_image_into_squares,
)
from smart_nvr.detection.opencv import OpenCVTensorflowSSDMobilenetDetectionModel

from smart_nvr.detection.visualizer import Visualizer
from smart_nvr.workers.base_worker import BaseWorker

from smart_nvr.workers.camera_feed_worker import CameraFeedWorker
from smart_nvr.workers.detection_worker import DetectionWorker
from smart_nvr.workers.video_worker import VideoWorker
from smart_nvr.workers.visualize_worker import VisualizeWorker
from smart_nvr.app_config import ApplicationConfig


def warmup_model(model):
    img_raw = cv2.imread("data/images/cat.jpeg")
    img = CameraImageContainer.create(
        "warmup", img_raw, get_split_image_dimensions(img_raw)
    )
    detections = detect(model, img)
    assert "cat" in {d.name for d in detections}


def run2():
    config = ApplicationConfig.load_from_file("config.yaml")
    print(config)

    # return

    device = torch.device("cpu")
    model = torch.hub.load(
        "ultralytics/yolov5", "yolov5m"
    )  # or yolov5m, yolov5l, yolov5x, custom
    model = model.to(device)

    warmup_model(model)

    workers: List[BaseWorker] = []

    # camera_workers = [
    #     CameraFeedWorker(camera_config[0], camera_config[1], camera_config[2])
    #     for camera_config in camera_configs
    # ]
    camera_workers = [
        CameraFeedWorker(camera_name, camera_config)
        for camera_name, camera_config in config.camera_feeds.items()
    ]
    workers.extend(camera_workers)

    detection_worker = DetectionWorker(
        model=model,
        image_queue=CameraFeedMultiplexer([c.image_queue for c in camera_workers]),
    )
    # Run detection worker in main thread

    visualizer_worker = VisualizeWorker(
        detection_queue=detection_worker.detection_queue,
    )
    workers.append(visualizer_worker)

    video_worker = VideoWorker(
        output_directory="output",
        detection_queue=visualizer_worker.output_queue,
    )
    workers.append(video_worker)

    for worker in workers:
        worker.start()

    def timeout_stop_detection_worker():
        time.sleep(60)
        detection_worker.stop()

    threading.Thread(target=timeout_stop_detection_worker).start()

    # RUN
    # time.sleep(30)
    detection_worker.run()

    for worker in workers:
        worker.stop()

    for worker in workers:
        worker.join()


def run3():
    model = OpenCVTensorflowSSDMobilenetDetectionModel()
    model.load()

    # img_raw = cv2.imread("IMG_DDAC28943512-1.jpeg")
    # img_raw = cv2.imread("IMG_DDAC28943512-3.jpeg")
    # img_raw = cv2.imread("data/images/cat.jpeg")
    # img_raw = cv2.imread("data/images/kitten.jpeg")
    img_raw = cv2.imread("data/images/people-back-yard.jpeg")
    print(img_raw.shape)

    img = CameraImageContainer.create(
        "file", img_raw, get_split_image_dimensions(img_raw)
    )

    detections = model.detect(img)

    for detection in detections:
        print(detection.dict())
        Visualizer.draw_detection(img.raw_image_np, detection)

    cv2.imwrite("output.jpg", img.raw_image_np)


def run5():
    # fmt: off
    COCO_LABELS = {
        0: "background", 1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane", 6: "bus", 7: "train",
        8: "truck", 9: "boat", 10: "traffic light", 11: "fire hydrant", 13: "stop sign", 14: "parking meter",
        15: "zebra", 16: "bird", 17: "cat", 18: "dog", 19: "horse", 20: "sheep", 21: "cow", 22: "elephant",
        23: "bear", 24: "zebra", 25: "giraffe", 27: "backpack", 28: "umbrella", 31: "handbag", 32: "tie",
        33: "suitcase", 34: "frisbee", 35: "skis", 36: "snowboard", 37: "sports ball", 38: "kite",
        39: "baseball bat", 40: "baseball glove", 41: "skateboard", 42: "surfboard", 43: "tennis racket",
        44: "bottle", 46: "wine glass", 47: "cup", 48: "fork", 49: "knife", 50: "spoon", 51: "bowl", 52: "banana",
        53: "apple", 54: "sandwich", 55: "orange", 56: "broccoli", 57: "carrot", 58: "hot dog", 59: "pizza",
        60: "donut", 61: "cake", 62: "chair", 63: "couch", 64: "potted plant", 65: "bed", 67: "dining table",
        70: "toilet", 72: "tv", 73: "laptop", 74: "mouse", 75: "remote", 76: "keyboard", 78: "microwave",
        79: "oven", 80: "toaster", 81: "sink", 82: "refrigerator", 84: "book", 85: "clock", 86: "vase",
        87: "scissors", 88: "teddy bear", 89: "hair drier", 90: "toothbrush",
    }
    # fmt: on

    t = int(time.perf_counter() * 1000)
    model = cv2.dnn.readNetFromTensorflow(
        "data/models/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb",
        "data/models/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pbtxt",
    )
    image_size = 300
    print(f"Model load took {int(time.perf_counter() * 1000) - t} ms")
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    img_raw = cv2.imread("data/images/cat.jpeg")
    img = CameraImageContainer.create(
        "warmup", img_raw, get_split_image_dimensions(img_raw)
    )

    for i in range(5):
        t = int(time.perf_counter() * 1000)
        model_img = cv2.resize(img.cropped_images[0], (image_size, image_size))
        print(f"Img resize took {int(time.perf_counter() * 1000) - t} ms")

        t = int(time.perf_counter() * 1000)
        model.setInput(
            cv2.dnn.blobFromImage(model_img, size=(image_size, image_size), swapRB=True)
        )
        print(f"Model set input took {int(time.perf_counter() * 1000) - t} ms")

        t = int(time.perf_counter() * 1000)
        output = model.forward()
        print(f"Model infer took {int(time.perf_counter() * 1000) - t} ms")

        for detection in output[0, 0, :, :]:
            confidence = detection[2]
            if confidence > 0.5:
                class_id = detection[1]
                print(
                    str(
                        str(class_id)
                        + " "
                        + str(detection[2])
                        + " "
                        + COCO_LABELS.get(int(class_id))
                    )
                )

        t = int(time.perf_counter() * 1000)
        model_img = cv2.resize(img.cropped_images[1], (image_size, image_size))
        print(f"Img resize took {int(time.perf_counter() * 1000) - t} ms")

        t = int(time.perf_counter() * 1000)
        model.setInput(
            cv2.dnn.blobFromImage(model_img, size=(image_size, image_size), swapRB=True)
        )
        print(f"Model set input took {int(time.perf_counter() * 1000) - t} ms")

        t = int(time.perf_counter() * 1000)
        output = model.forward()
        print(f"Model infer took {int(time.perf_counter() * 1000) - t} ms")

        for detection in output[0, 0, :, :]:
            confidence = detection[2]
            if confidence > 0.5:
                class_id = detection[1]
                print(
                    str(
                        str(class_id)
                        + " "
                        + str(detection[2])
                        + " "
                        + COCO_LABELS.get(int(class_id))
                    )
                )


if __name__ == "__main__":
    run3()
    print("Done")
