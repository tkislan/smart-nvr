import datetime
import os.path
from typing import Dict, List, Optional, Tuple

import cv2

from smart_nvr.video.output_file import OutputFile, OutputFileType

from ..camera.image import DetectionCameraImageContainer
from .video_output import VideoOutput

VIDEO_MAX_TIME_MILLIS = 15 * 1000  # 15 seconds
VIDEO_MAX_NO_DETECTION_TIME_MILLIS = 5 * 1000  # 5 seconds


class VideoWriterManager:
    _output_directory: str
    _video_outputs: Dict[str, VideoOutput]

    def __init__(self, output_directory: str):
        self._output_directory = output_directory
        self._video_outputs = {}

    def write_image(self, img: DetectionCameraImageContainer) -> Optional[OutputFile]:
        if not img.has_detections() and not self.has_video_output(
            img.camera_image_container.camera_name
        ):
            return None

        camera_name = img.camera_image_container.camera_name
        detection_image_file: Optional[OutputFile] = None

        video_output = self._video_outputs.get(camera_name)
        if video_output is None:
            video_output = self.create_video_output(img)

            image_file_path = self.write_image_file(img)
            detection_image_file = OutputFile(
                file_type=OutputFileType.image,
                file_path=image_file_path,
                timestamp=video_output.timestamp,
            )

        video_output.write_image(img)

        return detection_image_file

    def has_video_output(self, camera_name: str) -> bool:
        return camera_name in self._video_outputs

    def create_video_output(self, img: DetectionCameraImageContainer) -> VideoOutput:
        camera_name = img.camera_image_container.camera_name
        height = img.camera_image_container.raw_image_np.shape[0]
        width = img.camera_image_container.raw_image_np.shape[1]

        video_output = self._video_outputs.get(camera_name)

        if video_output is None:
            img_datetime = datetime.datetime.fromtimestamp(
                img.camera_image_container.created_at / 1000, datetime.timezone.utc
            )

            file_path = os.path.join(
                self._output_directory,
                f"{camera_name}_{img_datetime.strftime('%Y-%m-%dT%H%M%S')}.mp4",
            )
            video_output = VideoOutput(file_path, width, height)
            print("Created video output:", file_path, video_output)
            self._video_outputs[camera_name] = video_output

        return video_output

    def write_image_file(self, img: DetectionCameraImageContainer) -> str:
        img_datetime = datetime.datetime.fromtimestamp(
            img.camera_image_container.created_at / 1000, datetime.timezone.utc
        )
        file_path = os.path.join(
            self._output_directory,
            f"{img.camera_image_container.camera_name}_{img_datetime.strftime('%Y-%m-%dT%H%M%S')}.jpeg",
        )
        cv2.imwrite(file_path, img.camera_image_container.raw_image_np)
        return file_path

    @staticmethod
    def close_video_outputs(
        video_outputs: List[Tuple[str, VideoOutput]]
    ) -> List[Tuple[str, VideoOutput]]:
        closed_video_outputs: List[Tuple[str, VideoOutput]] = []

        for name, video_output in video_outputs:
            video_output.close()
            closed_video_outputs.append((name, video_output))

        return closed_video_outputs

    def close_eligible_video_outputs(self) -> List[OutputFile]:
        closed_video_outputs = self.close_video_outputs(
            [
                (name, video_worker)
                for name, video_worker in self._video_outputs.items()
                if video_worker.is_max_time_running()
                or video_worker.is_max_time_no_detection_running()
            ]
        )

        for name, _ in closed_video_outputs:
            del self._video_outputs[name]

        return [
            OutputFile(
                file_type=OutputFileType.video,
                file_path=video_output.file_path,
                timestamp=video_output.timestamp,
            )
            for _, video_output in closed_video_outputs
        ]

    def close_all_video_outputs(self) -> List[OutputFile]:
        closed_video_outputs = self.close_video_outputs(
            list(self._video_outputs.items())
        )

        for name, _ in closed_video_outputs:
            del self._video_outputs[name]

        return [
            OutputFile(
                file_type=OutputFileType.video,
                file_path=video_output.file_path,
                timestamp=video_output.timestamp,
            )
            for _, video_output in closed_video_outputs
        ]
