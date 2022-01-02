import os
import queue
from pathlib import PosixPath

from minio import Minio

from ..app_config import MinioConfig
from ..video.output_file import OutputFile
from .base_worker import BaseWorker


def get_object_name(output_file: OutputFile) -> str:
    file_name, _ = os.path.splitext(output_file.file_path)
    return str(
        PosixPath(output_file.file_type.value)
        / str(output_file.timestamp.year).zfill(2)
        / str(output_file.timestamp.month).zfill(2)
        / str(output_file.timestamp.day).zfill(2)
        / file_name
    )


class MinioWorker(BaseWorker):
    def __init__(
        self,
        config: MinioConfig,
        file_queue: "queue.Queue[OutputFile]",
    ):
        super().__init__(name="MinioWorker")
        self._file_queue = file_queue

        self._minio_client = Minio(
            f"{config.host}:{config.port}",
            secure=config.secure,
            access_key=config.access_key,
            secret_key=config.secret_key,
        )
        self._bucket_name = config.bucket

        if not self._minio_client.bucket_exists(self._bucket_name):
            self._minio_client.make_bucket(self._bucket_name)

    def run_processing(self):
        try:
            output_file = self._file_queue.get(block=True, timeout=1)

            print(output_file)
        except queue.Empty:
            pass

    def upload_file(self, output_file: OutputFile):
        try:
            object_name = get_object_name(output_file)

            print(
                f"Uploading file from {output_file.file_path} to {self._bucket_name}/{object_name}"
            )
            self._minio_client.fput_object(
                bucket_name=self._bucket_name,
                object_name=object_name,
                file_path=output_file.file_path,
            )
        except Exception as error:
            print(error)
        finally:
            try:
                os.remove(output_file.file_path)
            except Exception as error:
                print(error)
