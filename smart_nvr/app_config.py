from enum import Enum
from typing import Dict, Optional

from pydantic import BaseModel, Extra, Field
from typing_extensions import Literal
from yaml import safe_load

DEFAULT_CONFIG_FILE_PATH = "config.yaml"


def get_default_config_file_path() -> str:
    return DEFAULT_CONFIG_FILE_PATH


class AuthConfig(BaseModel, extra=Extra.ignore):
    username: str
    password: str


class HikvisionMotionConfig(BaseModel, extra=Extra.ignore):
    type: Literal["hikvision"]
    host: str
    port: int
    auth: Optional[AuthConfig]
    ssl: bool = Field(default=False)


class CameraFeedConfig(BaseModel, extra=Extra.ignore):
    host: str
    port: int = Field(default=554)
    auth: Optional[AuthConfig]
    path: str

    motion: HikvisionMotionConfig

    @property
    def rtsp_url(self) -> str:
        auth = (
            f"{self.auth.username}:{self.auth.password}@"
            if self.auth is not None
            else ""
        )
        return f"rtsp://{auth}{self.host}:{self.port}{self.path}"


class ModelNameEnum(str, Enum):
    yolo_v5_s = "yolo_v5_s"
    tf_ssd_mobilenet_v2 = "tf_ssd_mobilenet_v2"
    tf_ssdlite_mobilenet_v2 = "tf_ssdlite_mobilenet_v2"


class ModelConfig(BaseModel, extra=Extra.ignore):
    name: ModelNameEnum


class ApplicationConfig(BaseModel, extra=Extra.ignore):
    camera_feeds: Dict[str, CameraFeedConfig] = Field(default_factory=dict)
    model: ModelConfig

    @classmethod
    def load_from_file(cls, file_path: Optional[str] = None) -> "ApplicationConfig":
        with open(
            file_path if file_path is not None else get_default_config_file_path()
        ) as f:
            config_data = safe_load(f)

        return cls.parse_obj(config_data if config_data is not None else {})
