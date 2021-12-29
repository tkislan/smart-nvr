from typing import Any, Callable, Dict, List, Optional

from pyhik.hikvision import HikCamera

from ...app_config import HikvisionMotionConfig

MOTION_EVENTS = {
    "Motion",
    "Line Crossing",
    "Field Detection",
    "Tamper Detection",
    "PIR Alarm",
    "Scene Change Detection",
    "Exiting Region",
    "Entering Region",
}


# class HikvisionEventConfig:
#     host: str
#     port: int
#     username: str
#     password: str
#     ssl: bool

#     def __init__(
#         self, host: str, port: int, username: str, password: str, ssl: bool = False
#     ):
#         self.host = host
#         self.port = port
#         self.username = username
#         self.password = password
#         self.ssl = ssl


def event_states_has_motion(event_states: Dict[str, List[List[Any]]]) -> bool:
    for sensor, channel_list in event_states.items():
        if sensor in MOTION_EVENTS:
            for channel in channel_list:
                if channel[0] == True:
                    return True

    return False


def has_motion(cam: HikCamera) -> bool:
    event_states = cam.current_event_states

    return event_states_has_motion(event_states) if event_states is not None else False
    # return True  # TODO - hack


class HikvisionMotionDetection:
    def __init__(self, config: HikvisionMotionConfig):
        self._cam = HikCamera(
            f"http://{config.host}",
            config.port,
            config.auth.username if config.auth is not None else None,
            config.auth.password if config.auth is not None else None,
        )

        for sensor, channel_list in self._cam.current_event_states.items():
            for channel in channel_list:
                id = f"{self._cam.cam_id}.{sensor}.{channel[1]}"
                self._cam.add_update_callback(self._update_callback, id)

        self._callback: Optional[Callable[[bool], None]] = None

    def set_callback(self, callback: Callable[[bool], None]):
        self._callback = callback

    def start(self):
        self.trigger_callback(has_motion(self._cam))
        self._cam.start_stream()

    def stop(self):
        self._cam.disconnect()

    def trigger_callback(self, motion: bool):
        if self._callback is not None:
            self._callback(motion)

    def _update_callback(self, id: str):
        print(f"Callback signal from: {id}")

        motion = has_motion(self._cam)
        print(f"Has motion: {motion}")

        self.trigger_callback(motion)
