from typing import List
import numpy as np
from utils.utils import decord_bytes2numpy
from typing import Dict, Any

class Client:
    def __init__(self, init_buffer: int):
        self.cached = []
        self.played = 0
        self.buffer = init_buffer
        self.init_chunk = None # only video stream

    def receive(self, received: bytes, is_init: bool):
        if is_init:
            self.init_chunk = received
        else:
            handled = self.__handle(received)
            self.cached.append(handled)

    def save_video_rebuffer(self, lag_info: List[float], filename: str):
        # TODO
        pass

    def __handle(self, received: bytes) -> np.ndarray:
        """
        If the architecture needs to do something at the client, override this function.
        Process the chunk and return a numpy array (NCHW, RGB). The array will be automatically appended to cache.

        If this function is not overridden, it will concatenate the init_chunk and whatever it receives from the server,
        and try to decode the bytes.

        :param received: Binary data from server
        """
        video_bytes = self.init_chunk + received
        video_np, fps = decord_bytes2numpy(video_bytes, threads=1)
        return video_np

    def init_arg(self) -> Dict[str, Any]:
        """
        If the architecture needs to send some arguments to the edge at the beginning, override this function.
        """
        return {}

    def common_arg(self) -> Dict[str, Any]:
        """
        If the architecture needs to send some arguments to the edge every time it sends a request, override this function.
        """
        return {}