from torch.multiprocessing import Manager
from typing import Dict, Any

class Edge:
    def __init__(self):
        self.manager = Manager()
        self.streamer_status = self.manager.dict() # save header and other info of different streams

    def receive(self, identifier: str, received: bytes, args, is_init: bool) -> bytes:
        if is_init:
            self.streamer_status[identifier]['header'] = received
            self.init_args(identifier, args)
            return received
        else:
            return self.__handle(identifier, received, args)

    def init_args(self, identifier: str, args: Dict[str, Any]):
        pass

    def __handle(self, identifier: str, received: bytes, args) -> bytes:
        """
        Override this function to handle the video at edge server

        :param identifier: Identify the video to distinguish different streamers
        :param received: Binary data from server
        """
        return self.streamer_status[identifier]['header'] + received
