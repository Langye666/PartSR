from torch.multiprocessing import Manager
from typing import Dict, Any

class Edge:
    def __init__(self):
        self.manager = Manager()
        self.streamers = self.manager.dict() # save header and other info of different streams

    def receive(self, identifier: str, received: bytes, args: Dict, is_init: bool) -> bytes:
        print(f"[Edge] received: {identifier}, is_init: {is_init}, args: {args}")
        if is_init:
            if identifier not in self.streamers:
                self.streamers[identifier] = {'header': received}
            else:
                self.streamers[identifier]['header'] = received
            self.init_args(identifier, args)
            return received
        else:
            return self._handle(identifier, received, args)

    def init_args(self, identifier: str, args: Dict[str, Any]):
        pass

    def _handle(self, identifier: str, received: bytes, args) -> bytes:
        """
        Override this function to handle the video at edge server

        :param identifier: Identify the video to distinguish different streamers
        :param received: Binary data from server
        """
        return self.streamers[identifier]['header'] + received
