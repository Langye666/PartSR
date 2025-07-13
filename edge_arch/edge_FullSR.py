import torch
import multiprocessing as mp

from basicsr.archs.edsr_arch import EDSR

from utils.utils import decord_bytes2numpy
from .edge import Edge
from config import config, device_name

sr_model_path = config['sr_model'][0].lower()

def process_sr(sr_queue):
    if not sr_model_path.startswith('edsr'):
        raise RuntimeError('only EDSR model is supported')
    if 's' in sr_model_path[4:]:
        model = EDSR(3, 3, 32, 8)
    elif 'm' in sr_model_path[4:]:
        model = EDSR(3, 3, 64, 16)
    elif 'l' in sr_model_path[4:]:
        model = EDSR(3, 3, 256, 32, res_scale=0.1)
    else:
        raise NotImplementedError(f"Cannot resolve EDSR type (M/L): {sr_model_path}")
    state_dict = torch.load(sr_model_path, map_location="cpu", weights_only=True)
    print(model.load_state_dict(state_dict['params'], strict=True))
    model = model.to(device_name)
    for param in model.parameters():
        param.requires_grad = False
    while True:
        tmp_pipe, tensors = sr_queue.get()
        if tmp_pipe is None:
            break
        with torch.no_grad():
            sr_tensors = model(tensors)
        tmp_pipe.send(sr_tensors)
        tmp_pipe.close()


class EdgeFullSR(Edge):
    def __init__(self):
        mp.set_start_method('spawn', force=True)
        torch.multiprocessing.set_sharing_strategy('file_descriptor')
        super().__init__()
        self.sr_queue = mp.Queue()
        try:
            process_b = mp.Process(target=process_sr, args=(self.sr_queue,))
            process_b.start()
        except Exception as e:
            print(f"An error occurred when creating EdgePartSR: {e}")
            self.sr_queue.put((None, None))
            process_b.terminate()
            exit(-1)

    def _handle(self, identifier: str, received: bytes, args) -> bytes:
        if identifier not in self.streamer_status:
            raise RuntimeError("Found no header " + identifier)
        status = self.streamer_status[identifier]
        video_bytes = status["header"] + received

        lr_numpy, fps = decord_bytes2numpy(video_bytes)
        lr_tensor = torch.from_numpy(lr_numpy).float() / 255
        sr_parent_pipe, sr_child_pipe = mp.Pipe()
        self.sr_queue.put((sr_child_pipe, lr_tensor))
        sr_tensor = sr_child_pipe.recv()
