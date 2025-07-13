import struct
from typing import Dict, Any

import numpy as np
import torch

from super_resolution.infer import generate_sr_patch, Inferrer
from utils.utils import decord_bytes2numpy
from config import config
from .client import Client

width = config['video_width'] * 4
height = config['video_height'] * 4

class ClientPartSR(Client):
    def __init__(self, *args):
        super().__init__(*args)
        self.inferrer = Inferrer(config['sr_models'])

    def __handle(self, video_chunk: bytes) -> np.ndarray:
        lr_len, = struct.unpack('>I', video_chunk[:4])
        lr_bytes = video_chunk[4:4 + lr_len]
        lr_numpy, fps = decord_bytes2numpy(lr_bytes)
        lr_tensor = torch.from_numpy(lr_numpy)
        upscaled = torch.nn.functional.interpolate(
            lr_tensor,
            scale_factor=(width, height),
            mode='bicubic',
            align_corners=False,
            antialias=False
        ).numpy()
        ptr = 4 + lr_len
        SR_type, SR_size = struct.unpack('>BI', video_chunk[ptr:ptr + 5])
        scaled_size = SR_size * 4
        ptr += 5
        if SR_type == 0:
            patch_len, = struct.unpack('>I', video_chunk[ptr:ptr + 4])
            ptr += 4
            patch_bytes = video_chunk[ptr:ptr + patch_len]
            patch_numpy, fps_ = decord_bytes2numpy(patch_bytes)
            assert fps_ == fps and patch_numpy.shape[3] == SR_size * 4
            ptr += patch_len
            for idx, i in enumerate(range(ptr, len(video_chunk), 8)):
                x, y = struct.unpack('>II', video_chunk[i:i+8])
                upscaled[idx, :, x*4:x*4+scaled_size, y*4:y*4+scaled_size] = patch_numpy[idx]
        else:
            roi_xyxy = []
            for i in range(ptr, len(video_chunk), 8):
                x, y = struct.unpack('>II', video_chunk[i:i+8])
                roi_xyxy.append([x, y, x + SR_size, y + SR_size])
            roi_xyxy = np.array(roi_xyxy)
            tensor_for_SR = generate_sr_patch(lr_tensor.float() / 255, roi_xyxy)
            patch_numpy = self.inferrer(tensor_for_SR, SR_size, 0).numpy()
            scaled_roi = roi_xyxy * 4
            for idx in range(patch_numpy.shape[0]):
                upscaled[idx, :, scaled_roi[idx, 0]: scaled_roi[idx, 2], scaled_roi[idx, 1]: scaled_roi[idx, 3]] = patch_numpy[idx]
        return upscaled

    def init_arg(self) -> Dict[str, Any]:
        k, b = self.inferrer.run_benchmark()
        return {
            'k': k,
            'b': b
        }

    def common_arg(self) -> Dict[str, Any]:
        return {
            'buffer': self.buffer
        }
