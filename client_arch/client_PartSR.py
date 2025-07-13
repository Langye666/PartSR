import struct
import cv2
import numpy as np
import torch
from typing import Dict, Any, List, Tuple

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
        self.sr_pos = []

    def save_video_rebuffer(self, lag_info: List[float], filename: str, annotate: bool, gt_roi: List[Tuple]=None):
        if annotate:
            assert gt_roi is not None, "must set gt_roi to annotate RoI"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        width = self.cached[0].shape[3]
        height = self.cached[0].shape[2]
        video_writer = cv2.VideoWriter(filename, fourcc, 30.0, (width, height))
        for idx, clip in enumerate(self.cached):
            for i in range(clip.shape[0]):
                global_idx = clip.shape[0] * idx + i
                # 转换为OpenCV格式 (HWC, BGR)
                scaled_viz = self.cached[idx][i].transpose(1, 2, 0)[:, :, ::-1].copy()  # RGB to BGR
                scaled_viz = np.ascontiguousarray(scaled_viz, dtype=np.uint8)
                x, y, w, h = gt_roi[global_idx]
                x1_gt = int((x - w / 2) * width)
                y1_gt = int((y - h / 2) * height)
                x2_gt = x1_gt + int(w * width)
                y2_gt = y1_gt + int(h * height)
                cv2.rectangle(scaled_viz, (x1_gt, y1_gt), (x2_gt, y2_gt), (0, 0, 255), 2)  # 绘制真实ROI (红色框)
                x, y, SR_size = self.sr_pos[global_idx]
                x1 = x * 4
                y1 = y * 4
                x2 = (x + SR_size) * 4
                y2 = (y + SR_size) * 4
                cv2.rectangle(scaled_viz, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绘制检测ROI (绿色框)
                # 添加文字说明
                cv2.putText(scaled_viz, f"Frame: {i}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(scaled_viz, "Ground Truth ROI", (x1_gt, y1_gt - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                cv2.putText(scaled_viz, "Ground Truth ROI", (x1_gt, y1_gt - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                # 写入视频帧
                video_writer.write(scaled_viz)
        video_writer.release()

    def _handle(self, video_chunk: bytes) -> np.ndarray:
        lr_len, = struct.unpack('>I', video_chunk[:4])
        lr_bytes = video_chunk[4:4 + lr_len]
        lr_numpy, fps = decord_bytes2numpy(self.init_chunk + lr_bytes)
        lr_tensor = torch.from_numpy(lr_numpy)
        upscaled = torch.nn.functional.interpolate(
            lr_tensor,
            size=(height, width),
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
                upscaled[idx, :, y*4:y*4+scaled_size, x*4:x*4+scaled_size] = patch_numpy[idx]
                self.sr_pos.append((x, y, SR_size))
        else:
            roi_xyxy = []
            for i in range(ptr, len(video_chunk), 8):
                x, y = struct.unpack('>II', video_chunk[i:i+8])
                roi_xyxy.append([x, y, x + SR_size, y + SR_size])
                self.sr_pos.append((x, y, SR_size))
            roi_xyxy = np.array(roi_xyxy)
            tensor_for_SR = generate_sr_patch(lr_tensor.float() / 255, roi_xyxy)
            patch_numpy = (self.inferrer(tensor_for_SR, SR_size, 0) * 255).numpy().astype(np.uint8)
            scaled_roi = roi_xyxy * 4
            for idx in range(patch_numpy.shape[0]):
                upscaled[idx, :, scaled_roi[idx, 1]: scaled_roi[idx, 3], scaled_roi[idx, 0]: scaled_roi[idx, 2]] = patch_numpy[idx]
        return upscaled

    def init_arg(self) -> Dict[str, Any]:
        k, b = self.inferrer.run_benchmark()[0]
        return {
            'k': k,
            'b': b
        }

    def common_arg(self) -> Dict[str, Any]:
        return {
            'buffer': self.buffer
        }
