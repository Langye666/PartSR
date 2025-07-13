import cv2
import numpy as np
from typing import List, Dict, Any, Tuple

from utils.utils import decord_bytes2numpy

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
            handled = self._handle(received)
            self.cached.append(handled)

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
                scaled_viz = self.cached[i].transpose(1, 2, 0)[:, :, ::-1].copy()  # RGB to BGR
                scaled_viz = np.ascontiguousarray(scaled_viz, dtype=np.uint8)
                x, y, w, h = gt_roi[global_idx]
                x1_gt = x * width
                y1_gt = y * height
                x2_gt = (x + w) * width
                y2_gt = (y + h) * height
                cv2.rectangle(scaled_viz, (x1_gt, y1_gt), (x2_gt, y2_gt), (0, 0, 255), 2)  # 绘制真实ROI (红色框)
                # 添加文字说明
                cv2.putText(scaled_viz, f"Frame: {i}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(scaled_viz, "Ground Truth ROI", (x1_gt, y1_gt - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                # 写入视频帧
                video_writer.write(scaled_viz)
        video_writer.release()

    def _handle(self, received: bytes) -> np.ndarray:
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