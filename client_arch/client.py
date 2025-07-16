import math

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
                frame = self.cached[i].transpose(1, 2, 0)[:, :, ::-1].copy()  # RGB to BGR
                frame = np.ascontiguousarray(frame, dtype=np.uint8)
                if annotate:
                    x, y, w, h = gt_roi[global_idx]
                    x1_gt = x * width
                    y1_gt = y * height
                    x2_gt = (x + w) * width
                    y2_gt = (y + h) * height
                    cv2.rectangle(frame, (x1_gt, y1_gt), (x2_gt, y2_gt), (0, 0, 255), 2)  # 绘制真实ROI (红色框)
                    cv2.putText(frame, "Ground Truth ROI", (x1_gt, y1_gt - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                cv2.putText(frame, f"Frame: {i}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                video_writer.write(frame)
            if lag_info[idx] > 0:
                # 计算卡顿帧数（向上取整）
                lag_frames = math.ceil(lag_info[idx] * 30)
                last_frame = frame.copy()  # 获取当前块的最后一帧

                # 在卡顿期间重复最后一帧并添加加载动画
                for j in range(lag_frames):
                    lag_frame = last_frame.copy()

                    # 在右下角添加加载动画（三个动态小点）
                    dot_radius = 5
                    dot_spacing = 15
                    start_x = width - 80
                    start_y = height - 30

                    # 计算当前帧的点状态（每10帧循环一次动画）
                    phase = (j % 10) // 3  # 0-2: 哪个点高亮

                    # 绘制三个点
                    for k in range(3):
                        color = (255, 255, 255) if k == phase else (100, 100, 100)
                        center = (start_x + k * dot_spacing, start_y)
                        cv2.circle(lag_frame, center, dot_radius, color, -1)

                    # 添加"Buffering..."文字
                    cv2.putText(lag_frame, f"Buffering: {lag_info[idx]:.1f}s",
                                (width - 150, height - 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

                    video_writer.write(lag_frame)
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