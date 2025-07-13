from .detector import Detector
from .tracker import track, init_tracker
from typing import List, Tuple
import numpy as np
import torch
import math

from config import config
beta = config['beta']
dyn_thres = config['dyn_thres']

def get_inside(block_interval: Tuple[int, int], roi_interval: Tuple[int, int]) -> int:
    left_or_top = max(block_interval[0], roi_interval[0])
    right_or_bottom = min(block_interval[1], roi_interval[1])
    return 0 if left_or_top >= right_or_bottom else right_or_bottom - left_or_top

def calc_IoU(RoI1: Tuple[int, int, int, int], RoI2: Tuple[int, int, int, int]) -> float:
    # 解包矩形坐标
    x1_1, y1_1, x2_1, y2_1 = RoI1
    x1_2, y1_2, x2_2, y2_2 = RoI2

    # 计算相交区域的坐标
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)

    # 检查是否有相交区域
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # 计算相交区域面积
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # 计算两个矩形的面积
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

    # 计算并集面积
    union_area = area1 + area2 - intersection_area

    # 避免除以零的情况
    assert union_area != 0

    # 计算IoU
    iou = intersection_area / union_area
    return iou

def accumulate(mv, x: int, y: int, w: int, h: int) -> float:
    x1, y1 = x, y
    x2, y2 = x + w, y + h
    tot = 0
    for item in mv:
        source, w, h, src_x, src_y, dst_x, dst_y, motion_x, motion_y, motion_scale = item
        x_inside = get_inside((dst_x, dst_x + w), (x1, x2))
        y_inside = get_inside((dst_y, dst_y + h), (y1, y2))
        area = w * h - x_inside * y_inside
        if area > 0:
            module = math.sqrt((motion_x / motion_scale) ** 2 + (motion_y / motion_scale) ** 2)
            tot += module * area
    return tot

class RoIExtractor:
    def __init__(self, tensors: torch.Tensor, ndarray: np.ndarray,
                 types: List[str], mvs, framerate: float,
                 residual_arr: np.ndarray, detector: Detector):
        self.tensors = tensors
        self.ndarray = ndarray
        self.types = types
        self.mvs = mvs
        self.framerate = framerate
        self.residual_arr = residual_arr
        self.detector = detector

        self.diag_length = math.sqrt(ndarray.shape[1] ** 2 + ndarray.shape[2] ** 2)
        self.video_size = ndarray.shape[1] * ndarray.shape[2]

        self.anchor_num = 0
        self.tracker = None
        self.dynamicity = 0

    def _new_anchor(self, idx: int) -> Tuple[int, int]:
        self.anchor_num += 1
        import time
        bg = time.time()
        x1, y1, x2, y2 = self.detector(self.tensors[idx])[0]
        # print(f"{idx}, detect time: {time.time() - bg:.3f}s, result: ({x1}, {y1}, {x2}, {y2})")
        w, h = x2 - x1, y2 - y1
        self.tracker = init_tracker(self.ndarray[idx], (x1, y1, w, h))
        self.dynamicity = 0
        return x1 + w // 2, y1 + h // 2

    def calc_dyn_exclusive(self, idx: int, roi: Tuple[int, int, int, int]) -> np.ndarray:
        x, y, w, h = roi
        x1, y1 = x, y
        x2, y2 = x + w, y + h
        motion = 0
        for item in self.mvs[idx]:
            source, w, h, src_x, src_y, dst_x, dst_y, motion_x, motion_y, motion_scale = item
            x_inside = get_inside((dst_x, dst_x + w), (x1, x2))
            y_inside = get_inside((dst_y, dst_y + h), (y1, y2))
            area = w * h - x_inside * y_inside
            if area > 0:
                module = math.sqrt((motion_x / motion_scale) ** 2 + (motion_y / motion_scale) ** 2)
                motion += module * area
        motion = motion * self.framerate / self.video_size / self.diag_length
        mb_id = 0
        residual = 0
        for mb_y in range(0, self.ndarray.shape[1], 16):
            for mb_x in range(0, self.ndarray.shape[2], 16):
                residual += self.residual_arr[idx][mb_id] * (1 - calc_IoU((mb_x, mb_y, mb_x + 16, mb_y + 16), (x1, y1, x2, y2)))
                mb_id += 1
        return motion + beta * residual

    def _temporal_reuse(self, idx: int) -> Tuple[bool, int, int]:  # bool -> whether reusing succeeded
        success, x, y, w, h = track(self.tracker, self.ndarray[idx])
        # print(f"{idx}, track result: {success}, ({x}, {y}, {w}, {h})")
        if not success:
            return False, 0, 0
        self.dynamicity += self.calc_dyn_exclusive(idx, (x, y, w, h))
        # print(f"dynamicity for {idx}: {self.dynamicity:.3f}, motion: {motion:.3f}, residual: {self.residual_list[idx]:.3f}")
        return (True, x + w // 2, y + h // 2) if self.dynamicity < dyn_thres else (False, 0, 0)

    def run(self) -> List[Tuple[int, int]]:
        roi_list = []
        for idx in range(self.tensors.shape[0]):
            if self.tracker is None or self.types[idx] == 'I':
                x, y = self._new_anchor(idx)
            else:
                success, x, y = self._temporal_reuse(idx)
                if not success:
                    x, y = self._new_anchor(idx)
            roi_list.append((int(x), int(y)))
        print(f"anchor num: {self.anchor_num}")
        return roi_list

    # def calc_dyn(self, idx: int) -> float:
    #     motion = 0
    #     for item in self.mvs[idx]:
    #         source, w, h, src_x, src_y, dst_x, dst_y, motion_x, motion_y, motion_scale = item
    #         area = w * h
    #         module = math.sqrt((motion_x / motion_scale) ** 2 + (motion_y / motion_scale) ** 2)
    #         motion += module * area
    #     motion = motion * self.framerate / self.video_size / self.diag_length
    #     return motion + beta * self.residual_list[idx]
    #
    # def _get_anchors(self, i_frames: List[int]) -> List[int]:
    #     dynamicity = 0
    #     anchor = []
    #     for idx in range(self.tensors.shape[0]):
    #         if idx in i_frames:
    #             dynamicity = 0
    #             anchor.append(idx)
    #         else:
    #             dynamicity += self.calc_dyn(idx)
    #             if dynamicity >= dyn_thres:
    #                 dynamicity = 0
    #                 anchor.append(idx)
    #     return anchor
    #
    # def run(self) -> List[Tuple[int, int]]:
    #     i_frames = []
    #     for idx in range(self.tensors.shape[0]):
    #         if self.types[idx] == 'I':
    #             i_frames.append(idx)
    #     anchors = self._get_anchors(i_frames)
    #     anchor_tensors = self.tensors[anchors]
    #     assert anchor_tensors.dim() == 4, "the dim of anchor frames tensor should be 4 (NCHW)"
    #     proposal_list = self.detector(anchor_tensors)
    #     assert len(proposal_list) == len(anchors)
    #
    #     ptr = 0
    #     roi_list = []
    #     tracker = None
    #     for idx in range(self.tensors.shape[0]):
    #         if ptr < len(anchors) and idx == anchors[ptr]:
    #             x1, y1, x2, y2 = proposal_list[ptr]
    #             w, h = x2 - x1, y2 - y1
    #             roi_list.append((int(x1 + w // 2), int(y1 + h // 2)))
    #             tracker = init_tracker(self.ndarray[idx], proposal_list[ptr])
    #             ptr += 1
    #         else:
    #             success, x, y, w, h = track(tracker, self.ndarray[idx])
    #             if not success:
    #                 x1, y1, x2, y2 = self.detector(self.tensors[idx])[0]
    #                 x, y, w, h = x1, y1, x2 - x1, y2 - y1
    #                 print(f"Track failed, RPN: {(x1, y1, x2, y2)}")
    #             roi_list.append((int(x + w // 2), int(y + h // 2)))
    #
    #     print(f"anchor num: {len(anchors)}")
    #     return roi_list
