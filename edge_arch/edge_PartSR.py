import time
import torch
import ctypes
import struct

from torchvision import transforms
from torch import multiprocessing as mp
from typing import Dict, Any

from super_resolution.infer import generate_sr_patch, mp_server_sr
from utils.mp4_bin_editor import update_reencode_metadata
from utils.utils import ffmpeg_decode_mv_residual, ffmpeg_tensor_to_bytes, roi_center_to_xyxy
from classifier.classifier import Classifier
from extractor.extractor import RoIExtractor
from extractor.detector import Detector
from scheduler.scheduler import Scheduler
from config import config
from .edge import Edge


def crop_rois(video_tensor, bboxes):
    """
    从NCHW视频张量中批量裁剪指定ROI区域
    Args:
        video_tensor: torch.Tensor 形状为(N, C, H, W)
        bboxes: torch.Tensor 形状为(N, 4)，格式为(x1, y1, x2, y2)
    Returns:
        cropped: torch.Tensor 形状为(N, C, crop_h, crop_w)
    """
    device = video_tensor.device
    N, C, H, W = video_tensor.shape

    # 计算裁剪尺寸 (所有帧尺寸相同)
    crop_h = bboxes[0, 3] - bboxes[0, 1] + 1
    crop_w = bboxes[0, 2] - bboxes[0, 0] + 1

    # 生成行/列索引网格
    row_indices = torch.arange(crop_h, device=device).view(1, crop_h, 1)
    col_indices = torch.arange(crop_w, device=device).view(1, 1, crop_w)

    # 计算绝对索引 (广播机制)
    abs_y = bboxes[:, 1].view(N, 1, 1) + row_indices  # (N, crop_h, 1)
    abs_x = bboxes[:, 0].view(N, 1, 1) + col_indices  # (N, 1, crop_w)

    # 扩展通道维度
    abs_y = abs_y.expand(N, crop_h, crop_w).unsqueeze(1)  # (N, 1, crop_h, crop_w)
    abs_x = abs_x.expand(N, crop_h, crop_w).unsqueeze(1)  # (N, 1, crop_h, crop_w)

    # 生成批次索引
    batch_idx = torch.arange(N, device=device).view(N, 1, 1, 1)
    batch_idx = batch_idx.expand(N, 1, crop_h, crop_w)  # (N, 1, crop_h, crop_w)

    # 组合索引 (N, 1, crop_h, crop_w)
    indices = torch.cat([batch_idx, abs_y, abs_x], dim=1)

    # 按索引收集像素 (高效实现)
    cropped = video_tensor.permute(0, 2, 3, 1)  # (N, H, W, C)
    cropped = cropped[indices[:, 0], indices[:, 1], indices[:, 2], :]
    cropped = cropped.permute(0, 3, 1, 2)  # (N, C, crop_h, crop_w)

    for i in range(video_tensor.shape[0]):
        # TODO: delete this assert
        assert cropped[i] == video_tensor[i, :, bboxes[i, 0]:bboxes[i, 2], bboxes[i, 1]:bboxes[i, 3]]

    return cropped

class WaitHandler:
    def __init__(self):
        self.time_lock = mp.Lock()
        self.shared_sr_wait_tot = mp.Value(ctypes.c_double, 0)
        self.shared_sr_wait_now = mp.Value(ctypes.c_double, 0)
        self.shared_sr_wait_queue = mp.Queue()
        self.shared_sr_wait_current_begin = mp.Value(ctypes.c_double, 0)

    def add_wait(self, wait_len):
        with self.time_lock:
            if self.shared_sr_wait_now.value == 0:
                self.shared_sr_wait_tot.value = wait_len
                self.shared_sr_wait_current_begin.value = time.perf_counter()
            else:
                self.shared_sr_wait_queue.put(wait_len)
            self.shared_sr_wait_tot.value += wait_len

    def remove_wait(self):
        with self.time_lock:
            if self.shared_sr_wait_queue.empty():
                self.shared_sr_wait_tot.value = 0
                self.shared_sr_wait_now.value = 0
            else:
                self.shared_sr_wait_tot.value -= self.shared_sr_wait_now.value
                self.shared_sr_wait_now.value = self.shared_sr_wait_queue.get()
                self.shared_sr_wait_current_begin.value = time.perf_counter()

    def get_wait(self):
        with self.time_lock:
            elapsed = time.perf_counter() - self.shared_sr_wait_current_begin.value
            if elapsed > self.shared_sr_wait_now.value:
                elapsed = self.shared_sr_wait_now.value
            return self.shared_sr_wait_tot.value - elapsed

class EdgePartSR(Edge):
    def __init__(self):
        mp.set_start_method('spawn', force=True)
        torch.multiprocessing.set_sharing_strategy('file_descriptor')
        super().__init__()

        sr_result_dict = self.manager.dict()
        parent_conn, child_conn = mp.Pipe()
        self.sr_queue = mp.Queue()

        self.sr_model_list = config["sr_models"]
        self.sr_sizes = config["sr_sizes"]
        self.sr_model_number = len(self.sr_model_list)
        self.sr_result = sr_result_dict

        self.classifier = Classifier("models/classifier.pth", num_classes=3)
        self.detector = Detector("extractor/best_rpn.pth")
        self.scheduler = Scheduler("")

        self.sr_wait = WaitHandler()
        # schedule by order
        self.schedule_lock = mp.Lock()
        self.shared_sr_number = mp.Value(ctypes.c_int, 0)
        self.shared_sr_send = mp.Value(ctypes.c_int, 1) # only use in __send_task
        self.shared_not_send_list = self.manager.dict()
        self.client_sr_speed = {}

        self.hist_encode = mp.Value(ctypes.c_double, 0)

        try:
            process_b = mp.Process(target=mp_server_sr, args=(self.sr_queue, child_conn))
            process_b.start()
            self.sr_latency_param = parent_conn.recv() # benchmark result from subprocess
            parent_conn.close()
        except Exception as e:
            print(f"An error occurred when creating EdgePartSR: {e}")
            self.sr_queue.put((None, None, None, None))
            process_b.terminate()
            exit(-1)

    def init_args(self, identifier: str, args: Dict[str, Any]):
        self.client_sr_speed[identifier] = (args['k'], args['b'])

    def __take_number(self): # call with schedule_lock, no parallel
        self.shared_sr_number.value += 1
        return self.shared_sr_number.value

    def __send_task(self, number, sr_tuple):
        self.shared_sr_send.acquire()
        send_now = self.shared_sr_send.value
        if number == send_now:
            self.sr_queue.put(sr_tuple)
            send_now += 1
            while send_now in self.shared_not_send_list:
                self.sr_queue.put(self.shared_not_send_list[send_now])
                send_now += 1
            self.shared_sr_send.value = send_now
        else:
            self.shared_not_send_list[number] = sr_tuple
        self.shared_sr_send.release()

    def __update_encode_time(self, new_time):
        self.hist_encode.acquire()
        if self.hist_encode.value == 0:
            self.hist_encode.value = new_time
        else:
            self.hist_encode.value = (new_time + self.hist_encode.value) / 2
        self.hist_encode.release()

    def _handle(self, identifier: str, received: bytes, args: Dict) -> bytes:
        if identifier not in self.streamer_status:
            raise RuntimeError("Found no header " + identifier)
        print(f'[PartSR] received {len(received)} bytes')
        status = self.streamer_status[identifier]
        video_bytes = status["header"] + received

        # 1. decode to tensor & extract MV, RE
        bg = time.perf_counter()
        ndarray, framerate, mvs, types, residual_arr = ffmpeg_decode_mv_residual(video_bytes, identifier)
        tensors_div_255 = torch.from_numpy(ndarray.transpose(0, 3, 1, 2)).float().div(255)  # NHWC → NCHW
        tensors_normalized = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(
            tensors_div_255)
        print(f"[PartSR] 1.decode: {time.perf_counter() - bg:.3f}s, shape: {ndarray.shape}")

        # 2. classify the stream if it hasn't been classified
        if "class" not in status:
            bg = time.perf_counter()
            classified = self.classifier(tensors_normalized)
            self.streamer_status[identifier]["class"] = classified
            print(f"[PartSR] 2.classify: {time.perf_counter() - bg:.3f}s, classified: {classified}")

        # 3. extract RoI
        bg = time.perf_counter()
        extractor = RoIExtractor(tensors_normalized, ndarray, types, mvs, framerate, residual_arr, self.detector)
        roi_array = extractor.run()
        roi_xyxy_max = roi_center_to_xyxy(roi_array, max(self.sr_sizes), (ndarray.shape[1], ndarray.shape[2]))
        print(f"[PartSR] 3.extract RoI: {time.perf_counter() - bg:.3f}s")  # , roi: {roi_array}")

        # 4. choose proper SR action according to environment and video features
        bg = time.perf_counter()
        with self.schedule_lock:
            wait_time = self.sr_wait.get_wait()
            number_taken = self.__take_number()
            env = {
                'bandwidth': self.streamer_status[identifier]["min_bandwidth"],
                'sr_wait': wait_time,
                'hist_encode': self.hist_encode,
            }
            SR_size, action = self.scheduler(env, crop_rois(tensors_normalized, roi_xyxy_max))
            if action < self.sr_model_number:
                k, b = self.sr_latency_param[action]
                self.sr_wait.add_wait(k * SR_size + b)
        if SR_size == max(self.sr_sizes):
            roi_xyxy = roi_xyxy_max
        else:
            roi_xyxy = roi_center_to_xyxy(roi_array, SR_size, (ndarray.shape[1], ndarray.shape[2]))
        print(f"[PartSR] 4.schedule: {time.perf_counter() - bg:.3f}s (wait {wait_time}s), action: {action}, SR_size: {SR_size}")

        if action < self.sr_model_number:  # 5. if SR at the server
            bg = time.perf_counter()
            tensor_for_SR = generate_sr_patch(tensors_div_255, roi_xyxy)
            sr_parent_pipe, sr_child_pipe = mp.Pipe()
            self.__send_task(number_taken, (sr_child_pipe, tensor_for_SR, SR_size, action))
            sr_tensor = sr_parent_pipe.recv()
            self.sr_wait.remove_wait()
            sr_parent_pipe.close()
            print(f"[PartSR] 5.super resolution: {time.perf_counter() - bg:.3f}s, tensor: {sr_tensor.shape}")

            bg = time.perf_counter()
            reencode = ffmpeg_tensor_to_bytes(sr_tensor, framerate, identifier)
            final_video = update_reencode_metadata(video_bytes, reencode)
            encode_time = time.perf_counter() - bg
            self.__update_encode_time(encode_time)
            print(f"[PartSR] 6.re-encode {encode_time:.3f}s, encoded patch size: {len(final_video)}")

            response_data = struct.pack('>BII', int(0), SR_size, len(final_video)) + final_video
            for i in range(ndarray.shape[0]):
                response_data += struct.pack('>II', roi_xyxy[i][0], roi_xyxy[i][1])

        else:  # 5. if SR at the client: send RoI info
            response_data = struct.pack('>BI', int(1), SR_size)
            for i in range(ndarray.shape[0]):
                response_data += struct.pack('>II', roi_xyxy[i][0], roi_xyxy[i][1])

        return struct.pack('>I', len(received)) + received + response_data