import time
import torch
import struct
import numpy as np
from typing import Dict, Any

from utils.utils import ffmpeg_decode_mv_residual, ffmpeg_tensor_to_bytes
from utils.mp4_bin_editor import update_reencode_metadata
from config import config
from .edge import Edge

class EdgeBiSR(Edge):
    def __init__(self):
        super().__init__()
        
    def _downsample_keyframes(self, video_tensor: torch.Tensor, frame_types: list) -> torch.Tensor:
        """对关键帧进行4倍降采样处理"""
        downsampled_frames = []
        
        for i, frame_type in enumerate(frame_types):
            if frame_type == 'I':  # 关键帧
                downsampled_frame = torch.nn.functional.interpolate(
                    video_tensor[i:i+1], 
                    scale_factor=0.25,
                    mode='bicubic',
                    align_corners=False,
                    antialias=True
                )
                downsampled_frames.append(downsampled_frame)
            else:  # 非关键帧保持原样
                downsampled_frames.append(video_tensor[i:i+1])
        
        return torch.cat(downsampled_frames, dim=0)

    def _separate_keyframes_and_others(self, video_tensor: torch.Tensor, frame_types: list):
        """分离关键帧和非关键帧"""
        keyframes = []
        other_frames = []
        keyframe_indices = []
        other_indices = []
        
        for i, frame_type in enumerate(frame_types):
            if frame_type == 'I':
                keyframes.append(video_tensor[i:i+1])
                keyframe_indices.append(i)
            else:
                other_frames.append(video_tensor[i:i+1])
                other_indices.append(i)
        
        keyframes_tensor = torch.cat(keyframes, dim=0) if keyframes else torch.empty(0)
        other_tensor = torch.cat(other_frames, dim=0) if other_frames else torch.empty(0)
        
        return keyframes_tensor, other_tensor, keyframe_indices, other_indices

    def _handle(self, identifier: str, received: bytes, args: Dict) -> bytes:
        if identifier not in self.streamer_status:
            raise RuntimeError("Found no header " + identifier)
        
        print(f'[BiSR] received {len(received)} bytes')
        status = self.streamer_status[identifier]
        video_bytes = status["header"] + received

        # 1. 解码视频并获取帧类型信息
        bg = time.perf_counter()
        ndarray, framerate, mvs, frame_types, residual_arr = ffmpeg_decode_mv_residual(video_bytes, identifier)
        print(f"[BiSR] Raw ndarray shape: {ndarray.shape}, dtype: {ndarray.dtype}, range: [{ndarray.min()}, {ndarray.max()}]")
        
        # 检查几个样本像素值
        sample_frame = ndarray[0]
        print(f"[BiSR] Sample frame: shape={sample_frame.shape}, sample_pixels={sample_frame[0, 0, :]} (first pixel RGB)")
    
        video_tensor = torch.from_numpy(ndarray.transpose(0, 3, 1, 2)).float()
        print(f"[BiSR] Video tensor range: [{video_tensor.min()}, {video_tensor.max()}]")
        
        print(f"[BiSR] 1.decode: {time.perf_counter() - bg:.3f}s, shape: {ndarray.shape}")

        # 2. 分离关键帧和非关键帧
        bg = time.perf_counter()
        keyframes, other_frames, keyframe_indices, other_indices = self._separate_keyframes_and_others(
            video_tensor, frame_types
        )
        print(f"[BiSR] 2.separate: {time.perf_counter() - bg:.3f}s")
        
        # 3. 只对关键帧进行降采样
        bg = time.perf_counter()
        if len(keyframes) > 0:
            downsampled_keyframes = torch.nn.functional.interpolate(
                keyframes, 
                scale_factor=0.25,
                mode='bicubic',
                align_corners=False,
                antialias=True
            )
            print(f"[BiSR] Keyframes downsampled: {keyframes.shape} -> {downsampled_keyframes.shape}")
        else:
            downsampled_keyframes = keyframes
        print(f"[BiSR] 3.downsample: {time.perf_counter() - bg:.3f}s")
        
        # 4. 分别编码
        bg = time.perf_counter()
        
        # 编码降采样的关键帧
        if len(downsampled_keyframes) > 0:
            print(f"[BiSR] Encoding keyframes: shape={downsampled_keyframes.shape}, range=[{downsampled_keyframes.min()}, {downsampled_keyframes.max()}]")
            keyframe_bytes = ffmpeg_tensor_to_bytes(downsampled_keyframes, framerate, identifier + "_keyframes")
            keyframe_final = update_reencode_metadata(video_bytes, keyframe_bytes)
            print(f"[BiSR] Keyframes encoded to {len(keyframe_final)} bytes")
        else:
            keyframe_final = b''

        # 编码原始非关键帧
        if len(other_frames) > 0:
            print(f"[BiSR] Encoding other frames: shape={other_frames.shape}, range=[{other_frames.min()}, {other_frames.max()}]")
            other_bytes = ffmpeg_tensor_to_bytes(other_frames, framerate, identifier + "_others")
            other_final = update_reencode_metadata(video_bytes, other_bytes)
            print(f"[BiSR] Other frames encoded to {len(other_final)} bytes")
        else:
            other_final = b''
        
        print(f"[BiSR] 4.encode: {time.perf_counter() - bg:.3f}s")

        # 5. 构造响应数据
        response_data = struct.pack('>I', len(received)) + received
        response_data += struct.pack('>II', len(keyframe_indices), len(keyframe_final))
        response_data += keyframe_final
        response_data += struct.pack('>II', len(other_indices), len(other_final))
        response_data += other_final
        
        for idx in keyframe_indices:
            response_data += struct.pack('>I', idx)
        for idx in other_indices:
            response_data += struct.pack('>I', idx)

        return response_data