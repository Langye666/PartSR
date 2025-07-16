import struct
import torch
import numpy as np
from typing import Dict, Any

from super_resolution.infer import Inferrer
from utils.utils import decord_bytes2numpy
from config import config
from .client import Client

class ClientBiSR(Client):
    def __init__(self, *args):
        super().__init__(*args)
        # 使用EDSR_S_x4_0.8799.pth模型
        self.inferrer = Inferrer([config['sr_models'][0]])
        
    def _super_resolution_keyframes(self, keyframes_tensor: torch.Tensor) -> torch.Tensor:
        """对关键帧进行超分处理"""
        if len(keyframes_tensor) == 0:
            return keyframes_tensor
            
        print(f"[ClientBiSR] Input keyframes shape: {keyframes_tensor.shape}")
        
        # 转换为模型输入格式
        input_tensor = keyframes_tensor.float() / 255.0
        print(f"[ClientBiSR] Keyframes range before normalization: [{keyframes_tensor.min()}, {keyframes_tensor.max()}]")
        print(f"[ClientBiSR] Normalized input range: [{input_tensor.min()}, {input_tensor.max()}]")
        
        # 使用EDSR模型进行4倍超分
        target_width = keyframes_tensor.shape[-1] * 4
        sr_tensor = self.inferrer(input_tensor, target_width, 0)
        
        print(f"[ClientBiSR] EDSR raw output range: [{sr_tensor.min()}, {sr_tensor.max()}]")
        
        # 关键修复：使用tanh激活函数而不是简单clamp
        sr_tensor = torch.clamp(sr_tensor, -0.1, 1.1)  # 允许轻微超出范围
        sr_tensor = (sr_tensor + 0.1) / 1.2  # 归一化到[0,1]
        print(f"[ClientBiSR] After sigmoid: [{sr_tensor.min()}, {sr_tensor.max()}]")
        
        # 目标尺寸调整
        target_height = keyframes_tensor.shape[-2] * 4
        target_width = keyframes_tensor.shape[-1] * 4
        
        current_height = sr_tensor.shape[-2]
        current_width = sr_tensor.shape[-1]
        
        if current_height != target_height or current_width != target_width:
            print(f"[ClientBiSR] Adjusting size from {current_height}x{current_width} to {target_height}x{target_width}")
            sr_tensor = torch.nn.functional.interpolate(
                sr_tensor,
                size=(target_height, target_width),
                mode='bilinear',
                align_corners=False
            )
        
        # 转换回0-255范围
        sr_tensor = (sr_tensor * 255.0)
        print(f"[ClientBiSR] After scaling: [{sr_tensor.min()}, {sr_tensor.max()}]")
        print(f"[ClientBiSR] Final SR shape: {sr_tensor.shape}")
        
        return sr_tensor

    def _merge_frames(self, keyframes: torch.Tensor, other_frames: torch.Tensor, 
                     keyframe_indices: list, other_indices: list, total_frames: int) -> torch.Tensor:
        """合并关键帧和非关键帧，处理尺寸不匹配问题"""
        
        print(f"[ClientBiSR] Merging frames:")
        print(f"  Keyframes: {keyframes.shape if len(keyframes) > 0 else 'empty'}")
        print(f"  Other frames: {other_frames.shape if len(other_frames) > 0 else 'empty'}")
        
        # 确保tensor格式正确 (NCHW)
        if len(keyframes) > 0:
            keyframes = self._ensure_nchw_format(keyframes)
        if len(other_frames) > 0:
            other_frames = self._ensure_nchw_format(other_frames)
        
        # 确定目标尺寸
        if len(keyframes) > 0:
            target_C, target_H, target_W = keyframes.shape[1], keyframes.shape[2], keyframes.shape[3]
            print(f"[ClientBiSR] Using keyframes dimensions: {target_C}x{target_H}x{target_W}")
        elif len(other_frames) > 0:
            target_C, target_H, target_W = other_frames.shape[1], other_frames.shape[2], other_frames.shape[3]
            print(f"[ClientBiSR] Using other frames dimensions: {target_C}x{target_H}x{target_W}")
        else:
            # 如果都为空，返回空tensor
            return torch.empty(0, 3, 720, 1280)
            
        merged_tensor = torch.zeros(total_frames, target_C, target_H, target_W)
        
        # 填充关键帧
        for i, idx in enumerate(keyframe_indices):
            if i < len(keyframes):
                merged_tensor[idx] = keyframes[i]
        
        # 填充非关键帧 - 处理尺寸不匹配
        for i, idx in enumerate(other_indices):
            if i < len(other_frames):
                other_frame = other_frames[i]
                
                # 检查尺寸是否匹配
                if other_frame.shape[-2:] != (target_H, target_W):
                    print(f"[ClientBiSR] Resizing other frame from {other_frame.shape[-2:]} to {target_H}x{target_W}")
                    
                    # 调整非关键帧尺寸以匹配超分后的关键帧
                    other_frame = torch.nn.functional.interpolate(
                        other_frame.unsqueeze(0),
                        size=(target_H, target_W),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)
                
                merged_tensor[idx] = other_frame
        
        print(f"[ClientBiSR] Final merged shape: {merged_tensor.shape}")
        return merged_tensor

    def _ensure_nchw_format(self, tensor: torch.Tensor) -> torch.Tensor:
        """确保tensor格式是NCHW"""
        if tensor.dim() == 4:
            N, dim1, dim2, dim3 = tensor.shape
            
            # 如果第二个维度不是3（RGB通道），可能需要重新排列
            if dim1 != 3 and dim2 == 3:
                # 从NHCW转换为NCHW
                tensor = tensor.permute(0, 2, 1, 3)
                print(f"[ClientBiSR] Converted tensor from NHCW to NCHW: {tensor.shape}")
            elif dim1 != 3 and dim3 == 3:
                # 从NHWC转换为NCHW
                tensor = tensor.permute(0, 3, 1, 2)
                print(f"[ClientBiSR] Converted tensor from NHWC to NCHW: {tensor.shape}")
                
        return tensor

    def _Client__handle(self, video_chunk: bytes) -> np.ndarray:
        """
        处理视频数据块
        对于BiSR架构，这里应该解析服务器发送的分离后的关键帧和非关键帧数据
        """
        try:
            # 首先尝试解析BiSR格式的数据
            if len(video_chunk) < 8:
                print("[ClientBiSR] Data too short, using default processing")
                return self._default_handle(video_chunk)
            
            # 尝试解析BiSR数据格式
            ptr = 0
            
            # 原始数据长度
            if len(video_chunk) < ptr + 4:
                return self._default_handle(video_chunk)
            
            original_len, = struct.unpack('>I', video_chunk[ptr:ptr+4])
            ptr += 4
            
            # 跳过原始数据
            if len(video_chunk) < ptr + original_len:
                return self._default_handle(video_chunk)
            ptr += original_len
            
            # 关键帧信息
            if len(video_chunk) < ptr + 8:
                return self._default_handle(video_chunk)
                
            keyframe_count, keyframe_len = struct.unpack('>II', video_chunk[ptr:ptr+8])
            ptr += 8
            
            # 检查数据长度
            if len(video_chunk) < ptr + keyframe_len:
                return self._default_handle(video_chunk)
                
            keyframe_bytes = video_chunk[ptr:ptr+keyframe_len]
            ptr += keyframe_len
            
            # 非关键帧信息
            if len(video_chunk) < ptr + 8:
                return self._default_handle(video_chunk)
                
            other_count, other_len = struct.unpack('>II', video_chunk[ptr:ptr+8])
            ptr += 8
            
            if len(video_chunk) < ptr + other_len:
                return self._default_handle(video_chunk)
                
            other_bytes = video_chunk[ptr:ptr+other_len]
            ptr += other_len
            
            # 解析索引信息
            keyframe_indices = []
            for _ in range(keyframe_count):
                if len(video_chunk) < ptr + 4:
                    return self._default_handle(video_chunk)
                idx, = struct.unpack('>I', video_chunk[ptr:ptr+4])
                keyframe_indices.append(idx)
                ptr += 4
                
            other_indices = []
            for _ in range(other_count):
                if len(video_chunk) < ptr + 4:
                    return self._default_handle(video_chunk)
                idx, = struct.unpack('>I', video_chunk[ptr:ptr+4])
                other_indices.append(idx)
                ptr += 4

            print(f"[ClientBiSR] Parsing: {keyframe_count} keyframes, {other_count} others")

            # 解码关键帧和非关键帧
            keyframes_tensor = torch.empty(0)
            other_tensor = torch.empty(0)
            
            # 修复关键帧处理
            if keyframe_len > 0:
                try:
                    keyframes_np, fps = decord_bytes2numpy(keyframe_bytes)
                    if keyframes_np.shape[0] > 0:
                        print(f"[ClientBiSR] Raw keyframes_np: shape={keyframes_np.shape}, dtype={keyframes_np.dtype}, range=[{keyframes_np.min()}, {keyframes_np.max()}]")
                        # decord_bytes2numpy返回的已经是NCHW格式，直接使用
                        keyframes_tensor = torch.from_numpy(keyframes_np)
                        print(f"[ClientBiSR] Decoded keyframes: {keyframes_tensor.shape}")
                except Exception as e:
                    print(f"Error decoding keyframes: {e}")

            # 修复非关键帧处理
            if other_len > 0:
                try:
                    other_np, fps = decord_bytes2numpy(other_bytes)
                    if other_np.shape[0] > 0:
                        # decord_bytes2numpy返回的已经是NCHW格式，直接使用
                        other_tensor = torch.from_numpy(other_np)
                        print(f"[ClientBiSR] Decoded other frames: {other_tensor.shape}")
                except Exception as e:
                    print(f"Error decoding other frames: {e}")

            # 对关键帧进行超分处理
            if len(keyframes_tensor) > 0:
                print("[ClientBiSR] Performing super-resolution on keyframes...")
                sr_keyframes = self._super_resolution_keyframes(keyframes_tensor)
            else:
                sr_keyframes = keyframes_tensor

            # 合并关键帧和非关键帧
            total_frames = keyframe_count + other_count
            if total_frames > 0:
                merged_tensor = self._merge_frames(sr_keyframes, other_tensor, 
                                                 keyframe_indices, other_indices, total_frames)
                # 转换回numpy格式 (NCHW → NHWC)
                # 应该检查输出数据：
                print(f"[ClientBiSR] Merged tensor range: [{merged_tensor.min()}, {merged_tensor.max()}]")
                result_np = merged_tensor.permute(0, 2, 3, 1).numpy().astype(np.uint8)
                print(f"[ClientBiSR] Final result range: [{result_np.min()}, {result_np.max()}]")
                return result_np
            else:
                print("[ClientBiSR] No frames to process, returning empty array")
                return np.empty((0, 720, 1280, 3), dtype=np.uint8)
            
        except Exception as e:
            print(f"Error in BiSR processing: {e}")
            import traceback
            traceback.print_exc()
            return self._default_handle(video_chunk)

    def _default_handle(self, video_chunk: bytes) -> np.ndarray:
        """默认处理方式，用于处理非BiSR格式的数据"""
        try:
            if hasattr(self, 'init_chunk') and self.init_chunk:
                video_bytes = self.init_chunk + video_chunk
            else:
                video_bytes = video_chunk
                
            video_np, fps = decord_bytes2numpy(video_bytes, threads=1)
            
            if video_np.shape[0] == 0:
                return np.zeros((1, 720, 1280, 3), dtype=np.uint8)
            
            # 转换格式 NCHW -> NHWC
            result = video_np.transpose(0, 2, 3, 1)
            return result
            
        except Exception as e:
            print(f"Error in default processing: {e}")
            # 返回一个空的视频帧
            return np.zeros((1, 720, 1280, 3), dtype=np.uint8)

    def init_arg(self) -> Dict[str, Any]:
        return {}

    def common_arg(self) -> Dict[str, Any]:
        return {
            'buffer': self.buffer
        }
