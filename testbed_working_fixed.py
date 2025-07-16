import os
# 在所有导入之前 设置环境变量
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

import math
import multiprocessing
import requests
import struct
import time
import csv
import numpy as np

from typing import Dict, Tuple, List, Optional, Any, Union
from skimage.metrics import structural_similarity as ssim

# 先导入config和torch
from config import config
import torch

# 然后导入其他模块
from utils.utils import decord_video2numpy
from client_arch.client_BiSR import ClientBiSR

print("All imports successful!")

client_arch = ClientBiSR

datasets = {
    'sell_4s_720p': ['plant']
}

# 从testbed.py复制的参数
test_len = 24
dash_chunk = 6  # 保持原来的值
chunk_len = 4
buffer_lim = 5
ssim_calc_thread = 16

mu = 1  # RoI ssim weight
lambda1 = 1  # ssim difference coefficient
lambda2 = 1  # rebuffer coefficient

# 网络trace数据
network_stat = [11256, 13946, 12408, 13777, 11297, 7744, 13324, 4796, 7356, 11768, 13922, 12629, 11842, 16473, 14448, 18178, 16397, 15147, 11007, 9258, 9854, 6110, 11633, 12596, 6258]
net_time = 0

def get_time(size):
    """
    使用网络trace计算传输时延
    Args:
        size: 传输数据大小（字节）
    Returns:
        传输时延（秒）
    """
    global net_time
    ans = 0
    while size > 0:
        speed = network_stat[net_time] * 128  # 转换为字节/秒
        net_time += 1
        if speed >= size:
            return ans + size / speed
        ans += 1
        size -= speed
        if net_time >= len(network_stat):
            net_time = 0
    return ans

def gen_file_info(dataset: str, name: str):
    local_folder = os.path.join('/home/zlp/wyc/vid_source_server/files/', os.path.join(dataset, name))
    return {
        'local_gt': os.path.join(local_folder, f'720p-{dataset}-{name}.mp4'),
        'local_lr': os.path.join(local_folder, f'180p-{dataset}-{name}.mp4'),
        'roi_annotation': os.path.join(local_folder, f'{name}_annotate/labels'),
        'server': f'{config["edge_server"]}/{dataset}/{name}/',
        'video_save': f'client_result/ours-{dataset}-{name}.mp4',
    }

def calc_qoe(ssim_metric: List[List[Union[int, float]]], rebuffer: List[float]):
    def weighted_ssim(line: List[Union[int, float]]):
        return line[1] + mu * line[2]
    last_weighted = weighted_ssim(ssim_metric[0])
    ssim_avg = last_weighted
    ssim_dif_avg = 0
    rebuffer_sum = sum(rebuffer)
    for i in range(1, len(ssim_metric)):
        weighted = weighted_ssim(ssim_metric[i])
        ssim_avg += weighted
        ssim_dif_avg += math.fabs(weighted - last_weighted)
        last_weighted = weighted
    return (ssim_avg - lambda1 * ssim_dif_avg - lambda2 * rebuffer_sum) / len(ssim_metric)

def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    return ssim(img1, img2, channel_axis=2, data_range=255)

def calc_ssim_worker(args: Tuple[int, np.ndarray, np.ndarray, List[Tuple]]) -> List[Tuple[int, float, float]]:
    start_idx, gt_chunk, processed_chunk, roi_chunk = args
    chunk_metrics = []
    for i in range(gt_chunk.shape[0]):
        idx = start_idx + i
        if i < len(roi_chunk):
            x, y, w, h = roi_chunk[i]
        else:
            x, y, w, h = 0.5, 0.5, 0.3, 0.3
            
        H, W = gt_chunk.shape[1], gt_chunk.shape[2]  # NHWC format

        # 计算 ROI 边界
        x1 = max(0, int((x - w/2) * W))
        y1 = max(0, int((y - h/2) * H))
        x2 = min(W, int((x + w/2) * W))
        y2 = min(H, int((y + h/2) * H))

        # 确保ROI区域有效
        if x2 <= x1 or y2 <= y1:
            x1, y1, x2, y2 = 0, 0, W, H

        # 计算全局和 ROI 的 SSIM
        global_ssim = compute_ssim(gt_chunk[i], processed_chunk[i])
        roi_ssim = compute_ssim(gt_chunk[i, y1:y2, x1:x2], processed_chunk[i, y1:y2, x1:x2])
        chunk_metrics.append((idx, global_ssim, roi_ssim))
        if idx % 50 == 0:  # 每50帧打印一次，避免输出过多
            print(f"frame{idx}: global={global_ssim:.4f}, roi={roi_ssim:.4f}")
    return chunk_metrics

def calc_ssim(gt: np.ndarray, processed: np.ndarray, roi_xywh: List[Tuple]) -> List[List[Any]]:
    print(f"Calculating SSIM for {gt.shape[0]} frames...")
    print(f"GT shape: {gt.shape}, Processed shape: {processed.shape}")
    
    # 确保两个数组的形状一致
    min_frames = min(gt.shape[0], processed.shape[0])
    gt = gt[:min_frames]
    processed = processed[:min_frames]
    
    # 确保ROI数据长度匹配
    roi_xywh = roi_xywh[:min_frames]
    while len(roi_xywh) < min_frames:
        roi_xywh.append((0.5, 0.5, 0.3, 0.3))
    
    n_frames = gt.shape[0]
    chunk_size = max(1, (n_frames + ssim_calc_thread - 1) // ssim_calc_thread)

    # 准备多进程参数
    chunks = []
    for i in range(0, n_frames, chunk_size):
        end_idx = min(i + chunk_size, n_frames)
        chunks.append((
            i,
            gt[i:end_idx],
            processed[i:end_idx],
            roi_xywh[i:end_idx]
        ))

    try:
        with multiprocessing.Pool(processes=min(ssim_calc_thread, len(chunks))) as pool:
            results = pool.map(calc_ssim_worker, chunks)
    except Exception as e:
        print(f"Error in multiprocessing: {e}")
        # 单线程后备方案
        results = [calc_ssim_worker(chunk) for chunk in chunks]

    metric = []
    for chunk_result in results:
        for idx, global_ssim, roi_ssim in chunk_result:
            metric.append([idx, global_ssim, roi_ssim])

    if metric:
        global_avg = sum(m[1] for m in metric) / n_frames
        roi_avg = sum(m[2] for m in metric) / n_frames
        metric.append([-1, global_avg, roi_avg])
    else:
        metric.append([-1, 0.0, 0.0])

    return metric

def read_yolo_roi(folder: str):
    roi_xywh = []
    if not os.path.exists(folder):
        print(f"ROI folder not found: {folder}, using default ROI")
        return [(0.5, 0.5, 0.3, 0.3)] * 720
    
    try:
        for file in sorted(os.listdir(folder)):
            if not file.endswith('.txt'):
                continue
            with open(os.path.join(folder, file), 'r') as f:
                parts = f.read().strip().split(' ')
            if len(parts) >= 5:
                roi_xywh.append((float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
    except Exception as e:
        print(f"Error reading ROI files: {e}")
    
    if not roi_xywh:
        print("No valid ROI found, using default")
        return [(0.5, 0.5, 0.3, 0.3)] * 720
    
    return roi_xywh

def save_keyframes_for_analysis(client, file_info):
    """保存关键帧用于分析"""
    try:
        # 获取关键帧数据
        keyframes_data = []
        for i, chunk in enumerate(client.cached):
            if i > 0:  # 跳过第一个chunk（没有关键帧）
                # 每个chunk的第一帧是关键帧
                keyframe = chunk[0]  # 第一帧
                keyframes_data.append(keyframe)
        
        if keyframes_data:
            keyframes_array = np.array(keyframes_data)
            print(f"Saving {len(keyframes_data)} keyframes...")
            
            # 保存关键帧视频
            keyframes_path = file_info['video_save'].replace('.mp4', '_keyframes.mp4')
            
            import cv2
            height, width = keyframes_array.shape[1], keyframes_array.shape[2]
            fps = 30.0 / 120  # 关键帧帧率 (每120帧一个关键帧)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(keyframes_path, fourcc, fps, (width, height))
            
            for i, frame in enumerate(keyframes_array):
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
                print(f"Saved keyframe {i+1}/{len(keyframes_array)}")
            
            out.release()
            print(f"✓ Keyframes saved to {keyframes_path}")
            
            # 保存关键帧图片用于对比
            keyframes_dir = keyframes_path.replace('.mp4', '_frames')
            os.makedirs(keyframes_dir, exist_ok=True)
            
            for i, frame in enumerate(keyframes_array):
                frame_path = os.path.join(keyframes_dir, f"keyframe_{i+1:03d}.png")
                cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
            print(f"✓ Keyframe images saved to {keyframes_dir}")
            
    except Exception as e:
        print(f"Error saving keyframes: {e}")

def run_test(dataset: str, name: str) -> Tuple[Dict, List]:
    print(f"Running test: {dataset}/{name}")
    
    client = client_arch(chunk_len)
    file_info = gen_file_info(dataset, name)
    
    print(f"File info: {file_info}")
    
    perform_metric = {
        'arch_delay': [],
        'trans_size': [],
        'trans_delay': [],  # 添加传输时延记录
        'rebuffer': [],
    }

    try:
        # 初始化
        print("Requesting initialization...")
        response = requests.get(file_info['server'] + f'init-stream0.m4s', 
                              params={**client.init_arg(), **client.common_arg()})
        response.raise_for_status()
        init_chunk = response.content
        client.receive(init_chunk, is_init=True)
        print("Initialization successful")

        # 处理视频块
        for chunk_idx in range(1, dash_chunk + 1):
            print(f"\nProcessing chunk {chunk_idx}...")
            response = requests.get(file_info['server'] + f'chunk-stream0-{chunk_idx:05d}.m4s', 
                                  params=client.common_arg())
            response.raise_for_status()
            received = response.content

            # 解析服务器延迟
            server_delay, = struct.unpack(">f", received[:4])
            timer = time.perf_counter()
            client.receive(received[4:], is_init=False)
            client_delay = time.perf_counter() - timer
            arch_delay = server_delay + client_delay

            # 计算传输相关指标
            trans_size = len(received) - 4
            trans_delay = get_time(trans_size)  # 使用网络trace计算传输时延
            
            # 计算总延迟
            tot_delay = server_delay + client_delay + trans_delay

            # 更新缓冲区 - 使用总延迟，参考testbed.py的逻辑
            client.buffer -= tot_delay
            if client.buffer < 0:
                rebuffer = -client.buffer
                client.buffer = chunk_len  # 参考testbed.py，使用chunk_len而不是2
            else:
                rebuffer = 0
                client.buffer += chunk_len  # 参考testbed.py，使用chunk_len而不是2
                
            perform_metric['arch_delay'].append(arch_delay)
            perform_metric['trans_size'].append(trans_size)
            perform_metric['trans_delay'].append(trans_delay)  # 记录传输时延
            perform_metric['rebuffer'].append(rebuffer)
            
            # 更新打印信息，包含传输时延
            print(f"idx={chunk_idx}, server: {server_delay:.3f}s, client: {client_delay:.3f}s, trans: {trans_delay:.3f}s, tot: {tot_delay:.3f}s, rebuffer: {rebuffer:.3f}s")
            
            if client.buffer > buffer_lim:
                time.sleep(client.buffer - buffer_lim)
                client.buffer = buffer_lim

        # 质量评估部分保持不变...
        print("\nCalculating quality metrics...")
        processed_frames = client.cached

        # 转换列表为numpy数组
        if processed_frames and isinstance(processed_frames, list):
            print(f"Converting {len(processed_frames)} processed frames to numpy array...")
            processed = np.concatenate(processed_frames, axis=0)
            print(f"Processed array shape: {processed.shape}")
        else:
            print("No processed frames available")
            return perform_metric, [[-1, 0.0, 0.0, 0.0]]

        if os.path.exists(file_info['local_gt']):
            print("Loading GT file...")
            gt, fps = decord_video2numpy(file_info['local_gt'])
            roi_xywh = read_yolo_roi(file_info['roi_annotation'])
            
            # 转换GT格式以匹配SSIM计算
            if gt.ndim == 4 and gt.shape[1] == 3:  # NCHW -> NHWC
                gt = gt.transpose(0, 2, 3, 1)
            
            print(f"Final GT shape: {gt.shape}")
            print(f"Final processed shape: {processed.shape}")
            
            # 确保两个数组的格式匹配
            if len(gt.shape) >= 3 and len(processed.shape) >= 3:
                if gt.shape[1:3] != processed.shape[1:3]:
                    print(f"Shape mismatch: GT{gt.shape[1:3]} vs Processed{processed.shape[1:3]}")
                    print("Resizing processed frames to match GT...")
                    import cv2
                    resized_processed = []
                    target_h, target_w = gt.shape[1], gt.shape[2]
                    for frame in processed:
                        if frame.shape[:2] != (target_h, target_w):
                            resized_frame = cv2.resize(frame, (target_w, target_h))
                            resized_processed.append(resized_frame)
                        else:
                            resized_processed.append(frame)
                    processed = np.array(resized_processed)
                    print(f"Resized processed shape: {processed.shape}")
            
            quality_metric = calc_ssim(gt, processed, roi_xywh)
            print(f"\nOverall quality: global={quality_metric[-1][1]:.4f}, roi={quality_metric[-1][2]:.4f}")
        else:
            print(f"GT file not found: {file_info['local_gt']}")
            quality_metric = [[-1, 0.0, 0.0]]

        # 计算 QoE
        print("Calculating QoE...")
        if len(quality_metric) > 1:
            qoe_value = calc_qoe(quality_metric[:-1], perform_metric['rebuffer'])
            print(f"QoE: {qoe_value:.4f}")
            
            # 为每个帧数据添加QoE值
            for i, metric_row in enumerate(quality_metric):
                if i < len(quality_metric) - 1:
                    metric_row.append(qoe_value)
                else:
                    metric_row.append(qoe_value)
        else:
            print("Insufficient data for QoE calculation")
            for metric_row in quality_metric:
                metric_row.append(0.0)

        # 暂时禁用关键帧保存功能
        # print("\nSaving keyframes for analysis...")
        # save_keyframes_for_analysis(client, file_info)

        # 保存视频部分保持不变...
        print("\nSaving processed video...")
        os.makedirs(os.path.dirname(file_info['video_save']), exist_ok=True)
        
        # 使用FFmpeg保存视频
        def save_video_ffmpeg(frames, save_path):
            import subprocess
            
            if len(frames) == 0:
                print("No frames to save")
                return
                
            print(f"Saving {len(frames)} frames to {save_path}")
            
            height, width = frames.shape[1], frames.shape[2]
            fps = 30.0
            
            cmd = [
                'ffmpeg', '-y', '-loglevel', 'error',
                '-f', 'rawvideo',
                '-pix_fmt', 'rgb24',
                '-s', f'{width}x{height}',
                '-r', str(fps),
                '-i', '-',
                '-vcodec', 'libx264',
                '-pix_fmt', 'yuv420p',
                save_path
            ]
            
            try:
                process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
                
                for i, frame in enumerate(frames):
                    try:
                        frame_bytes = frame.astype(np.uint8).tobytes()
                        process.stdin.write(frame_bytes)
                        
                        if i % 100 == 0:
                            print(f"Saved {i}/{len(frames)} frames...")
                    except BrokenPipeError:
                        break
                
                process.stdin.close()
                _, stderr = process.communicate()
                
                if process.returncode == 0:
                    print(f"✓ Video saved to {save_path}")
                else:
                    print(f"✗ FFmpeg error: {stderr.decode()}")
                    
            except Exception as e:
                print(f"Error saving video: {e}")
                
                print("Trying OpenCV as fallback...")
                try:
                    import cv2
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
                    
                    for i, frame in enumerate(frames):
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        out.write(frame_bgr)
                        
                        if i % 100 == 0:
                            print(f"OpenCV saved {i}/{len(frames)} frames...")
                    
                    out.release()
                    print(f"✓ Video saved with OpenCV to {save_path}")
                    
                except Exception as e2:
                    print(f"OpenCV also failed: {e2}")
            
        # 保存处理后的视频
        if processed_frames and isinstance(processed_frames, list):
            save_video_ffmpeg(processed, file_info['video_save'])
        
        return perform_metric, quality_metric
        
    except Exception as e:
        print(f"Error in run_test: {e}")
        import traceback
        traceback.print_exc()
        return perform_metric, [[-1, 0.0, 0.0, 0.0]]

def run_dataset(dataset: str):
    print(f"Running dataset {dataset}...")
    dataset_dir = os.path.join("client_result", dataset)
    
    for test in datasets[dataset]:
        perform_metric, quality_metric = run_test(dataset, test)
        
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
            
        # 保存结果
        with open(os.path.join(dataset_dir, f"{test}.csv"), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['idx', 'global_ssim', 'roi_ssim', 'QoE'])
            for data in quality_metric:
                row = [*data]
                writer.writerow(row)
        
        print(f"Results saved to {dataset_dir}/{test}.csv")
        
        # 打印详细的性能统计
        print(f"\nPerformance Summary for {test}:")
        print(f"  Average server delay: {np.mean(perform_metric['arch_delay']):.3f}s")
        print(f"  Average transmission size: {np.mean(perform_metric['trans_size']):.0f} bytes")
        print(f"  Average transmission delay: {np.mean(perform_metric['trans_delay']):.3f}s")
        print(f"  Total rebuffer time: {sum(perform_metric['rebuffer']):.3f}s")
        if len(quality_metric) > 0:
            print(f"  Final QoE: {quality_metric[-1][3]:.4f}")
        
        # 保存性能指标到单独的文件
        perf_file = os.path.join(dataset_dir, f"{test}_performance.csv")
        with open(perf_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['chunk_idx', 'arch_delay', 'trans_size', 'trans_delay', 'rebuffer'])
            for i in range(len(perform_metric['arch_delay'])):
                writer.writerow([
                    i+1,
                    perform_metric['arch_delay'][i],
                    perform_metric['trans_size'][i],
                    perform_metric['trans_delay'][i],
                    perform_metric['rebuffer'][i]
                ])
        print(f"Performance metrics saved to {perf_file}")

def main():
    print("Starting PartSR BiSR evaluation...")
    
    for k in datasets.keys():
        run_dataset(k)
        
    print("Evaluation completed!")

if __name__ == '__main__':
    main()
