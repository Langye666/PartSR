import math
import multiprocessing

import requests
import struct
import time
import csv
import os
import numpy as np

from typing import Dict, Tuple, List, Optional, Any
from skimage.metrics import structural_similarity as ssim

from utils.utils import decord_video2numpy
from client_arch.client_PartSR import ClientPartSR
from config import config
client_arch = ClientPartSR

datasets = {
    'sell': ['plant']
}
dash_chunk = 12 # number of chunks
chunk_len = 2
buffer_lim = 5
ssim_calc_thread = 16

mu = 1 # RoI ssim weight
lambda1 = 1 # ssim difference coefficient
lambda2 = 1 # rebuffer coefficient

def gen_file_info(dataset: str, name: str):
    local_folder = os.path.join('/home/guest/server_folder/', os.path.join(dataset, name))
    return {
        'local_gt': os.path.join(local_folder, f'720p-{dataset}-{name}.mp4'),
        'local_lr': os.path.join(local_folder, f'180p-{dataset}-{name}.mp4'),
        'roi_annotation': os.path.join(local_folder, f'{name}_annotate/labels'),
        'server': f'{config['edge_server']}/{dataset}/{name}/',
        'video_save': f'client_result/ours-{dataset}-{name}.mp4',
    }

def calc_qoe(ssim_metric: List[List[Optional[int, float]]], rebuffer: List[float]):
    def weighted_ssim(line: List[Optional[int, float]]):
        return line[1] + mu * line[2]
    last_weighted = weighted_ssim(ssim_metric[0])
    ssim_avg = last_weighted
    ssim_dif_avg = 0
    rebuffer = sum(rebuffer)
    for i in range(1, len(ssim_metric)):
        weighted = weighted_ssim(ssim_metric[i])
        ssim_avg += weighted
        ssim_dif_avg += math.fabs(weighted - last_weighted)
        last_weighted = weighted
    return (ssim_avg - lambda1 * ssim_dif_avg - lambda2 * rebuffer) / len(ssim_metric)

def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    return ssim(img1, img2, multichannel=True, data_range=255, channel_axis=0)


def calc_ssim_worker(args: Tuple[int, np.ndarray, np.ndarray, List[Tuple]]) -> List[Tuple[int, float, float]]:
    start_idx, gt_chunk, processed_chunk, roi_chunk = args
    chunk_metrics = []
    for i in range(gt_chunk.shape[0]):
        idx = start_idx + i
        x, y, w, h = roi_chunk[i]
        H, W = gt_chunk.shape[2], gt_chunk.shape[3]

        # 计算 ROI 边界
        x1 = int(x * W)
        y1 = int(y * H)
        x2 = x1 + int(w * W)
        y2 = y1 + int(h * H)

        # 计算全局和 ROI 的 SSIM
        global_ssim = compute_ssim(gt_chunk[i], processed_chunk[i])
        roi_ssim = compute_ssim(gt_chunk[i, :, y1:y2, x1:x2], processed_chunk[i, :, y1:y2, x1:x2])
        chunk_metrics.append((idx, global_ssim, roi_ssim))
        print(f"frame{idx}: global={global_ssim:.4f}, roi={roi_ssim:.4f}", end=' ')
    return chunk_metrics


def calc_ssim(gt: np.ndarray, processed: np.ndarray, roi_xywh: List[Tuple]) -> List[List[Any]]:
    assert gt.shape == processed.shape and gt.shape[1] == 3  # NCHW, RGB
    assert gt.shape[0] == len(roi_xywh)

    n_frames = gt.shape[0]
    chunk_size = (n_frames + ssim_calc_thread - 1) // ssim_calc_thread  # 计算每个进程处理的帧数

    # 准备多进程参数
    chunks = []
    for i in range(0, n_frames, chunk_size):
        end_idx = min(i + chunk_size, n_frames)
        chunks.append((
            i,  # 起始索引
            gt[i:end_idx],
            processed[i:end_idx],
            roi_xywh[i:end_idx]  # 对应分块的 ROI 数据
        ))

    with multiprocessing.Pool(processes=ssim_calc_thread) as pool:
        results = pool.map(calc_ssim_worker, chunks)

    metric = []
    for chunk_result in results:
        for idx, global_ssim, roi_ssim in chunk_result:
            metric.append([idx, global_ssim, roi_ssim])

    global_avg = sum(m[1] for m in metric) / n_frames
    roi_avg = sum(m[2] for m in metric) / n_frames
    metric.append([-1, global_avg, roi_avg])

    return metric

# def calc_ssim(gt: np.ndarray, processed: np.ndarray, roi_xywh: List[Tuple[float, float, float, float]])\
#     -> List[List[Optional[int, float]]]:
#     assert gt.shape == processed.shape and gt.shape[1] == 3 # NCHW, RGB
#     assert gt.shape[0] == len(roi_xywh)
#     metric = []
#     global_avg = 0
#     roi_avg = 0
#     for idx in range(gt.shape[0]):
#         x, y, w, h = roi_xywh[idx]
#         x1 = int(x * gt.shape[3])
#         y1 = int(y * gt.shape[2])
#         x2 = x1 + int(w * gt.shape[3])
#         y2 = y1 + int(h * gt.shape[2])
#         global_ssim = compute_ssim(gt[idx], processed[idx])
#         roi_ssim = compute_ssim(gt[idx, :, y1:y2, x1:x2], processed[idx, :, y1:y2, x1:x2])
#         metric.append([idx, global_ssim, roi_ssim])
#         global_avg += global_ssim
#         roi_avg += roi_ssim
#         print(f"frame{idx}: global={global_ssim}, roi={roi_ssim}", end=' ')
#     metric.append([-1, global_avg / gt.shape[0], roi_avg / gt.shape[0]])
#     return metric

def read_yolo_roi(folder: str):
    roi_xywh = []
    for file in sorted(os.listdir(folder)):
        if not file.endswith('.txt'):
            continue
        with open(os.path.join(folder, file), 'r') as f:
            parts = f.read().strip().split(' ')
        roi_xywh.append((float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
    return roi_xywh

def run_test(dataset: str, name: str) -> Tuple[Dict, List]:
    client = client_arch(chunk_len)
    file_info = gen_file_info(dataset, name)
    perform_metric = {
        'arch_delay': [],
        'trans_size': [],
        'rebuffer': [],
    }

    response = requests.get(file_info['server'] + f'init-stream0.m4s', {**client.init_arg(), **client.common_arg()})
    response.raise_for_status()
    init_chunk = response.content
    client.receive(init_chunk, is_init=True)

    for chunk_idx in range(1, dash_chunk + 1):
        response = requests.get(file_info['server'] + f'chunk-stream0-{chunk_idx: 05d}.m4s', params=client.common_arg())
        response.raise_for_status()
        received = response.content

        # Use the latency recorded on the server to avoid actual network factors. Use a bandwidth dataset later to calc the delay.
        server_delay, = struct.unpack(">f", received[:4])
        timer = time.perf_counter()
        client.receive(received[4:], is_init=False)
        client_delay = time.perf_counter() - timer
        arch_delay = server_delay + client_delay

        client.buffer -= server_delay + client_delay
        if client.buffer < 0:
            rebuffer = -client.buffer
            client.buffer = 2
        else:
            rebuffer = 0
            client.buffer += 2
        perform_metric['arch_delay'].append(arch_delay)
        perform_metric['trans_size'].append(len(received) - 4)
        perform_metric['rebuffer'].append(rebuffer)
        print(f"\nidx={chunk_idx}, server: {server_delay}s, client: {client_delay}s, rebuffer: {rebuffer}s")
        if client.buffer > buffer_lim: # buffer limit simulation, to avoid computational overhead in multi-stream scenes
            time.sleep(client.buffer - buffer_lim)
            client.buffer = buffer_lim

    gt, fps = decord_video2numpy(file_info['local_gt'])
    processed = client.cached
    roi_xywh = read_yolo_roi(file_info['roi_annotation'])
    quality_metric = calc_ssim(gt, processed, roi_xywh)

    print(f"\n overall quality: global={quality_metric[-1][1]}, roi={quality_metric[-1][2]}")
    client.save_video_rebuffer(perform_metric['rebuffer'], file_info['video_save'])
    return perform_metric, quality_metric

def run_dataset(dataset: str):
    print(f"Running dataset {dataset}...")
    dataset_dir = os.path.join("client_result", dataset)
    for test in datasets[dataset]:
        perform_metric, quality_metric = run_test(dataset, test)
        if not os.path.exists(dataset_dir):
            os.mkdir(dataset_dir)
        with open(os.path.join(dataset_dir, f"{test}.csv"), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['idx', 'global_ssim', 'roi_ssim', 'QoE'])
            for data in quality_metric:
                row = [*data]
                writer.writerow(row)

def main():
    for k in datasets.keys():
        run_dataset(k)

if __name__ == '__main__':
    main()