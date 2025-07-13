import argparse
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
    'sell': ['beef', 'garlic', 'plant']
}

test_len = 24
chunk_len = 4
dash_chunk = test_len // chunk_len # number of chunks
buffer_lim = 5
ssim_calc_thread = 8

mu = 1 # RoI ssim weight
lambda1 = 1 # ssim difference coefficient
lambda2 = 1 # rebuffer coefficient

network_stat = [11256, 13946, 12408, 13777, 11297, 7744, 13324, 4796, 7356, 11768, 13922, 12629, 11842, 16473, 14448, 18178, 16397, 15147, 11007, 9258, 9854, 6110, 11633, 12596, 6258]
net_time = 0

video_save = False
video_annotate = False

def get_time(size):
    global net_time
    ans = 0
    while size > 0:
        speed = network_stat[net_time]*128
        net_time += 1
        if speed >= size:
            return ans + size / speed
        ans += 1
        size -= speed
        if net_time >= len(network_stat):
            net_time = 0

def gen_file_info(dataset: str, name: str):
    local_folder = os.path.join('/home/guest/server_folder/', os.path.join(f"{dataset}_{chunk_len}s", name))
    return {
        'local_gt': os.path.join(local_folder, f'720p-{dataset}-{name}.mp4'),
        'local_lr': os.path.join(local_folder, f'180p-{dataset}-{name}.mp4'),
        'roi_annotation': os.path.join(local_folder, f'{name}_annotate/labels'),
        'server': f'{config['edge_server']}/{dataset}_{chunk_len}s/{name}/',
        'video_save': f'client_result/ours-{dataset}-{name}.mp4',
    }

def calc_qoe(ssim_metric: List[List[int | float]], rebuffer: List[float]):
    def weighted_ssim(line: List[int | float]):
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
        x1 = int((x - w / 2) * W)
        y1 = int((y - h / 2) * H)
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
#     -> List[List[int | float]]:
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

def run_test(dataset: str, name: str) -> Tuple[Dict, List, float]:
    client = client_arch(chunk_len)
    file_info = gen_file_info(dataset, name)
    perform_metric = {
        'arch_delay': [],
        'trans_size': [],
        'trans_delay': [],
        'rebuffer': [],
    }

    init_param = {**client.init_arg(), **client.common_arg()}
    response = requests.get(file_info['server'] + f'init-stream0.m4s', init_param)
    response.raise_for_status()
    init_chunk = response.content
    client.receive(init_chunk, is_init=True)

    for chunk_idx in range(1, dash_chunk + 1):
        response = requests.get(file_info['server'] + f'chunk-stream0-{chunk_idx:05d}.m4s', params=client.common_arg())
        response.raise_for_status()
        received = response.content

        # Use the latency recorded on the server to avoid actual network factors. Use a bandwidth dataset later to calc the delay.
        server_delay, = struct.unpack(">f", received[:4])
        timer = time.perf_counter()
        client.receive(received[4:], is_init=False)
        client_delay = time.perf_counter() - timer
        arch_delay = server_delay + client_delay

        trans_size = len(received) - 4
        trans_delay = get_time(trans_size)

        tot_delay = server_delay + client_delay + trans_delay
        client.buffer -= tot_delay
        if client.buffer < 0:
            rebuffer = -client.buffer
            client.buffer = chunk_len
        else:
            rebuffer = 0
            client.buffer += chunk_len
        perform_metric['arch_delay'].append(arch_delay)
        perform_metric['trans_size'].append(trans_size)
        perform_metric['trans_delay'].append(trans_delay)
        perform_metric['rebuffer'].append(rebuffer)
        print(f"\nidx={chunk_idx}, server: {server_delay}s, trans: {trans_delay}s, client: {client_delay}s, tot: {tot_delay}; rebuffer={rebuffer}s")
        if client.buffer > buffer_lim: # buffer limit simulation, to avoid computational overhead in multi-stream scenes
            time.sleep(client.buffer - buffer_lim)
            client.buffer = buffer_lim

    gt, fps = decord_video2numpy(file_info['local_gt'])
    processed = np.concatenate(client.cached)
    roi_xywh = read_yolo_roi(file_info['roi_annotation'])
    quality_metric = calc_ssim(gt, processed, roi_xywh)
    qoe = calc_qoe(quality_metric, perform_metric['rebuffer'])

    print(f"\n overall quality: global={quality_metric[-1][1]}, roi={quality_metric[-1][2]}, QoE={qoe}")
    if video_save:
        client.save_video_rebuffer(perform_metric['rebuffer'], file_info['video_save'], video_annotate, roi_xywh)
    return perform_metric, quality_metric, qoe

def run_dataset(dataset: str):
    print(f"Running dataset {dataset}...")
    dataset_dir = os.path.join("client_result", dataset)
    dataset_global_ssim = 0
    dataset_roi_ssim = 0
    for test in datasets[dataset]:
        perform_metric, quality_metric, qoe = run_test(dataset, test)
        dataset_global_ssim += quality_metric[-1][1]
        dataset_roi_ssim += quality_metric[-1][2]
        if not os.path.exists(dataset_dir):
            os.mkdir(dataset_dir)
        with open(os.path.join(dataset_dir, f"{test}.csv"), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['idx', 'global_ssim', 'roi_ssim'])
            for data in quality_metric:
                row = [*data]
                writer.writerow(row)
            writer.writerow(["Rebuffer", *perform_metric['rebuffer']])
            writer.writerow(["Overall QoE", qoe])
    print("Dataset global ssim =", dataset_global_ssim / len(datasets[dataset]))
    print("Dataset roi ssim =", dataset_roi_ssim / len(datasets[dataset]))

def main():
    for k in datasets.keys():
        run_dataset(k)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--save",
                        action="store_true",
                        help="save the processed video (with rebuffer) to a file")
    parser.add_argument("--annotate",
                        action="store_true",
                        help="annotate ground truth RoI and SR patch in video")
    args = parser.parse_args()
    video_save = args.save
    video_annotate = args.annotate
    main()