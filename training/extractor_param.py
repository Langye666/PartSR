import math, json, statistics
import os
import subprocess
import cv2
from mvextractor.videocap import VideoCap
from typing import List, Tuple

import numpy as np
import tqdm

from extractor.extractor import RoIExtractor
from extractor.tracker import init_tracker, track

ffmpeg = "ffmpeg" # "/home/yosame/Desktop/Grad_Proj/FFmpeg-Residual/ffmpeg"
ffmpeg_residual = "/home/yosame/Desktop/Grad_Proj/FFmpeg-Residual/ffmpeg_residual_3"

def load_annotations(json_path: str) -> List[Tuple[int, int, int, int]]:
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 获取标注字典 {帧编号: [x, y, w, h]}
    annotations_dict = data["annotations"]
    # 创建连续帧列表
    bbox_list = []
    # 当前帧号从0开始
    frame_idx = 0

    # 按顺序处理连续帧
    while str(frame_idx) in annotations_dict:
        # 获取当前帧的标注
        x, y, w, h = annotations_dict[str(frame_idx)]
        bbox_list.append((x, y, w, h))
        # 检查下一帧
        frame_idx += 1

    return bbox_list

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

class DynCalculator:
    def __init__(self, framerate: float, video_h: int, video_w: int, beta_arr: np.ndarray):
        self.framerate = framerate
        self.video_h = video_h
        self.video_w = video_w
        self.video_size = video_h * video_w
        self.diag_length = math.sqrt(video_h ** 2 + video_w ** 2)
        self.beta_arr = beta_arr

    def get_inside(self, block_interval: Tuple[int, int], roi_interval: Tuple[int, int]) -> int:
        left_or_top = max(block_interval[0], roi_interval[0])
        right_or_bottom = min(block_interval[1], roi_interval[1])
        return 0 if left_or_top >= right_or_bottom else right_or_bottom - left_or_top

    def calc_dyn_exclusive(self, mv: list, residual_arr: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
        x, y, w, h = roi
        x1, y1 = x, y
        x2, y2 = x + w, y + h
        motion = 0
        for item in mv:
            source, w, h, src_x, src_y, dst_x, dst_y, motion_x, motion_y, motion_scale = item
            x_inside = self.get_inside((dst_x, dst_x + w), (x1, x2))
            y_inside = self.get_inside((dst_y, dst_y + h), (y1, y2))
            area = w * h - x_inside * y_inside
            if area > 0:
                module = math.sqrt((motion_x / motion_scale) ** 2 + (motion_y / motion_scale) ** 2)
                motion += module * area
        motion = motion * self.framerate / self.video_size / self.diag_length
        mb_id = 0
        residual = 0
        for mb_y in range(0, self.video_h, 16):
            for mb_x in range(0, self.video_w, 16):
                residual += residual_arr[mb_id] * (1 - calc_IoU((mb_x, mb_y, mb_x + 16, mb_y + 16), (x1, y1, x2, y2)))
                mb_id += 1
        return motion + self.beta_arr * residual

    def calc_dyn_inclusive(self, mv: list, residual_arr: np.ndarray) -> np.ndarray:
        motion = 0
        for item in mv:
            source, w, h, src_x, src_y, dst_x, dst_y, motion_x, motion_y, motion_scale = item
            area = w * h
            module = math.sqrt((motion_x / motion_scale) ** 2 + (motion_y / motion_scale) ** 2)
            motion += module * area
        motion = motion * self.framerate / self.video_size / self.diag_length
        return motion + self.beta_arr * residual_arr.sum()

def test_dynamicity(beta_arr: np.ndarray, theta: float, video_data_list: List) -> Tuple[float, float]:
    inclusive_records = [[] for _ in range(len(beta_arr))]  # 每个元素是一个列表，存储该beta参数的所有inclusive_dyn
    exclusive_records = [[] for _ in range(len(beta_arr))]  # 每个元素是一个列表，存储该beta参数的所有exclusive_dyn

    for idx, video_data in tqdm.tqdm(enumerate(video_data_list)):
        ndarray, framerate, mvs, types, residual_list, bbox_list, dataset = video_data
        idx = 0
        dyn_calc = DynCalculator(framerate, ndarray.shape[2], ndarray.shape[1], beta_arr)
        no_anchor = True

        while idx < len(bbox_list):
            tracker = init_tracker(ndarray[idx], bbox_list[idx])
            inclusive_dyn = np.zeros_like(beta_arr)
            exclusive_dyn = np.zeros_like(beta_arr)
            idx += 1

            while idx < len(bbox_list):
                success, x, y, w, h = track(tracker, ndarray[idx])
                if not success:
                    print(f"Track failed: {dataset}")
                    break # restart this frame

                frame_inclusive_dyn = dyn_calc.calc_dyn_inclusive(mvs[idx], residual_list[idx])
                frame_exclusive_dyn = dyn_calc.calc_dyn_exclusive(mvs[idx], residual_list[idx], (x, y, w, h))
                inclusive_dyn += frame_inclusive_dyn
                exclusive_dyn += frame_exclusive_dyn

                gt_x, gt_y, gt_w, gt_h = bbox_list[idx]
                IoU = calc_IoU((x, y, x + w, y + h), (gt_x, gt_y, gt_x + gt_w, gt_y + gt_h))
                idx += 1

                if IoU < theta:
                    no_anchor = False
                    for i in range(len(beta_arr)):
                        inclusive_records[i].append(inclusive_dyn[i])
                        exclusive_records[i].append(exclusive_dyn[i])
                    break
        if no_anchor:
            print(f"No anchor in video {idx}, dynamicity = {(inclusive_dyn, exclusive_dyn)}")

    # 计算每个beta参数的统计量（带离群值过滤和变异系数）
    inclusive_means = np.zeros(len(beta_arr))
    inclusive_outliers = np.zeros(len(beta_arr))  # 变异系数数组
    exclusive_means = np.zeros(len(beta_arr))
    exclusive_outliers = np.zeros(len(beta_arr))  # 变异系数数组

    for i in range(len(beta_arr)):
        # 处理inclusive数据
        if inclusive_records[i]:
            # 使用IQR方法过滤离群值
            data = np.array(inclusive_records[i])
            q1 = np.percentile(data, 25)
            q3 = np.percentile(data, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            # 过滤离群值
            filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]

            inclusive_means[i] = np.mean(filtered_data)
            inclusive_outliers[i] = len(data) - len(filtered_data)

        # 处理exclusive数据（同上）
        if exclusive_records[i]:
            data = np.array(exclusive_records[i])
            q1 = np.percentile(data, 25)
            q3 = np.percentile(data, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]

            exclusive_means[i] = np.mean(filtered_data)
            exclusive_outliers[i] = len(data) - len(filtered_data)


        print(f"beta: {beta_arr[i]}, inclusive = (mean: {inclusive_means[i]:.4f}, num_outliers: {inclusive_outliers[i]}), "
              f"exclusive = (mean: {exclusive_means[i]:.4f}, num_outliers: {exclusive_outliers[i]})")
    print(f"Total num: {len(inclusive_records[0])}")
    # return inclusive_means, inclusive_cvs, exclusive_means, exclusive_cvs


def generate_parameters(datasets: List, theta: float):
    video_data_list = []
    for dataset in datasets:
        bbox_list = load_annotations(dataset[1])
        valid_num = len(bbox_list)
        if valid_num < 60:
            continue
        with open(dataset[0], 'rb') as f:
            video_bytes = f.read()
        ndarray, framerate, mvs, types, residual_list = decode_mv_residual(video_bytes, "extractor_param")
        video_data_list.append((ndarray[:valid_num], framerate, mvs[:valid_num], types[:valid_num], residual_list[:valid_num], bbox_list, dataset))
    test_dynamicity(np.array([x for x in range(0, 100)]) / 10, theta, video_data_list)
    # if variance < min_variance:
    #     overall_mean = mean
    # print(f"beta: {overall_mean}, variance: {min_variance}")

pattern = 'sell-'
def dyn_dataset(vid_list: List[str], json_list: List[str]):
    import os
    assert len(vid_list) == len(json_list)
    path_pair = []
    for idx in range(len(vid_list)):
        print(os.listdir(vid_list[idx]))
        for file in os.listdir(vid_list[idx]):
            if not (file.startswith("180p-") and file.endswith(".mp4")):
                continue
            main_name = file[5:-4]
            pattern_pos = main_name.find(pattern)
            if pattern_pos != -1:
                main_name = main_name[:pattern_pos] + main_name[pattern_pos + len(pattern):]
            for json_file in os.listdir(json_list[idx]):
                if not json_file.endswith(".json"):
                    continue
                pattern_pos = json_file.find(pattern)
                json_name = json_file[:-5]
                if pattern_pos != -1:
                    json_name = json_name[:pattern_pos] + json_name[pattern_pos + len(pattern):]
                if not main_name in json_name:
                    continue
                new_pair = (os.path.join(vid_list[idx], file), os.path.join(json_list[idx], json_file))
                path_pair.append(new_pair)
                print(new_pair)
    return path_pair

if __name__ == "__main__":
    datasets = dyn_dataset([r'/home/yosame/Desktop/Grad_Proj/#semiauto-roi-labeler/RoI_Datasets/A', '/home/yosame/Desktop/Grad_Proj/#semiauto-roi-labeler/RoI_Datasets/B', r'../#semiauto-roi-labeler/180P-annotated/6.27'],
                           [r'../#semiauto-roi-labeler/180P-annotated/5.9', '../#semiauto-roi-labeler/180P-annotated/5.9', r'../#semiauto-roi-labeler/180P-annotated/6.27'])
    theta = float(input("Please input the desired IoU threshold (theta):"))
    while theta <= 0 or theta >= 1:
        theta = float(input("Invalid IoU threshold, please input another value: "))
    generate_parameters(datasets, theta)