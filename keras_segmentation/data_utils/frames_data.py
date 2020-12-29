import math
import os
from typing import List, Dict

import cv2


def get_frame_and_ground_truth_crop(frame_path, ground_truth_path=None):
    extra_area = 50

    if isinstance(frame_path, str):
        frame = cv2.imread(frame_path)
    else:
        frame = frame_path

    if isinstance(ground_truth_path, str):
        ground_truth = cv2.imread(ground_truth_path)
    else:
        ground_truth = ground_truth_path

    height, width, _ = frame.shape
    split_height = math.floor(height / 3)
    split_width = math.floor(width / 3)

    # top left
    frame1 = frame[0:split_height + extra_area, 0:split_width + extra_area, :]
    # top middle
    frame2 = frame[0:split_height + extra_area, split_width - extra_area:split_width + split_width + extra_area, :]
    # top right
    frame3 = frame[0:split_height + extra_area, (split_width + split_width) - extra_area:width, :]
    # middle left
    frame4 = frame[split_height - extra_area:split_height + split_height + extra_area, 0:split_width + extra_area, :]
    # middle middle
    frame5 = frame[split_height - extra_area:split_height + split_height + extra_area,
             split_width - extra_area:split_width + split_width + extra_area, :]
    # middle right
    frame6 = frame[split_height - extra_area:split_height + split_height + extra_area,
             (split_width + split_width) - extra_area:width, :]
    # bottom left
    frame7 = frame[(split_height + split_height) - extra_area:height, 0:split_width + extra_area, :]
    # bottom middle
    frame8 = frame[(split_height + split_height) - extra_area:height,
             split_width - extra_area:split_width + split_width + extra_area, :]
    # bottom right
    frame9 = frame[(split_height + split_height) - extra_area:height, (split_width + split_width) - extra_area:width, :]

    if ground_truth_path is not None:
        # top left
        ground_truth1 = ground_truth[0:split_height + extra_area, 0:split_width + extra_area, :]
        # top middle
        ground_truth2 = ground_truth[0:split_height + extra_area,
                        split_width - extra_area:split_width + split_width + extra_area, :]
        # top right
        ground_truth3 = ground_truth[0:split_height + extra_area, (split_width + split_width) - extra_area:width, :]
        # middle left
        ground_truth4 = ground_truth[split_height - extra_area:split_height + split_height + extra_area,
                        0:split_width + extra_area, :]
        # middle middle
        ground_truth5 = ground_truth[split_height - extra_area:split_height + split_height + extra_area,
                        split_width - extra_area:split_width + split_width + extra_area, :]
        # middle right
        ground_truth6 = ground_truth[split_height - extra_area:split_height + split_height + extra_area,
                        (split_width + split_width) - extra_area:width, :]
        # bottom left
        ground_truth7 = ground_truth[(split_height + split_height) - extra_area:height, 0:split_width + extra_area, :]
        # bottom middle
        ground_truth8 = ground_truth[(split_height + split_height) - extra_area:height,
                        split_width - extra_area:split_width + split_width + extra_area, :]
        # bottom right
        ground_truth9 = ground_truth[(split_height + split_height) - extra_area:height,
                        (split_width + split_width) - extra_area:width, :]

        return [frame1, frame2, frame3, frame4, frame5, frame6, frame7, frame8, frame9], [ground_truth1, ground_truth2,
                                                                                          ground_truth3, ground_truth4,
                                                                                          ground_truth5, ground_truth6,
                                                                                          ground_truth7, ground_truth8,
                                                                                          ground_truth9]
    else:
        return [frame1, frame2, frame3, frame4, frame5, frame6, frame7, frame8, frame9]


def get_video_wise_list(frames_list: List[str]):
    video_wise_frame_paths: Dict[str, List[str]] = {}
    for frame_path in frames_list:
        # Clip_041_000052.png
        filename = os.path.basename(frame_path)
        # Clip_041
        video_name = filename[0:8]

        if video_name not in video_wise_frame_paths.keys():
            video_wise_frame_paths[video_name] = []

        video_wise_frame_paths[video_name].append(frame_path)
    return video_wise_frame_paths
