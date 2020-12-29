import os
from math import ceil

import cv2
import numpy as np

from keras_segmentation.data_utils.data_loader import get_image_arr


def get_video_frames(video_path: str):
    assert os.path.isfile(video_path), "Provided path should exist and it should be a video file"

    cap = cv2.VideoCapture(video_path)

    cur_frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        yield cap.get(cv2.CAP_PROP_FRAME_COUNT), cur_frame_number, frame

        cur_frame_number += 1
    cap.release()


def calculate_number_of_patches(height, width, height_of_patch, width_of_patch):
    number_of_rows = height // height_of_patch
    number_of_cols = width // width_of_patch
    return number_of_rows * number_of_cols


def get_patches_from_frame(frame: np.ndarray, patch_height, patch_width):
    patches_list = []
    for y in range(0, frame.shape[0] - patch_height, patch_height):
        for x in range(0, frame.shape[1] - patch_width, patch_width):
            patches_list.append(frame[y:y + patch_height, x:x + patch_width])
    return patches_list


def convert_frames_to_volume_of_patches(frames, number_of_patches, patch_width_height):
    # making patches list
    frame_patches_list = []
    for i in range(number_of_patches):
        frame_patches_list.append([])

    for frm in frames:
        frame_patches = get_patches_from_frame(frm, patch_width_height, patch_width_height)
        for frame_patch_index, frame_patch in enumerate(frame_patches):
            frame_patches_list[frame_patch_index].append(frame_patch)
    return frame_patches_list


def get_video_volumes(video_path: str, annotations_folder: str, number_of_frames_in_volume=8, patch_width_height=224):
    key_frame_number = int(ceil(number_of_frames_in_volume / 2))

    video_filename = os.path.basename(video_path)
    video_filename_wo_ext = os.path.splitext(video_filename)[0]

    video_frames_generator = get_video_frames(video_path)
    for total_frames, frame_number, frame in video_frames_generator:
        current_key_frame_number = key_frame_number + frame_number

        all_frames = []

        frame_filename = f"{video_filename_wo_ext}_{str(current_key_frame_number).zfill(6)}.png"
        annotation_path = os.path.join(annotations_folder, frame_filename)
        if os.path.exists(annotation_path):
            frame = get_image_arr(frame, frame.shape[1], frame.shape[0], odering='channels_last')
            # add already loaded frame
            all_frames.append(frame)

            frames_start_index = frame_number

            cnt = 0
            # load n-1 frames as well
            for total_frames, frame_number, frame in video_frames_generator:
                frame = get_image_arr(frame, frame.shape[1], frame.shape[0], odering='channels_last')
                all_frames.append(frame)
                cnt += 1
                if cnt >= number_of_frames_in_volume - 1:
                    break

            if len(all_frames) == number_of_frames_in_volume:
                annotation = cv2.imread(annotation_path)
                number_of_patches = calculate_number_of_patches(annotation.shape[0], annotation.shape[1],
                                                                patch_width_height, patch_width_height)

                frame_patches_list = convert_frames_to_volume_of_patches(all_frames, number_of_patches,
                                                                         patch_width_height)

                annotation_patches = get_patches_from_frame(annotation, patch_width_height, patch_width_height)

                for patch_number in range(number_of_patches):
                    yield frame_patches_list[patch_number], annotation_patches[patch_number]
