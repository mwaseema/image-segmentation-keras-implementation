import random
from math import isnan
from random import randint
from typing import Dict, List, Tuple

import numpy as np
import cv2
import skimage.measure
from scipy.optimize import linear_sum_assignment


def get_region_props(image):
    im = image.copy()

    if len(im.shape) == 3 and im.shape[2] == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    label_image = skimage.measure.label(im)
    region_props = skimage.measure.regionprops(label_image)
    return region_props


def get_bounding_boxes(binary_mask: np.ndarray) -> List[Dict[str, int]]:
    binary_mask: np.ndarray = binary_mask.copy()

    if len(binary_mask.shape) and binary_mask.shape[2] == 3:
        binary_mask = cv2.cvtColor(binary_mask, cv2.COLOR_BGR2GRAY)

    binary_mask[binary_mask > 0] = 255

    region_props = get_region_props(binary_mask)

    boxes = []
    for region_prop in region_props:
        y1, x1, y2, x2 = region_prop.bbox
        boxes.append({
            'x1': x1,
            'x2': x2,
            'y1': y1,
            'y2': y2,
        })

    return boxes


def bb_intersection_over_union(boxA: Tuple, boxB: Tuple) -> float:
    """
    Get IoU between two bounding boxes

    :param boxA: (y1, x1, y2, x2)
    :param boxB: (y1, x1, y2, x2)
    :return: IoU value
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # print(iou)
    # return the intersection over union value
    return iou


def get_iou_score(mask_1: np.ndarray, mask_2: np.ndarray):
    props_mask_1 = get_region_props(mask_1)
    props_mask_2 = get_region_props(mask_2)

    IOU_bbx_mul = np.zeros((props_mask_1.__len__(), props_mask_2.__len__()))

    # returning -1 only if there is no gt bbox found
    if len(props_mask_1) == 0:
        return -1

    for g_b in range(0, props_mask_1.__len__()):
        for p_b in range(0, props_mask_2.__len__()):
            IOU_bbx_mul[g_b, p_b] = bb_intersection_over_union(props_mask_1[g_b].bbox, props_mask_2[p_b].bbox)

    row_ind, col_ind = linear_sum_assignment(1 - IOU_bbx_mul)

    calculated_IoU = []
    for ir in range(0, len(row_ind)):
        IOU_bbx_s = IOU_bbx_mul[row_ind[ir], col_ind[ir]]

        calculated_IoU.append(IOU_bbx_s)

        # if IOU_bbx_s >= 0.5:
        #     TP = TP + 1
        # else:
        #     FP = FP + 1
        #     # FN = FN + 1
        #     FP_loc = 1
    # if (props_im.__len__() - props_gt.__len__()) > 0:
    #     FP = FP + (props_im.__len__() - props_gt.__len__())
    #     FP_loc = 1

    if len(calculated_IoU) > 0:
        calculated_IoU_mean = np.mean(calculated_IoU)
    else:
        calculated_IoU_mean = 0.0

    if isnan(calculated_IoU_mean):
        calculated_IoU_mean = 0.0

    if calculated_IoU_mean < 0:
        calculated_IoU_mean = 0.0

    return calculated_IoU_mean


def generate_scaling_box_coordinates(height, width, x1, x2, y1, y2, extra_pixels=5) -> List[Dict[str, int]]:
    boxes = []
    for i in range(extra_pixels + 1):
        new_x1 = x1 - i
        new_x1 = new_x1 if new_x1 > 0 else 0

        new_x2 = x2 + i
        new_x2 = new_x2 if new_x2 < width else width

        new_y1 = y1 - i
        new_y1 = new_y1 if new_y1 > 0 else 0

        new_y2 = y2 + i
        new_y2 = new_y2 if new_y2 < height else height

        boxes.append({
            'x1': new_x1,
            'x2': new_x2,
            'y1': new_y1,
            'y2': new_y2,
        })
    return boxes


def generate_translational_bounding_box_coordinates(height, width, x1, x2, y1, y2, extra_pixels) -> List[
    Dict[str, int]]:
    boxes = []
    # for x1 and x2, left
    for i in range(1, extra_pixels + 1):
        left_x1 = x1 - i
        if left_x1 < 0:
            break

        left_x2 = x2 - i

        boxes.append({
            'x1': left_x1,
            'x2': left_x2,
            'y1': y1,
            'y2': y2,
        })

    # for x1 and x2, right
    for i in range(1, extra_pixels + 1):
        right_x1 = x1 + i

        right_x2 = x2 - i
        if right_x2 > width:
            break

        boxes.append({
            'x1': right_x1,
            'x2': right_x2,
            'y1': y1,
            'y2': y2,
        })

    # for y1 and y2, top
    for i in range(1, extra_pixels + 1):
        new_y1 = x1 - i
        if new_y1 < 0:
            break

        new_y2 = x2 - i

        boxes.append({
            'x1': x1,
            'x2': x2,
            'y1': new_y1,
            'y2': new_y2,
        })

    # for y1 and y2, bottom
    for i in range(1, extra_pixels + 1):
        new_y1 = x1 + i

        new_y2 = x2 + i
        if new_y2 > height:
            break

        boxes.append({
            'x1': x1,
            'x2': x2,
            'y1': new_y1,
            'y2': new_y2,
        })

    return boxes


def generate_box_coordinates(height: int, width: int, bounding_box: Dict[str, int], extra_pixels: int = 5,
                             number_of_bounding_boxes: int = 100) -> List[Dict[str, int]]:
    x1 = bounding_box['x1']
    x2 = bounding_box['x2']
    y1 = bounding_box['y1']
    y2 = bounding_box['y2']

    scaling_bounding_boxes = generate_scaling_box_coordinates(height, width, x1, x2, y1, y2, extra_pixels)
    translational_boxes = []
    for scaled_by, scaling_bounding_box in enumerate(scaling_bounding_boxes):
        s_x1 = scaling_bounding_box['x1']
        s_y1 = scaling_bounding_box['y1']
        s_x2 = scaling_bounding_box['x2']
        s_y2 = scaling_bounding_box['y2']
        translational_boxes.extend(
            generate_translational_bounding_box_coordinates(height, width, s_x1, s_x2, s_y1, s_y2,
                                                            extra_pixels - scaled_by))

    all_coords = []
    all_coords.extend(scaling_bounding_boxes)
    all_coords.extend(translational_boxes)

    min_x = x1 - extra_pixels
    min_x = min_x if min_x > 0 else 0

    min_y = y1 - extra_pixels
    min_y = min_y if min_y > 0 else 0

    max_x = x2 + extra_pixels
    max_x = max_x if max_x <= width else width

    max_y = y2 + extra_pixels
    max_y = max_y if max_y <= height else height

    new_coordinates = []
    for _ in range(number_of_bounding_boxes * 2):
        new_x1 = randint(min_x, max_x)
        new_x2 = randint(new_x1, max_x)
        new_y1 = randint(min_y, max_y)
        new_y2 = randint(new_y1, max_y)

        new_coordinates.append({
            'x1': new_x1,
            'x2': new_x2,
            'y1': new_y1,
            'y2': new_y2,
        })

    all_coords.extend(new_coordinates)

    all_coords_processed = []
    for coords in all_coords:
        x1 = coords['x1']
        y1 = coords['y1']
        x2 = coords['x2']
        y2 = coords['y2']

        # if 2nd coords are greater than 1 and have difference of 2 pixels at least
        if (x2 > x1 and x2 - x1 >= 2) and (y2 > y1 and y2 - y1 >= 2):
            all_coords_processed.append(coords)

    if len(all_coords_processed) > number_of_bounding_boxes:
        all_coords_processed = all_coords_processed[0:number_of_bounding_boxes]
    return all_coords_processed


def get_mask_while_checking_iou(ground_truth_mask: np.ndarray, gt_bounding_box: Dict[str, int],
                                bounding_box_coordinates: List[Dict[str, int]],
                                iou_threshold=0.5):
    for bbox_coords in bounding_box_coordinates:
        x1 = bbox_coords['x1']
        x2 = bbox_coords['x2']
        y1 = bbox_coords['y1']
        y2 = bbox_coords['y2']

        new_gt_mask = np.zeros(shape=ground_truth_mask.shape, dtype=np.uint8)
        new_gt_mask[gt_bounding_box['y1']:gt_bounding_box['y2'], gt_bounding_box['x1']:gt_bounding_box['x2']] = 1

        new_mask = np.zeros(shape=ground_truth_mask.shape, dtype=np.uint8)
        new_mask[y1:y2, x1:x2] = 1

        iou_score = get_iou_score(new_gt_mask, new_mask)

        if iou_score >= iou_threshold:
            yield new_mask, iou_score, bbox_coords


def get_random_background_coords(gt_mask: np.ndarray):
    mask_height, mask_width = gt_mask.shape[0:2]

    while True:
        rand_x1 = random.randint(0, mask_width)
        rand_y1 = random.randint(0, mask_height)

        rand_w = random.randint(10, 25)
        rand_h = random.randint(10, 25)

        rand_x2 = rand_x1 + rand_w
        rand_y2 = rand_y1 + rand_h

        if rand_x2 < mask_width and rand_y2 < mask_height:
            mask_patch = gt_mask[rand_y1:rand_y2, rand_x1:rand_x2]
            # there is no pixel with value greater than 0
            if np.count_nonzero(mask_patch > 0) == 0:
                return {
                    'x1': rand_x1,
                    'y1': rand_y1,
                    'x2': rand_x2,
                    'y2': rand_y2,
                }


def generate_augmentations(ground_truth_mask: np.ndarray, bounding_box: Dict[str, int], max_augmentations=10,
                           iou_threshold=0.5):
    height, width, channel = ground_truth_mask.shape

    # generate box coordinates
    bounding_box_coordinates = generate_box_coordinates(height, width, bounding_box, 2, max_augmentations * 5)

    # yielding original ground truth and its iou score
    yield ground_truth_mask, 1.0, bounding_box

    augmentation_count = 1
    iteration_count = 0
    # get mask for boxes with iou greater than 0.5
    for new_mask, iou_score, bbox_coords in get_mask_while_checking_iou(ground_truth_mask, bounding_box,
                                                                        bounding_box_coordinates, iou_threshold):
        if augmentation_count > max_augmentations:
            break
        yield new_mask, iou_score, bbox_coords
        augmentation_count += 1

        if iteration_count % 10 == 0:
            yield new_mask, 0.0, get_random_background_coords(ground_truth_mask)
        iteration_count += 1


def get_patch_of_image_with_bounding_box_coords(image: np.ndarray, bounding_box_coords: Dict[str, int]):
    x1 = bounding_box_coords['x1']
    x2 = bounding_box_coords['x2']
    y1 = bounding_box_coords['y1']
    y2 = bounding_box_coords['y2']

    im = image.copy()
    im = im[y1:y2, x1:x2]
    return im
