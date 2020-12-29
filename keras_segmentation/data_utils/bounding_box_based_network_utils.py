import sys
from math import isnan, log
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
from keras import backend as K

# configuration variables
from keras_segmentation.custom_losses import smooth_l1_loss

extra_pixels = 10
number_of_bounding_boxes = 15


def calculate_iou(gt_coords, pred_coords):
    from ..data_utils.standalone_IoU_model_libs import bb_intersection_over_union

    box_a = [gt_coords[1], gt_coords[0], gt_coords[1] + gt_coords[3], gt_coords[0] + gt_coords[2]]
    box_b = [pred_coords[1], pred_coords[0], pred_coords[1] + pred_coords[3], pred_coords[0] + pred_coords[2]]
    iou = bb_intersection_over_union(box_a, box_b)

    if isnan(iou):
        iou = 0.0

    if iou < 0:
        iou = 0.0

    return iou


def bounding_box_based_network_loss_python(y_true, y_pred):
    box_loss_avg_list = []
    iou_score_loss_avg_list = []

    # iterating over batches
    for batch in range(y_true.shape[0]):
        predicted_iou_score_list = []
        predicted_coords_list = []

        # iterating till the end of number of neurons with the step of 5
        for i in range(0, number_of_bounding_boxes * 5, 5):
            predicted_iou_score = y_pred[batch][i]
            predicted_iou_score_list.append(predicted_iou_score)

            x = y_pred[batch][i + 1]
            y = y_pred[batch][i + 2]
            w = y_pred[batch][i + 3]
            h = y_pred[batch][i + 4]
            predicted_coords_list.append((x, y, w, h))

        # normalize predicted iou scores
        predicted_iou_score_list = np.array(predicted_iou_score_list)
        predicted_iou_score_list = predicted_iou_score_list - np.min(predicted_iou_score_list)
        predicted_iou_score_list = predicted_iou_score_list / np.max(predicted_iou_score_list)

        # store iou that is calculated from the predicted and ground truth coordinates
        calculated_iou_score = []
        # loss for wrong bounding box predictions
        box_loss = 0

        # calculate iou score for all of the predicted bounding boxes
        for predicted_coords in predicted_coords_list:
            c_iou = calculate_iou(y_true[batch], predicted_coords)
            calculated_iou_score.append(c_iou)

            if c_iou == 0:
                c_iou = sys.float_info.epsilon

            box_loss += -log(c_iou)

        box_loss_avg_list.append(box_loss)

        iou_score_loss = 0
        # for generating loss from predicted and calculated iou score
        for calculated_iou_score_index in range(len(calculated_iou_score)):
            calculated_score = calculated_iou_score[calculated_iou_score_index]
            predicted_score = predicted_iou_score_list[calculated_iou_score_index]

            # subtracting predicted iou score from calculated score
            final_iou_score = calculated_score - predicted_score

            # if final iou score is greater than 0, it means predicted score is away from calculated score
            # log of small value is high and of high is low
            # less loss contribution if less difference, more loss contribution if more difference
            iou_score_loss += -log(1 - final_iou_score)

        iou_score_loss_avg_list.append(iou_score_loss)

    loss_output = 0

    if len(box_loss_avg_list) > 0:
        loss_output += np.mean(box_loss_avg_list)

    if len(iou_score_loss_avg_list) > 0:
        loss_output += np.mean(iou_score_loss_avg_list)

    return loss_output


def bounding_box_based_network_loss(y_true, y_pred):
    loss = tf.py_func(bounding_box_based_network_loss_python, inp=[y_true, y_pred], Tout=tf.float32)
    loss.set_shape((1,))
    loss_val = tf.dtypes.cast(loss, dtype=tf.float32)

    # loss = -tf.log(loss)

    return loss_val


def get_im_patch(im: np.ndarray, coords: Dict[str, int]) -> Tuple[np.ndarray, Dict]:
    x1 = coords['x1']
    x2 = coords['x2']
    y1 = coords['y1']
    y2 = coords['y2']

    height, width, _ = im.shape

    new_x1 = x1 - extra_pixels
    new_x1 = new_x1 if new_x1 > 0 else 0

    new_x2 = x2 + extra_pixels
    new_x2 = new_x2 if new_x2 < width else width

    new_y1 = y1 - extra_pixels
    new_y1 = new_y1 if new_y1 > 0 else 0

    new_y2 = y2 + extra_pixels
    new_y2 = new_y2 if new_y2 < height else height

    return im[new_y1:new_y2, new_x1:new_x2], {
        'x1': new_x1,
        'x2': new_x2,
        'y1': new_y1,
        'y2': new_y2,
    }


def generate_y_true(y_true, y_pred):
    new_y_true = np.zeros(shape=(y_true.shape[0], number_of_bounding_boxes * 5), dtype=np.float32)

    # iterating over batches
    for batch in range(y_true.shape[0]):
        predicted_iou_score_list = []
        predicted_coords_list = []

        # iterating till the end of number of neurons with the step of 5
        for i in range(0, number_of_bounding_boxes * 5, 5):
            predicted_iou_score = y_pred[batch][i]
            predicted_iou_score_list.append(predicted_iou_score)

            x = y_pred[batch][i + 1]
            y = y_pred[batch][i + 2]
            w = y_pred[batch][i + 3]
            h = y_pred[batch][i + 4]
            predicted_coords_list.append((x, y, w, h))

        # normalize predicted iou scores
        predicted_iou_score_list = np.array(predicted_iou_score_list)
        predicted_iou_score_list = predicted_iou_score_list - np.min(predicted_iou_score_list)
        predicted_iou_score_list = predicted_iou_score_list / np.max(predicted_iou_score_list)

        # placing normalized prediction iou score back in prediction tensor
        predicted_iou_score_list_counter = 0
        for i in range(0, number_of_bounding_boxes * 5, 5):
            y_pred[batch][i] = predicted_iou_score_list[predicted_iou_score_list_counter]
            predicted_iou_score_list_counter += 1

        # store iou that is calculated from the predicted and ground truth coordinates
        calculated_iou_score = []

        # calculate iou score for all of the predicted bounding boxes
        for predicted_coords in predicted_coords_list:
            c_iou = calculate_iou(y_true[batch], predicted_coords)
            calculated_iou_score.append(c_iou)

        calculated_iou_score_counter = 0
        for neuron_number in range(0, number_of_bounding_boxes * 5, 5):
            new_y_true[batch][neuron_number] = calculated_iou_score[calculated_iou_score_counter]
            calculated_iou_score_counter += 1

            new_y_true[batch][neuron_number + 1] = y_true[batch][0]
            new_y_true[batch][neuron_number + 2] = y_true[batch][1]
            new_y_true[batch][neuron_number + 3] = y_true[batch][2]
            new_y_true[batch][neuron_number + 4] = y_true[batch][3]

    return new_y_true, y_pred.astype(np.float32)


def bounding_box_based_network_loss_v2(y_true, y_pred):
    from keras_segmentation.custom_losses import smooth_l1_loss

    y_true_2, y_pred_2 = tf.py_func(generate_y_true, inp=[y_true, y_pred], Tout=[tf.float32, tf.float32])
    l1_loss = smooth_l1_loss(y_true_2, y_pred_2)
    return l1_loss


def bounding_box_based_network_loss_gpu(y_true, y_pred):
    # reshaping tensors to be like (batch size, number of boxes, tensors for box)
    y_pred = tf.reshape(y_pred, (-1, number_of_bounding_boxes, 5))

    # extract scores
    box_scores = y_pred[:, :, 0]
    # normalize box scores between 0 and 1
    box_scores = box_scores - tf.reduce_min(box_scores)
    box_scores = box_scores / tf.maximum(tf.reduce_max(box_scores), K.epsilon())
    # adding normalized score back to tensors
    # y_pred[:, :, 0] = box_scores

    # calculate gt coordinates
    gt_x1 = y_true[:, 0]
    gt_y1 = y_true[:, 1]
    gt_x2 = gt_x1 + y_true[:, 2]
    gt_y2 = gt_y1 + y_true[:, 3]

    gt_x1 = K.reshape(K.repeat_elements(gt_x1, number_of_bounding_boxes, axis=0), (-1, number_of_bounding_boxes))
    gt_y1 = K.reshape(K.repeat_elements(gt_y1, number_of_bounding_boxes, axis=0), (-1, number_of_bounding_boxes))
    gt_x2 = K.reshape(K.repeat_elements(gt_x2, number_of_bounding_boxes, axis=0), (-1, number_of_bounding_boxes))
    gt_y2 = K.reshape(K.repeat_elements(gt_y2, number_of_bounding_boxes, axis=0), (-1, number_of_bounding_boxes))

    gt_x1 = tf.cast(gt_x1, tf.float32)
    gt_y1 = tf.cast(gt_y1, tf.float32)
    gt_x2 = tf.cast(gt_x2, tf.float32)
    gt_y2 = tf.cast(gt_y2, tf.float32)

    # calculate prediction box coordinates
    pred_x1 = y_pred[:, :, 1]
    pred_y1 = y_pred[:, :, 2]
    pred_x2 = pred_x1 + y_pred[:, :, 3]
    pred_y2 = pred_y1 + y_pred[:, :, 4]

    # calculate intersection rectangle coordinates
    intersection_x1 = tf.maximum(pred_x1, gt_x1)
    intersection_y1 = tf.maximum(pred_y1, gt_y1)
    intersection_x2 = tf.minimum(pred_x2, gt_x2)
    intersection_y2 = tf.minimum(pred_y2, gt_y2)

    intersection_area = tf.maximum(tf.zeros_like(intersection_x1),
                                   intersection_x2 - intersection_x1 + tf.ones_like(intersection_x1)) * tf.maximum(
        tf.zeros_like(intersection_y1), intersection_y2 - intersection_y1 + tf.ones_like(intersection_y1))

    pred_area = (pred_x2 - pred_x1 + tf.ones_like(pred_x1)) * (pred_y2 - pred_y1 + tf.ones_like(pred_y1))
    gt_area = (gt_x2 - gt_x1 + tf.ones_like(gt_x1)) * (gt_y2 - gt_y1 + tf.ones_like(gt_y1))

    union_area = (pred_area + gt_area) - intersection_area

    ious = intersection_area / tf.maximum(union_area, K.epsilon())
    # ious = tf.abs(ious)
    # ious = K.clip(ious, 0.0 + K.epsilon(), 1.0 - K.epsilon())
    # ious = K.clip(ious, 0.0, 1.0)

    # ious_loss = -tf.log(ious)
    ious_loss = 1.0 - ious

    # converting (?, 15) to (?). Different values for batch
    # ious_loss = tf.reduce_mean(ious_loss, axis=-1)
    ious_loss = tf.reduce_sum(ious_loss, axis=-1)

    # converting (?) to (1). Batch to single value
    ious_loss = tf.reduce_sum(ious_loss, axis=-1)

    # normalizing predicted box score taking l1 loss with calculated score
    ious_normalized = ious - tf.reduce_min(ious)
    ious_normalized = ious_normalized / tf.maximum(tf.reduce_max(ious_normalized), K.epsilon())
    l1_loss = smooth_l1_loss(ious_normalized, box_scores)

    return tf.cast(ious_loss, tf.float32) + tf.cast(l1_loss, tf.float32)
