import argparse
from typing import Union, Dict

from keras.models import load_model
import glob
import cv2
import numpy as np
import random
import os
from tqdm import tqdm

from .data_utils.i3d_inception_data_utils import convert_frames_to_volume_of_patches, calculate_number_of_patches
from .data_utils.standalone_IoU_model_libs import get_bounding_boxes
from .train import find_latest_checkpoint
import os
from .data_utils.data_loader import get_image_arr, get_segmentation_arr
import json
from .models.config import IMAGE_ORDERING
from . import metrics
from .models import model_from_name
from .data_utils import standalone_IoU_model_libs
from .data_utils.bounding_box_based_network_utils import number_of_bounding_boxes, get_im_patch

import six

random.seed(0)
class_colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(5000)]


def model_from_checkpoint_path(checkpoints_path):
    assert (os.path.isfile(checkpoints_path + "_config.json")), "Checkpoint not found."
    model_config = json.loads(open(checkpoints_path + "_config.json", "r").read())
    latest_weights = find_latest_checkpoint(checkpoints_path)
    assert (not latest_weights is None), "Checkpoint not found."
    model = model_from_name[model_config['model_class']](model_config['n_classes'],
                                                         input_height=model_config['input_height'],
                                                         input_width=model_config['input_width'])
    print("loaded weights ", latest_weights)
    model.load_weights(latest_weights)
    return model


def model_from_checkpoint_given_path(checkpoints_path):
    # remove extension which is a number
    checkpoints_path_wo_ext = os.path.splitext(checkpoints_path)[0]
    assert (os.path.isfile(checkpoints_path_wo_ext + "_config.json")), "Checkpoint not found."
    model_config = json.loads(open(checkpoints_path_wo_ext + "_config.json", "r").read())
    latest_weights = checkpoints_path
    assert (not latest_weights is None), "Checkpoint not found."
    model = model_from_name[model_config['model_class']](model_config['n_classes'],
                                                         input_height=model_config['input_height'],
                                                         input_width=model_config['input_width'])
    print("loaded weights ", latest_weights)
    model.load_weights(latest_weights)
    return model


def model_from_specific_checkpoint_path(checkpoints_path, model_number):
    assert (os.path.isfile(checkpoints_path + "_config.json")), "Checkpoint not found."
    model_config = json.loads(open(checkpoints_path + "_config.json", "r").read())
    latest_weights = f'{checkpoints_path}.{model_number}'
    assert (not latest_weights is None), "Checkpoint not found."
    model = model_from_name[model_config['model_class']](model_config['n_classes'],
                                                         input_height=model_config['input_height'],
                                                         input_width=model_config['input_width'])
    print("loaded weights ", latest_weights)
    model.load_weights(latest_weights)
    return model


def predict(model=None, inp=None, out_fname=None, checkpoints_path=None, threshold=None):
    if model is None and (not checkpoints_path is None):
        model = model_from_checkpoint_path(checkpoints_path)

    assert (not inp is None)
    assert ((type(inp) is np.ndarray) or isinstance(inp,
                                                    six.string_types)), "Inupt should be the CV image or the input file name"

    if isinstance(inp, six.string_types):
        inp = cv2.imread(inp)

    assert len(inp.shape) == 3, "Image should be h,w,3 "
    orininal_h = inp.shape[0]
    orininal_w = inp.shape[1]

    output_width = model.output_width
    output_height = model.output_height
    input_width = model.input_width
    input_height = model.input_height
    n_classes = model.n_classes

    x = get_image_arr(inp, input_width, input_height, odering=IMAGE_ORDERING)
    pr = model.predict(np.array([x]))[0]

    probabilities = cv2.resize(pr.reshape((output_height, output_width, n_classes)), (orininal_w, orininal_h))

    pr = pr.reshape((output_height, output_width, n_classes)).argmax(axis=2)

    seg_img = np.zeros((output_height, output_width, 3))

    if n_classes > 2:
        colors = class_colors
    else:
        colors = [(0, 0, 0), (255, 255, 255)]

    # # if threshold is given
    # if threshold is not None:
    #     # Convert values greater than threshold to 1
    #     pr[pr >= threshold] = 1

    for c in range(n_classes):
        seg_img[:, :, 0] += ((pr[:, :] == c) * (colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((pr[:, :] == c) * (colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((pr[:, :] == c) * (colors[c][2])).astype('uint8')

    seg_img = cv2.resize(seg_img, (orininal_w, orininal_h))

    if not out_fname is None:
        cv2.imwrite(out_fname, seg_img)

    return seg_img, probabilities


def predict_i3d_inception(model=None, input_img_frames_patch_volume=None, output_file_path=None, checkpoints_path=None,
                          threshold=None):
    if model is None and (not checkpoints_path is None):
        model = model_from_checkpoint_path(checkpoints_path)

    assert (input_img_frames_patch_volume is not None)
    assert (type(
        input_img_frames_patch_volume) is np.ndarray), "Inupt should be the image volume in the form of numpy array"

    assert len(input_img_frames_patch_volume.shape) == 4, "Image should be v,h,w,3 "
    orininal_h = input_img_frames_patch_volume.shape[1]
    orininal_w = input_img_frames_patch_volume.shape[2]

    output_width = model.output_width
    output_height = model.output_height
    input_width = model.input_width
    input_height = model.input_height
    n_classes = model.n_classes

    # pre-processing frames in volume
    for frame_index in range(input_img_frames_patch_volume.shape[0]):
        input_img_frames_patch_volume[frame_index] = get_image_arr(input_img_frames_patch_volume[frame_index],
                                                                   input_width, input_height, odering=IMAGE_ORDERING)

    pr = model.predict(np.array([input_img_frames_patch_volume]))[0]

    # implementing threshold value
    if threshold is not None:
        assert 0 < threshold < 0.5, "Threshold value should be greater than 0 or less than 0.5"
        threshold_diff = 0.5 - threshold
        pr[:, 0] = pr[:, 0] - threshold_diff
        pr[:, 1] = pr[:, 1] + threshold_diff

    pr = pr.reshape((output_height, output_width, n_classes))
    pr_out = pr.copy()
    pr = pr.argmax(axis=2)

    seg_img = np.zeros((output_height, output_width, 3))

    if n_classes > 2:
        colors = class_colors
    else:
        colors = [(0, 0, 0), (255, 255, 255)]

    for c in range(n_classes):
        seg_img[:, :, 0] += ((pr[:, :] == c) * (colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((pr[:, :] == c) * (colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((pr[:, :] == c) * (colors[c][2])).astype('uint8')

    seg_img = cv2.resize(seg_img, (orininal_w, orininal_h))

    if not output_file_path is None:
        cv2.imwrite(output_file_path, seg_img)

    return seg_img, pr_out


def predict_with_weighted_output(model=None, inp=None, out_fname=None, checkpoints_path=None, threshold=None):
    if model is None and (not checkpoints_path is None):
        model = model_from_checkpoint_path(checkpoints_path)

    assert (not inp is None)
    assert ((type(inp) is np.ndarray) or isinstance(inp,
                                                    six.string_types)), "Inupt should be the CV image or the input file name"

    if isinstance(inp, six.string_types):
        inp = cv2.imread(inp)

    assert len(inp.shape) == 3, "Image should be h,w,3 "
    orininal_h = inp.shape[0]
    orininal_w = inp.shape[1]

    output_width = model.output_width
    output_height = model.output_height
    input_width = model.input_width
    input_height = model.input_height
    n_classes = model.n_classes

    x = get_image_arr(inp, input_width, input_height, odering=IMAGE_ORDERING)
    pr_out = model.predict(np.array([x]))

    probabilities = []
    seg_masks = []
    for i in range(len(pr_out)):
        pr = pr_out[i][0]
        probabilities.append(cv2.resize(pr.reshape((output_height, output_width, n_classes)), (orininal_w, orininal_h)))

        pr = pr.reshape((output_height, output_width, n_classes)).argmax(axis=2)

        seg_img = np.zeros((output_height, output_width, 3))

        if n_classes > 2:
            colors = class_colors
        else:
            colors = [(0, 0, 0), (255, 255, 255)]

        for c in range(n_classes):
            seg_img[:, :, 0] += ((pr[:, :] == c) * (colors[c][0])).astype('uint8')
            seg_img[:, :, 1] += ((pr[:, :] == c) * (colors[c][1])).astype('uint8')
            seg_img[:, :, 2] += ((pr[:, :] == c) * (colors[c][2])).astype('uint8')

        seg_img = cv2.resize(seg_img, (orininal_w, orininal_h))
        seg_masks.append(seg_img)

        if out_fname is not None:
            if i == 0:
                output_path = out_fname
            else:
                output_filename = os.path.basename(out_fname)
                output_dirname = os.path.dirname(out_fname)
                output_dirname = os.path.abspath(os.path.join(output_dirname, '..', '2nd_detection'))
                output_path = os.path.join(output_dirname, output_filename)

            cv2.imwrite(output_path, seg_img)
    return seg_masks, probabilities


def predict_temporal_with_weighted_output(model=None, inp=None, out_fname=None, checkpoints_path=None, threshold=None):
    if model is None and (not checkpoints_path is None):
        model = model_from_checkpoint_path(checkpoints_path)

    assert (not inp is None)
    assert ((type(inp) is np.ndarray) or isinstance(inp,
                                                    six.string_types)), "Inupt should be the numpy array or the input file name"

    if isinstance(inp, six.string_types):
        inp = np.load(inp)

    assert len(inp.shape) == 4, "Image should be dept,h,w,3 "
    orininal_h = inp.shape[1]
    orininal_w = inp.shape[2]

    output_width = model.output_width
    output_height = model.output_height
    input_width = model.input_width
    input_height = model.input_height
    n_classes = model.n_classes

    x = []
    for i in range(len(inp)):
        x.append(get_image_arr(inp[i], input_width, input_height, odering=IMAGE_ORDERING))
    x = np.array(x)

    pr_out = model.predict(np.array([x]))

    probabilities = []
    seg_masks = []
    for i in range(len(pr_out)):
        pr = pr_out[i][0]
        probabilities.append(cv2.resize(pr.reshape((output_height, output_width, n_classes)), (orininal_w, orininal_h)))

        pr = pr.reshape((output_height, output_width, n_classes)).argmax(axis=2)

        seg_img = np.zeros((output_height, output_width, 3))

        if n_classes > 2:
            colors = class_colors
        else:
            colors = [(0, 0, 0), (255, 255, 255)]

        for c in range(n_classes):
            seg_img[:, :, 0] += ((pr[:, :] == c) * (colors[c][0])).astype('uint8')
            seg_img[:, :, 1] += ((pr[:, :] == c) * (colors[c][1])).astype('uint8')
            seg_img[:, :, 2] += ((pr[:, :] == c) * (colors[c][2])).astype('uint8')

        seg_img = cv2.resize(seg_img, (orininal_w, orininal_h))
        seg_masks.append(seg_img)

        if out_fname is not None:
            if i == 0:
                output_path = out_fname
            else:
                output_filename = os.path.basename(out_fname)
                output_dirname = os.path.dirname(out_fname)
                output_dirname = os.path.abspath(os.path.join(output_dirname, '..', '2nd_detection'))
                output_path = os.path.join(output_dirname, output_filename)

            cv2.imwrite(output_path, seg_img)
    return seg_masks, probabilities


def predict_bounding_box_based_network(model=None, inp=None, reference_mask_folder=None, out_fname=None,
                                       checkpoints_path=None, threshold=None):
    if model is None and (not checkpoints_path is None):
        model = model_from_checkpoint_path(checkpoints_path)

    assert (not inp is None)
    assert ((type(inp) is np.ndarray) or isinstance(inp,
                                                    six.string_types)), "Inupt should be the CV image or the input file name"
    assert reference_mask_folder is not None and os.path.isdir(
        reference_mask_folder), "reference_mask_folder should be a folder"

    if isinstance(inp, six.string_types):
        inp = cv2.imread(inp, 1)

    assert len(inp.shape) == 3, "Image should be h,w,3 "
    orininal_h = inp.shape[0]
    orininal_w = inp.shape[1]

    filename = os.path.basename(inp)
    reference_mask = os.path.join(reference_mask_folder, filename)
    reference_mask = cv2.imread(reference_mask, 1)

    output_width = model.output_width
    output_height = model.output_height
    input_width = model.input_width
    input_height = model.input_height
    n_classes = model.n_classes

    output_mask = np.zeros(shape=inp.shape, dtype=np.uint8)

    bounding_boxes = get_bounding_boxes(reference_mask)
    for bounding_box in bounding_boxes:
        inp_patch, patch_coords = get_im_patch(inp, bounding_box)
        patch_mask = np.zeros(shape=inp_patch.shape, dtype=np.uint8)

        x = get_image_arr(inp_patch, input_width, input_height, odering=IMAGE_ORDERING)
        pr = model.predict(np.array([x]))[0]

        predicted_scores = []
        predicted_coords = []
        # iterating on output neurons for number of bounding boxes
        for neuron_index in range(0, number_of_bounding_boxes * 5, 5):
            predicted_scores.append(pr[neuron_index])

            pr_x1 = pr[neuron_index + 1]
            pr_y1 = pr[neuron_index + 2]
            pr_w = pr[neuron_index + 3]
            pr_h = pr[neuron_index + 4]
            pr_x2 = pr_x1 + pr_w
            pr_y2 = pr_y1 + pr_h

            predicted_coords.append({
                'x1': round(pr_x1),
                'x2': round(pr_x2),
                'y1': round(pr_y1),
                'y2': round(pr_y2),
            })

        # getting coords with highest iou score
        max_index = predicted_scores.index(max(predicted_scores))
        max_coords = predicted_coords[max_index]

        # highlighting box in patch mask
        patch_mask[max_coords['y1']:max_coords['y2'], max_coords['x1']:max_coords['x2']] = 255

        # placing patch mask in full output mask
        output_mask[patch_coords['y1']:patch_coords['y2'], patch_coords['x1']:patch_coords['x2']] = patch_mask

    if out_fname is not None:
        cv2.imwrite(out_fname, output_mask)

    return output_mask


def predict_bounding_box_iou_based_network(model=None, inp=None, reference_mask_folder=None, out_fname=None,
                                           checkpoints_path=None, threshold=None):
    if model is None and (not checkpoints_path is None):
        model = model_from_checkpoint_path(checkpoints_path)

    assert (not inp is None)
    assert ((type(inp) is np.ndarray) or isinstance(inp,
                                                    six.string_types)), "Inupt should be the CV image or the input file name"
    assert reference_mask_folder is not None and os.path.isdir(
        reference_mask_folder), "reference_mask_folder should be a folder"

    filename = os.path.basename(inp)

    if isinstance(inp, six.string_types):
        inp = cv2.imread(inp, 1)

    assert len(inp.shape) == 3, "Image should be h,w,3 "
    orininal_h = inp.shape[0]
    orininal_w = inp.shape[1]

    reference_mask = os.path.join(reference_mask_folder, filename)
    reference_mask = cv2.imread(reference_mask, 1)

    input_width = model.input_width
    input_height = model.input_height
    n_classes = model.n_classes

    output_mask = np.zeros(shape=inp.shape, dtype=np.uint8)

    bounding_boxes = get_bounding_boxes(reference_mask)
    for bounding_box in bounding_boxes:
        inp_patch, patch_coords = get_im_patch(inp, bounding_box)
        patch_mask = np.zeros(shape=inp_patch.shape, dtype=np.uint8)

        x = get_image_arr(inp_patch, input_width, input_height, odering=IMAGE_ORDERING)
        pr = model.predict(np.array([x]))[0]

        predicted_scores = []
        predicted_coords = []
        # iterating on output neurons for number of bounding boxes
        for neuron_index in range(0, number_of_bounding_boxes * 5, 5):
            predicted_scores.append(pr[neuron_index])

            pr_x1 = pr[neuron_index + 1]
            pr_y1 = pr[neuron_index + 2]
            pr_w = pr[neuron_index + 3]
            pr_h = pr[neuron_index + 4]
            pr_x2 = pr_x1 + pr_w
            pr_y2 = pr_y1 + pr_h

            predicted_coords.append({
                'x1': round(pr_x1),
                'x2': round(pr_x2),
                'y1': round(pr_y1),
                'y2': round(pr_y2),
            })

        # getting coords with highest iou score
        max_index = predicted_scores.index(max(predicted_scores))
        max_coords = predicted_coords[max_index]

        # highlighting box in patch mask
        patch_mask[max_coords['y1']:max_coords['y2'], max_coords['x1']:max_coords['x2']] = 255

        # placing patch mask in full output mask
        output_mask[patch_coords['y1']:patch_coords['y2'], patch_coords['x1']:patch_coords['x2']] = patch_mask

    if out_fname is not None:
        cv2.imwrite(out_fname, output_mask)

    return output_mask


def predict_IoU_network(model=None, inp_frame: Union[str, np.ndarray] = None,
                        input_prediction_mask: Union[str, np.ndarray] = None, checkpoints_path=None, threshold=None):
    if model is None and (not checkpoints_path is None):
        model = model_from_checkpoint_path(checkpoints_path)

    assert (not inp_frame is None)
    assert ((type(inp_frame) is np.ndarray) or isinstance(inp_frame,
                                                          six.string_types)), "Input frame should be a CV image or path to the file"

    assert (not input_prediction_mask is None)
    assert ((type(input_prediction_mask) is np.ndarray) or isinstance(input_prediction_mask,
                                                                      six.string_types)), "Input prediction mask should be a CV image or path to the file"

    if isinstance(inp_frame, six.string_types):
        inp_frame = cv2.imread(inp_frame)

    if isinstance(input_prediction_mask, six.string_types):
        input_prediction_mask = cv2.imread(input_prediction_mask)

    assert len(inp_frame.shape) == 3, "Image should be h,w,3 "

    input_width = model.input_width
    input_height = model.input_height
    n_classes = model.n_classes

    bounding_boxes = standalone_IoU_model_libs.get_bounding_boxes(input_prediction_mask)
    if len(bounding_boxes) > 1:
        masks_at_max_iou = []
        max_iou_scores = []
        for bounding_box in bounding_boxes:
            max_iou_score = 0
            mask_at_max_iou = None
            for new_mask, iou_score, bbox_coords in standalone_IoU_model_libs.generate_augmentations(
                    input_prediction_mask,
                    bounding_box, 20,
                    0.5):
                patch_of_input_frame = standalone_IoU_model_libs.get_patch_of_image_with_bounding_box_coords(inp_frame,
                                                                                                             bbox_coords)

                x = get_image_arr(patch_of_input_frame, input_width, input_height, odering=IMAGE_ORDERING)
                pr = model.predict(np.array([x]))[0]

                if pr >= max_iou_score:
                    max_iou_score = pr
                    mask_at_max_iou = new_mask

                masks_at_max_iou.append(mask_at_max_iou)
                max_iou_scores.append(max_iou_score)

        return masks_at_max_iou, max_iou_scores
    else:  # doesn't have any box
        return input_prediction_mask, 0.0


def predict_IoU_network_plain(model=None, inp_frame: Union[str, np.ndarray] = None,
                              bbox_coords: Dict[str, int] = None, checkpoints_path=None,
                              threshold=None):
    if model is None and (not checkpoints_path is None):
        model = model_from_checkpoint_path(checkpoints_path)

    assert (not inp_frame is None)
    assert ((type(inp_frame) is np.ndarray) or isinstance(inp_frame,
                                                          six.string_types)), "Input frame should be a CV image or path to the file"

    if isinstance(inp_frame, six.string_types):
        inp_frame = cv2.imread(inp_frame)

    assert len(inp_frame.shape) == 3, "Image should be h,w,3 "

    input_width = model.input_width
    input_height = model.input_height
    n_classes = model.n_classes

    patch_of_input_frame = standalone_IoU_model_libs.get_patch_of_image_with_bounding_box_coords(inp_frame, bbox_coords)

    x = get_image_arr(patch_of_input_frame, input_width, input_height, odering=IMAGE_ORDERING)
    pr = model.predict(np.array([x]))[0]

    return pr


def i3d_predict(model=None, inp=None, out_fname=None, checkpoints_path=None, threshold=None):
    if model is None and (not checkpoints_path is None):
        model = model_from_checkpoint_path(checkpoints_path)

    assert (not inp is None)
    assert ((type(inp) is np.ndarray) or isinstance(inp,
                                                    six.string_types)), "Inupt should be the CV image or the input file name"

    orininal_h = 224
    orininal_w = 224

    output_width = model.output_width
    output_height = model.output_height
    input_width = model.input_width
    input_height = model.input_height
    n_classes = model.n_classes

    x = inp
    pr = model.predict(np.array([x]))[0][0]

    # implementing threshold value
    if threshold is not None:
        assert 0 < threshold < 0.5, "Threshold value should be greater than 0 or less than 0.5"
        threshold_diff = 0.5 - threshold
        pr[:, 0] = pr[:, 0] - threshold_diff
        pr[:, 1] = pr[:, 1] + threshold_diff

    pr = pr.reshape((output_height, output_width, n_classes))
    pr_o = pr.copy()
    pr = pr.argmax(axis=2)

    seg_img = np.zeros((output_height, output_width, 3))

    if n_classes > 2:
        colors = class_colors
    else:
        colors = [(0, 0, 0), (255, 255, 255)]

    # # if threshold is given
    # if threshold is not None:
    #     # Convert values greater than threshold to 1
    #     pr[pr >= threshold] = 1

    for c in range(n_classes):
        seg_img[:, :, 0] += ((pr[:, :] == c) * (colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((pr[:, :] == c) * (colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((pr[:, :] == c) * (colors[c][2])).astype('uint8')

    # start: calculate score of the box
    box_score = 0
    if np.count_nonzero(seg_img[:, :, 0] > 0) > 0:
        box_score = np.mean(pr_o[:, :, 1][seg_img[:, :, 0] > 0])
    if not (box_score > 0):
        box_score = 0
    # end: calculate score of the box

    seg_img = cv2.resize(seg_img, (orininal_w, orininal_h), interpolation=cv2.INTER_NEAREST)

    if not out_fname is None:
        cv2.imwrite(out_fname, seg_img)

    return seg_img, pr_o, box_score


def two_stream_predict(model=None, inp=None, flow=None, out_fname=None, checkpoints_path=None, threshold=None):
    if model is None and (not checkpoints_path is None):
        model = model_from_checkpoint_path(checkpoints_path)

    assert (not inp is None)
    assert (flow is not None)
    assert ((type(inp) is np.ndarray) or isinstance(inp,
                                                    six.string_types)), "Inupt should be the CV image or the input file name"
    assert ((type(flow) is np.ndarray) or isinstance(flow,
                                                     six.string_types)), "Flow should be the CV image or the input file name"

    if isinstance(inp, six.string_types):
        inp = cv2.imread(inp)
    if isinstance(flow, six.string_types):
        flow = cv2.imread(flow)

    assert len(inp.shape) == 3, "Image should be h,w,3 "
    assert len(flow.shape) == 3, "Image should be h,w,3 "
    orininal_h = inp.shape[0]
    orininal_w = inp.shape[1]

    output_width = model.output_width
    output_height = model.output_height
    input_width = model.input_width
    input_height = model.input_height
    n_classes = model.n_classes

    x = get_image_arr(inp, input_width, input_height, odering=IMAGE_ORDERING)
    flow_x = get_image_arr(flow, input_width, input_height, odering=IMAGE_ORDERING)
    pr = model.predict([np.array([x]), np.array([flow_x])])[0]
    pr = pr.reshape((output_height, output_width, n_classes)).argmax(axis=2)

    seg_img = np.zeros((output_height, output_width, 3))

    if n_classes > 2:
        colors = class_colors
    else:
        colors = [(0, 0, 0), (255, 255, 255)]

    # # if threshold is given
    # if threshold is not None:
    #     # Convert values greater than threshold to 1
    #     pr[pr >= threshold] = 1

    for c in range(n_classes):
        seg_img[:, :, 0] += ((pr[:, :] == c) * (colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((pr[:, :] == c) * (colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((pr[:, :] == c) * (colors[c][2])).astype('uint8')

    seg_img = cv2.resize(seg_img, (orininal_w, orininal_h))

    if not out_fname is None:
        cv2.imwrite(out_fname, seg_img)

    return pr


def predict_with_segmentation_return(model=None, inp=None, out_fname=None, checkpoints_path=None, threshold=None):
    if model is None and (not checkpoints_path is None):
        model = model_from_checkpoint_path(checkpoints_path)

    assert (not inp is None)
    assert ((type(inp) is np.ndarray) or isinstance(inp,
                                                    six.string_types)), "Inupt should be the CV image or the input file name"

    if isinstance(inp, six.string_types):
        inp = cv2.imread(inp)

    assert len(inp.shape) == 3, "Image should be h,w,3 "
    orininal_h = inp.shape[0]
    orininal_w = inp.shape[1]

    output_width = model.output_width
    output_height = model.output_height
    input_width = model.input_width
    input_height = model.input_height
    n_classes = model.n_classes

    x = get_image_arr(inp, input_width, input_height, odering=IMAGE_ORDERING)
    pr = model.predict(np.array([x]))[0]
    pr = pr.reshape((output_height, output_width, n_classes)).argmax(axis=2)

    seg_img = np.zeros((output_height, output_width, 3))

    if n_classes > 2:
        colors = class_colors
    else:
        colors = [(0, 0, 0), (255, 255, 255)]

    # # if threshold is given
    # if threshold is not None:
    #     # Convert values greater than threshold to 1
    #     pr[pr >= threshold] = 1

    for c in range(n_classes):
        seg_img[:, :, 0] += ((pr[:, :] == c) * (colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((pr[:, :] == c) * (colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((pr[:, :] == c) * (colors[c][2])).astype('uint8')

    seg_img = cv2.resize(seg_img, (orininal_w, orininal_h))

    if not out_fname is None:
        cv2.imwrite(out_fname, seg_img)

    return pr, seg_img


def predict_with_segmentation_return_i3d(model=None, inp=None, out_fname=None, checkpoints_path=None, threshold=None):
    if model is None and (not checkpoints_path is None):
        model = model_from_checkpoint_path(checkpoints_path)

    assert (not inp is None)
    assert ((type(inp) is np.ndarray) or isinstance(inp,
                                                    six.string_types)), "Inupt should be the CV image or the input file name"

    orininal_h = 224
    orininal_w = 244

    output_width = model.output_width
    output_height = model.output_height
    input_width = model.input_width
    input_height = model.input_height
    n_classes = model.n_classes

    x = inp
    pr = model.predict(np.array([x]))[0]
    pr = pr.reshape((output_height, output_width, n_classes)).argmax(axis=2)

    seg_img = np.zeros((output_height, output_width, 3))

    if n_classes > 2:
        colors = class_colors
    else:
        colors = [(0, 0, 0), (255, 255, 255)]

    # # if threshold is given
    # if threshold is not None:
    #     # Convert values greater than threshold to 1
    #     pr[pr >= threshold] = 1

    for c in range(n_classes):
        seg_img[:, :, 0] += ((pr[:, :] == c) * (colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((pr[:, :] == c) * (colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((pr[:, :] == c) * (colors[c][2])).astype('uint8')

    seg_img = cv2.resize(seg_img, (orininal_w, orininal_h))

    if not out_fname is None:
        cv2.imwrite(out_fname, seg_img)

    return pr, seg_img


def predict_with_segmentation_and_probabilities_return(model=None, inp=None, out_fname=None, checkpoints_path=None,
                                                       threshold=None):
    if model is None and (not checkpoints_path is None):
        model = model_from_checkpoint_path(checkpoints_path)

    assert (not inp is None)
    assert ((type(inp) is np.ndarray) or isinstance(inp,
                                                    six.string_types)), "Inupt should be the CV image or the input file name"

    if isinstance(inp, six.string_types):
        inp = cv2.imread(inp)

    assert len(inp.shape) == 3, "Image should be h,w,3 "
    orininal_h = inp.shape[0]
    orininal_w = inp.shape[1]

    output_width = model.output_width
    output_height = model.output_height
    input_width = model.input_width
    input_height = model.input_height
    n_classes = model.n_classes

    x = get_image_arr(inp, input_width, input_height, odering=IMAGE_ORDERING)
    pr = model.predict(np.array([x]))[0]
    pr = pr.reshape((output_height, output_width, n_classes))
    pr_original = pr.copy()
    pr = pr.argmax(axis=2)

    seg_img = np.zeros((output_height, output_width, 3))

    if n_classes > 2:
        colors = class_colors
    else:
        colors = [(0, 0, 0), (255, 255, 255)]

    # # if threshold is given
    # if threshold is not None:
    #     # Convert values greater than threshold to 1
    #     pr[pr >= threshold] = 1

    for c in range(n_classes):
        seg_img[:, :, 0] += ((pr[:, :] == c) * (colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((pr[:, :] == c) * (colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((pr[:, :] == c) * (colors[c][2])).astype('uint8')

    seg_img = cv2.resize(seg_img, (orininal_w, orininal_h))

    if not out_fname is None:
        cv2.imwrite(out_fname, seg_img)

    return pr, seg_img, pr_original


def two_stream_predict_with_segmentation_return(model=None, inp=None, flow=None, out_fname=None, checkpoints_path=None,
                                                threshold=None):
    if model is None and (not checkpoints_path is None):
        model = model_from_checkpoint_path(checkpoints_path)

    assert (not inp is None)
    assert (flow is not None)
    assert ((type(inp) is np.ndarray) or isinstance(inp,
                                                    six.string_types)), "Inupt should be the CV image or the input file name"
    assert ((type(flow) is np.ndarray) or isinstance(inp,
                                                     six.string_types)), "Flow should be the CV image or the input file name"

    if isinstance(inp, six.string_types):
        inp = cv2.imread(inp)
    if isinstance(flow, six.string_types):
        flow = cv2.imread(flow)

    assert len(inp.shape) == 3, "Image should be h,w,3 "
    assert len(flow.shape) == 3, "Image should be h,w,3 "
    orininal_h = inp.shape[0]
    orininal_w = inp.shape[1]

    output_width = model.output_width
    output_height = model.output_height
    input_width = model.input_width
    input_height = model.input_height
    n_classes = model.n_classes

    x = get_image_arr(inp, input_width, input_height, odering=IMAGE_ORDERING)
    flow_x = get_image_arr(flow, input_width, input_height, odering=IMAGE_ORDERING)
    pr = model.predict([np.array([x]), np.array([flow_x])])[0]
    pr = pr.reshape((output_height, output_width, n_classes)).argmax(axis=2)

    seg_img = np.zeros((output_height, output_width, 3))

    if n_classes > 2:
        colors = class_colors
    else:
        colors = [(0, 0, 0), (255, 255, 255)]

    # # if threshold is given
    # if threshold is not None:
    #     # Convert values greater than threshold to 1
    #     pr[pr >= threshold] = 1

    for c in range(n_classes):
        seg_img[:, :, 0] += ((pr[:, :] == c) * (colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((pr[:, :] == c) * (colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((pr[:, :] == c) * (colors[c][2])).astype('uint8')

    seg_img = cv2.resize(seg_img, (orininal_w, orininal_h))

    if not out_fname is None:
        cv2.imwrite(out_fname, seg_img)

    return pr, seg_img


def predict_multiple(model=None, inps=None, inp_dir=None, out_dir=None, checkpoints_path=None):
    if model is None and (not checkpoints_path is None):
        model = model_from_checkpoint_path(checkpoints_path)

    if inps is None and (not inp_dir is None):
        inps = glob.glob(os.path.join(inp_dir, "*.jpg")) + glob.glob(os.path.join(inp_dir, "*.png")) + glob.glob(
            os.path.join(inp_dir, "*.jpeg"))

    assert type(inps) is list

    all_prs = []

    for i, inp in enumerate(tqdm(inps)):
        if out_dir is None:
            out_fname = None
        else:
            if isinstance(inp, six.string_types):
                out_fname = os.path.join(out_dir, os.path.basename(inp))
            else:
                out_fname = os.path.join(out_dir, str(i) + ".jpg")

        pr = predict(model, inp, out_fname)
        all_prs.append(pr)

    return all_prs


def search_flow(item_to_search_for, search_list):
    search_item = os.path.basename(item_to_search_for)
    for sl in search_list:
        if search_item == os.path.basename(sl):
            return sl
    return None


def two_stream_predict_multiple(model=None, inps=None, inp_dir=None, flows=None, flow_dir=None, out_dir=None,
                                checkpoints_path=None):
    if model is None and (not checkpoints_path is None):
        model = model_from_checkpoint_path(checkpoints_path)

    if inps is None and (not inp_dir is None):
        inps = glob.glob(os.path.join(inp_dir, "*.jpg")) + glob.glob(os.path.join(inp_dir, "*.png")) + glob.glob(
            os.path.join(inp_dir, "*.jpeg"))
    if flows is None and (not flow_dir is None):
        flows = glob.glob(os.path.join(flow_dir, "*.jpg")) + glob.glob(os.path.join(flow_dir, "*.png")) + glob.glob(
            os.path.join(flow_dir, "*.jpeg"))

    assert type(inps) is list
    assert type(flows) is list

    all_prs = []

    for i, inp in enumerate(tqdm(inps)):
        if out_dir is None:
            out_fname = None
        else:
            if isinstance(inp, six.string_types):
                out_fname = os.path.join(out_dir, os.path.basename(inp))
            else:
                out_fname = os.path.join(out_dir, str(i) + ".jpg")

        flow = search_flow(inp, flows)
        assert flow is not None, f"Flow wasn't found for image: {inp}"
        pr = two_stream_predict(model, inp, flow, out_fname)
        all_prs.append(pr)

    return all_prs


def evaluate(model=None, inp_inmges=None, annotations=None, checkpoints_path=None):
    assert False, "not implemented "

    ious = []
    for inp, ann in tqdm(zip(inp_images, annotations)):
        pr = predict(model, inp)
        gt = get_segmentation_arr(ann, model.n_classes, model.output_width, model.output_height)
        gt = gt.argmax(-1)
        iou = metrics.get_iou(gt, pr, model.n_classes)
        ious.append(iou)
    ious = np.array(ious)
    print("Class wise IoU ", np.mean(ious, axis=0))
    print("Total  IoU ", np.mean(ious))
