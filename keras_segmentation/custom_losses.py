"""
Define our custom loss function.
"""
import cv2
from keras import backend as K
import tensorflow as tf
from scipy.optimize import linear_sum_assignment
import skimage.measure
import numpy as np
from math import exp, isnan, pow


# import dill


def binary_focal_loss(gamma=2., alpha=.25):
    """
    Binary form of focal loss.

      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)

      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.

    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)

    """

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        epsilon = K.epsilon()
        # clip to prevent NaN's and Inf's
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
               - K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return binary_focal_loss_fixed


def categorical_focal_loss(gamma=2., alpha=.25):
    """
    Softmax version of focal loss.

           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1

      where m = number of classes, c = class and o = observation

    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy
      gamma -- focusing parameter for modulating factor (1-p)

    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper

    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy

    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """

    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """

        # Scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        # Sum the losses in mini_batch
        return K.sum(loss, axis=1)

    return categorical_focal_loss_fixed


def categorical_focal_loss_with_iou(gamma=2., alpha=.25, model=None):
    """
    Softmax version of focal loss.

           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1

      where m = number of classes, c = class and o = observation

    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy
      gamma -- focusing parameter for modulating factor (1-p)

    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper

    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy

    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """

    def bb_intersection_over_union(boxA, boxB):
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

    def get_region_props(image: np.ndarray):
        im = image.copy()

        if len(im.shape) == 3 and im.shape[2] == 3:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        label_image = skimage.measure.label(im)
        region_props = skimage.measure.regionprops(label_image)
        return region_props

    def get_iou_score(y_true_reshaped, y_pred_reshaped):
        props_im = get_region_props(y_pred_reshaped)
        props_gt = get_region_props(y_true_reshaped)

        IOU_bbx_mul = np.zeros((props_gt.__len__(), props_im.__len__()))

        # returning -1 only if there is no gt bbox found
        if len(props_gt) == 0:
            return -1

        for g_b in range(0, props_gt.__len__()):
            for p_b in range(0, props_im.__len__()):
                IOU_bbx_mul[g_b, p_b] = bb_intersection_over_union(props_gt[g_b].bbox, props_im[p_b].bbox)

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

    def compute_iou(y_true, y_pred):
        IoUs = []
        # iterating over batch
        for i in range(y_true.shape[0]):
            y_true_single = y_true[i, :, :].reshape(model.output_width, model.output_height, 2).argmax(axis=2)
            y_pred_single = y_pred[i, :, :].reshape(model.output_width, model.output_height, 2).argmax(axis=2)

            IoU = get_iou_score(y_true_single, y_pred_single)
            if isnan(IoU):
                IoU = 0

            # only add IoU score if it is not -1
            if IoU != -1:
                IoUs.append(IoU)

        if len(IoUs) > 0:
            return float(np.mean(IoUs))
        else:
            return -1

    def loss_from_iou(y_true, y_pred):
        average_iou = compute_iou(y_true, y_pred)

        if average_iou >= 0.8 or average_iou == -1:
            loss = 0
        else:
            loss = exp(1 - average_iou)

        return float(loss)

    def get_center_of_bbox(mask_arr):
        mask_arr_region_props = get_region_props(mask_arr)

        centers = []
        for reg_prop in mask_arr_region_props:
            x1, y1, x2, y2 = reg_prop.bbox
            X = int(np.average([x1, x2]))
            Y = int(np.average([y1, y2]))

            centers.append({'x': X, 'y': Y})
        return centers

    def loss_for_difference_in_center(y_true, y_pred):
        total_distance = 0

        # iterating over batch
        for i in range(y_true.shape[0]):
            y_true_single = y_true[i, :, :].reshape(model.output_width, model.output_height, 2).argmax(axis=2)
            y_pred_single = y_pred[i, :, :].reshape(model.output_width, model.output_height, 2).argmax(axis=2)

            y_true_centers = get_center_of_bbox(y_true_single)
            y_pred_centers = get_center_of_bbox(y_pred_single)

            # Continue loop for next iteration if bbox for prediction or ground truth isn't found
            if len(y_true_centers) == 0 or len(y_pred_centers) == 0:
                continue

            center_losses = np.zeros((len(y_true_centers), len(y_pred_centers)), dtype=np.float)

            for i in range(len(y_true_centers)):
                y_true_x = y_true_centers[i]['x']
                y_true_y = y_true_centers[i]['y']

                for j in range(len(y_pred_centers)):
                    y_pred_x = y_pred_centers[j]['x']
                    y_pred_y = y_pred_centers[j]['y']

                    center_losses[i, j] = pow(y_true_x - y_pred_x, 2) + pow(y_true_y - y_pred_y, 2)

            for i in range(center_losses.shape[0]):
                total_distance += np.min(center_losses[i, :])

        total_distance *= 0.5
        return np.float(total_distance)

    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """

        # added IoU calculation code at start in order to decrease CPU, GPU jump
        iou_loss_val = tf.py_func(loss_from_iou, inp=[y_true, y_pred], Tout=tf.double)
        iou_loss_val.set_shape((1,))
        iou_val = tf.dtypes.cast(iou_loss_val, dtype=tf.float32)

        loss_for_diff = tf.py_func(loss_for_difference_in_center, inp=[y_true, y_pred], Tout=tf.double)
        loss_for_diff.set_shape((1,))
        loss_for_difference = tf.dtypes.cast(loss_for_diff, dtype=tf.float32)

        # Scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        # Sum the losses in mini_batch
        # return K.sum(loss, axis=1)
        f_loss = K.sum(loss, axis=1)

        return f_loss + iou_val + loss_for_difference

    return categorical_focal_loss_fixed


def iou_loss_GPU(y_true, y_pred):
    # iou loss for bounding box prediction
    # input must be as [x1, y1, x2, y2]

    # AOG = Area of Groundtruth box
    AoG = K.abs(K.transpose(y_true)[2] - K.transpose(y_true)[0] + 1) * K.abs(
        K.transpose(y_true)[3] - K.transpose(y_true)[1] + 1)

    # AOP = Area of Predicted box
    AoP = K.abs(K.transpose(y_pred)[2] - K.transpose(y_pred)[0] + 1) * K.abs(
        K.transpose(y_pred)[3] - K.transpose(y_pred)[1] + 1)

    # overlaps are the co-ordinates of intersection box
    overlap_0 = K.maximum(K.transpose(y_true)[0], K.transpose(y_pred)[0])
    overlap_1 = K.maximum(K.transpose(y_true)[1], K.transpose(y_pred)[1])
    overlap_2 = K.minimum(K.transpose(y_true)[2], K.transpose(y_pred)[2])
    overlap_3 = K.minimum(K.transpose(y_true)[3], K.transpose(y_pred)[3])

    # intersection area
    intersection = (overlap_2 - overlap_0 + 1) * (overlap_3 - overlap_1 + 1)

    # area of union of both boxes
    union = AoG + AoP - intersection

    # iou calculation
    iou = intersection / union

    # bounding values of iou to (0,1)
    iou = K.clip(iou, 0.0 + K.epsilon(), 1.0 - K.epsilon())

    # loss for the iou value
    iou_loss = -K.log(iou)

    return iou_loss


def smooth_l1_loss(y_true, y_pred):
    return tf.losses.huber_loss(y_true, y_pred)


if __name__ == '__main__':
    # Test serialization of nested functions
    # bin_inner = dill.loads(dill.dumps(binary_focal_loss(gamma=2., alpha=.25)))
    # print(bin_inner)
    #
    # cat_inner = dill.loads(dill.dumps(categorical_focal_loss(gamma=2., alpha=.25)))
    # print(cat_inner)
    pass
