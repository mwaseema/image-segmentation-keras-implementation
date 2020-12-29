import tensorflow as tf
import numpy as np
from keras import backend as K

from keras_segmentation.data_utils.standalone_IoU_model_libs import get_iou_score


def iou_metric_wrapper(height, width, n_classes):
    def iou_metric_python(y_true: np.ndarray, y_pred: np.ndarray):
        batch_iou = []
        for batch in range(y_true.shape[0]):
            y_true_reshaped = y_true[batch].reshape((height, width, n_classes)).argmax(axis=2)
            y_pred_reshaped = y_pred[batch].reshape((height, width, n_classes)).argmax(axis=2)

            iou = get_iou_score(y_true_reshaped, y_pred_reshaped)
            batch_iou.append(iou)
        return np.array(batch_iou).astype(np.float64)

    def iou_metric(y_true, y_pred):
        iou = tf.py_func(iou_metric_python, inp=[y_true, y_pred], Tout=[tf.float64])
        iou = tf.cast(iou, dtype=tf.float32)
        iou = K.clip(iou, 0, 1)
        return iou

    return iou_metric
