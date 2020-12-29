import tensorflow as tf
from keras import backend as K

from keras_segmentation.custom_losses import smooth_l1_loss


def bounding_box_iou_based_network_loss(y_true, y_pred):
    pred_x1 = y_pred[:, 0]
    pred_y1 = y_pred[:, 1]
    pred_x2 = pred_x1 + y_pred[:, 2]
    pred_y2 = pred_y1 + y_pred[:, 3]

    true_x1 = y_true[:, 0]
    true_y1 = y_true[:, 1]
    true_x2 = true_x1 + y_true[:, 2]
    true_y2 = true_y1 + y_true[:, 3]

    intersection_x1 = tf.maximum(pred_x1, true_x1)
    intersection_y1 = tf.maximum(pred_y1, true_y1)
    intersection_x2 = tf.minimum(pred_x2, true_x2)
    intersection_y2 = tf.minimum(pred_y2, true_y2)

    intersection_area = tf.maximum(tf.zeros_like(intersection_x1),
                                   intersection_x2 - intersection_x1 + tf.ones_like(intersection_x1)) * tf.maximum(
        tf.zeros_like(intersection_y1), intersection_y2 - intersection_y1 + tf.ones_like(intersection_y1))

    pred_area = (pred_x2 - pred_x1 + tf.ones_like(pred_x1)) * (pred_y2 - pred_y1 + tf.ones_like(pred_y1))
    true_area = (true_x2 - true_x1 + tf.ones_like(true_x1)) * (true_y2 - true_y1 + tf.ones_like(true_y1))

    union_area = pred_area + true_area - intersection_area

    iou = intersection_area / tf.maximum(union_area, K.epsilon())

    iou = K.clip(iou, 0.0 + K.epsilon(), 1.0)

    iou_loss = -tf.log(iou)

    # convert loss (?,) to (1)
    iou_loss = tf.reduce_sum(iou_loss, axis=-1)

    l1_loss = smooth_l1_loss(y_true, y_pred)

    return iou_loss + l1_loss


def bounding_box_iou_based_network_metric(y_true, y_pred):
    pred_x1 = y_pred[:, 0]
    pred_y1 = y_pred[:, 1]
    pred_x2 = pred_x1 + y_pred[:, 2]
    pred_y2 = pred_y1 + y_pred[:, 3]

    true_x1 = y_true[:, 0]
    true_y1 = y_true[:, 1]
    true_x2 = true_x1 + y_true[:, 2]
    true_y2 = true_y1 + y_true[:, 3]

    intersection_x1 = tf.maximum(pred_x1, true_x1)
    intersection_y1 = tf.maximum(pred_y1, true_y1)
    intersection_x2 = tf.minimum(pred_x2, true_x2)
    intersection_y2 = tf.minimum(pred_y2, true_y2)

    intersection_area = tf.maximum(tf.zeros_like(intersection_x1),
                                   intersection_x2 - intersection_x1 + tf.ones_like(intersection_x1)) * tf.maximum(
        tf.zeros_like(intersection_y1), intersection_y2 - intersection_y1 + tf.ones_like(intersection_y1))

    pred_area = (pred_x2 - pred_x1 + tf.ones_like(pred_x1)) * (pred_y2 - pred_y1 + tf.ones_like(pred_y1))
    true_area = (true_x2 - true_x1 + tf.ones_like(true_x1)) * (true_y2 - true_y1 + tf.ones_like(true_y1))

    union_area = pred_area + true_area - intersection_area

    iou = intersection_area / tf.maximum(union_area, K.epsilon())

    return iou
