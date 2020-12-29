import sys
from math import isnan, log
from types import MethodType
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
from keras import layers
from keras.models import Model

from ..data_utils.bounding_box_based_network_utils import number_of_bounding_boxes


def bounding_box_based_network_model(n_classes=2):
    from keras_segmentation.models.pspnet import pspnet_50
    from ..predict import predict_bounding_box_based_network
    from ..train import train_bounding_box_based_network

    pspnet_50_model = pspnet_50(n_classes=n_classes)

    pspnet_output = pspnet_50_model.layers[-4].output

    # adding fully connected layers
    new_out = layers.BatchNormalization(axis=-1)(pspnet_output)
    new_out = layers.Flatten()(new_out)
    # new_out = layers.Dropout(0.25)(new_out)

    new_out = layers.Dense(1024, activation='relu')(new_out)
    new_out = layers.BatchNormalization()(new_out)
    # new_out = layers.Dropout(0.25)(new_out)

    new_out = layers.Dense(1024, activation='relu')(new_out)
    new_out = layers.BatchNormalization()(new_out)
    new_out = layers.Dropout(0.25)(new_out)

    # One bounding box will have (confidence, x1, y1, w, h)
    new_out = layers.Dense(number_of_bounding_boxes * 5, activation='relu')(new_out)

    # defining model
    net = Model(inputs=pspnet_50_model.input, outputs=new_out)

    # adding extra information to the model
    net.n_classes = n_classes
    net.input_height = pspnet_50_model.input_height
    net.input_width = pspnet_50_model.input_width
    net.model_name = "bounding_box_based_network_model"

    # extra model functions
    net.train = MethodType(train_bounding_box_based_network, net)
    net.predict_boxes = MethodType(predict_bounding_box_based_network, net)

    return net
