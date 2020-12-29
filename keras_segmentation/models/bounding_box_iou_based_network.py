from types import MethodType

from keras import layers
from keras.models import Model

from keras_segmentation.models.pspnet import pspnet_50
from ..predict import predict_bounding_box_iou_based_network
from ..train import train_bounding_box_iou_based_network


def bounding_box_iou_based_network(n_classes=2):
    pspnet_50_model = pspnet_50(n_classes=n_classes)

    pspnet_output = pspnet_50_model.layers[-4].output

    new_out = layers.Activation('relu')(pspnet_output)

    # adding fully connected layers
    new_out = layers.BatchNormalization(axis=-1)(new_out)
    new_out = layers.Flatten()(new_out)
    # new_out = layers.Dropout(0.25)(new_out)

    new_out = layers.Dense(1024, activation='relu')(new_out)
    # new_out = layers.BatchNormalization()(new_out)
    new_out = layers.Dropout(0.25)(new_out)

    new_out = layers.Dense(1024, activation='relu')(new_out)
    # new_out = layers.BatchNormalization()(new_out)
    new_out = layers.Dropout(0.25)(new_out)

    # (x, y, w, h)
    new_out = layers.Dense(4, activation='relu')(new_out)

    # defining model
    net = Model(inputs=pspnet_50_model.input, outputs=new_out)

    # adding extra information to the model
    net.n_classes = n_classes
    net.input_height = pspnet_50_model.input_height
    net.input_width = pspnet_50_model.input_width
    net.model_name = "bounding_box_iou_based_network"

    # extra model functions
    net.train = MethodType(train_bounding_box_iou_based_network, net)
    net.predict_boxes = MethodType(predict_bounding_box_iou_based_network, net)

    return net
