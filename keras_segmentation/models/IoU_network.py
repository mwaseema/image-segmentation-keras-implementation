from types import MethodType

from keras import layers
from keras.models import Model

from .pspnet import pspnet_50
from ..predict import predict_IoU_network, predict_IoU_network_plain
from ..train import train_IoU_network


def IoU_net_layers(net_in):
    net = layers.Flatten()(net_in)

    net = layers.Dense(units=1024, activation='relu')(net)
    net = layers.BatchNormalization()(net)
    net = layers.Dense(units=512, activation='relu')(net)
    net = layers.Dropout(0.1)(net)
    net = layers.Dense(units=1024, activation='relu')(net)
    net = layers.Dropout(0.1)(net)
    net = layers.Dense(units=1, activation='sigmoid', name='IoU_output')(net)

    return net


def IoU_network_pspnet50(n_classes, input_height=473, input_width=473):
    # resnet_layers = 50

    # inp = layers.Input(shape=(input_height, input_width, 3))
    # net = ResNet(inp, layers=resnet_layers)

    psp = pspnet_50(n_classes)
    inp = psp.input
    net = psp.layers[-4].output
    net = IoU_net_layers(net)

    # making new network which has two inputs and single merged output
    IoU_network = Model(inputs=inp, outputs=net)

    IoU_network.model_name = 'IoU_network_pspnet50'

    # IoU_network.output_width = model1.output_width
    # IoU_network.output_height = model1.output_height
    IoU_network.n_classes = n_classes
    IoU_network.input_height = input_height
    IoU_network.input_width = input_width

    IoU_network.train = MethodType(train_IoU_network, IoU_network)
    IoU_network.predict_segmentation_with_iou = MethodType(predict_IoU_network, IoU_network)
    IoU_network.predict_iou_plain = MethodType(predict_IoU_network_plain, IoU_network)
    # IoU_network.predict_segmentation_with_segmentation_return = MethodType(
    #     two_stream_predict_with_segmentation_return, IoU_network)
    # IoU_network.predict_multiple = MethodType(two_stream_predict_multiple, IoU_network)
    # IoU_network.evaluate_segmentation = MethodType(evaluate, IoU_network)

    return IoU_network
