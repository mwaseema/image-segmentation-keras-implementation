from .pspnet import pspnet_101, pspnet_50, resnet50_pspnet
from keras.models import Model
from keras.layers.merge import concatenate
from keras.layers.merge import average
from keras.layers import Conv1D
from types import MethodType

from ..train import train_two_stream
from ..predict import two_stream_predict, two_stream_predict_multiple, evaluate, \
    two_stream_predict_with_segmentation_return


def two_stream_pspnet_101(n_classes, input_height=473, input_width=473):
    # making two models from current pspnet_101 network
    model1 = pspnet_101(n_classes=n_classes, input_height=input_height, input_width=input_width)
    model2 = pspnet_101(n_classes=n_classes, input_height=input_height, input_width=input_width)

    # change layer names for both models
    for layer in model1.layers:
        layer.name = f"1_{layer.name}"

    for layer in model2.layers:
        layer.name = f"2_{layer.name}"

    # merge output of both previous models
    merged = concatenate([model1.output, model2.output])
    merged = Conv1D(2, kernel_size=1, activation='relu')(merged)

    # making new network which has two inputs and single merged output
    two_stream_network = Model(inputs=[model1.input, model2.input], outputs=merged)

    two_stream_network.model_name = 'two_stream_pspnet_101'

    two_stream_network.output_width = model1.output_width
    two_stream_network.output_height = model1.output_height
    two_stream_network.n_classes = model1.n_classes
    two_stream_network.input_height = model1.input_height
    two_stream_network.input_width = model1.input_width

    two_stream_network.train = MethodType(train_two_stream, two_stream_network)
    two_stream_network.predict_segmentation = MethodType(two_stream_predict, two_stream_network)
    two_stream_network.predict_segmentation_with_segmentation_return = MethodType(
        two_stream_predict_with_segmentation_return, two_stream_network)
    two_stream_network.predict_multiple = MethodType(two_stream_predict_multiple, two_stream_network)
    two_stream_network.evaluate_segmentation = MethodType(evaluate, two_stream_network)

    return two_stream_network


def two_stream_pspnet_101_average_merge(n_classes, input_height=473, input_width=473):
    # making two models from current pspnet_101 network
    model1 = pspnet_101(n_classes=n_classes, input_height=input_height, input_width=input_width)
    model2 = pspnet_101(n_classes=n_classes, input_height=input_height, input_width=input_width)

    # change layer names for both models
    for layer in model1.layers:
        layer.name = f"1_{layer.name}"

    for layer in model2.layers:
        layer.name = f"2_{layer.name}"

    # merge output of both previous models
    merged = average([model1.output, model2.output])

    # making new network which has two inputs and single merged output
    two_stream_network = Model(inputs=[model1.input, model2.input], outputs=merged)

    two_stream_network.model_name = 'two_stream_pspnet_101_average_merge'

    two_stream_network.output_width = model1.output_width
    two_stream_network.output_height = model1.output_height
    two_stream_network.n_classes = model1.n_classes
    two_stream_network.input_height = model1.input_height
    two_stream_network.input_width = model1.input_width

    two_stream_network.train = MethodType(train_two_stream, two_stream_network)
    two_stream_network.predict_segmentation = MethodType(two_stream_predict, two_stream_network)
    two_stream_network.predict_segmentation_with_segmentation_return = MethodType(
        two_stream_predict_with_segmentation_return, two_stream_network)
    two_stream_network.predict_multiple = MethodType(two_stream_predict_multiple, two_stream_network)
    two_stream_network.evaluate_segmentation = MethodType(evaluate, two_stream_network)

    return two_stream_network


def two_stream_pspnet_50(n_classes, input_height=473, input_width=473):
    # making two models from current pspnet_101 network
    model1 = pspnet_50(n_classes=n_classes, input_height=input_height, input_width=input_width)
    model2 = pspnet_50(n_classes=n_classes, input_height=input_height, input_width=input_width)

    # change layer names for both models
    for layer in model1.layers:
        layer.name = f"1_{layer.name}"

    for layer in model2.layers:
        layer.name = f"2_{layer.name}"

    # merge output of both previous models
    merged = concatenate([model1.output, model2.output])
    merged = Conv1D(2, kernel_size=1, activation='relu')(merged)

    # making new network which has two inputs and single merged output
    two_stream_network = Model(inputs=[model1.input, model2.input], outputs=merged)

    two_stream_network.model_name = 'two_stream_pspnet_50'

    two_stream_network.output_width = model1.output_width
    two_stream_network.output_height = model1.output_height
    two_stream_network.n_classes = model1.n_classes
    two_stream_network.input_height = model1.input_height
    two_stream_network.input_width = model1.input_width

    two_stream_network.train = MethodType(train_two_stream, two_stream_network)
    two_stream_network.predict_segmentation = MethodType(two_stream_predict, two_stream_network)
    two_stream_network.predict_segmentation_with_segmentation_return = MethodType(
        two_stream_predict_with_segmentation_return, two_stream_network)
    two_stream_network.predict_multiple = MethodType(two_stream_predict_multiple, two_stream_network)
    two_stream_network.evaluate_segmentation = MethodType(evaluate, two_stream_network)

    return two_stream_network


def two_stream_pspnet_50_average_merge(n_classes, input_height=473, input_width=473):
    # making two models from current pspnet_101 network
    model1 = pspnet_50(n_classes=n_classes, input_height=input_height, input_width=input_width)
    model2 = pspnet_50(n_classes=n_classes, input_height=input_height, input_width=input_width)

    # change layer names for both models
    for layer in model1.layers:
        layer.name = f"1_{layer.name}"

    for layer in model2.layers:
        layer.name = f"2_{layer.name}"

    # merge output of both previous models
    merged = average([model1.output, model2.output])

    # making new network which has two inputs and single merged output
    two_stream_network = Model(inputs=[model1.input, model2.input], outputs=merged)

    two_stream_network.model_name = 'two_stream_pspnet_50_average_merge'

    two_stream_network.output_width = model1.output_width
    two_stream_network.output_height = model1.output_height
    two_stream_network.n_classes = model1.n_classes
    two_stream_network.input_height = model1.input_height
    two_stream_network.input_width = model1.input_width

    two_stream_network.train = MethodType(train_two_stream, two_stream_network)
    two_stream_network.predict_segmentation = MethodType(two_stream_predict, two_stream_network)
    two_stream_network.predict_segmentation_with_segmentation_return = MethodType(
        two_stream_predict_with_segmentation_return, two_stream_network)
    two_stream_network.predict_multiple = MethodType(two_stream_predict_multiple, two_stream_network)
    two_stream_network.evaluate_segmentation = MethodType(evaluate, two_stream_network)

    return two_stream_network


def two_stream_resnet50_pspnet(n_classes, input_height=384, input_width=576):
    # making two models from current pspnet_101 network
    model1 = resnet50_pspnet(n_classes=n_classes, input_height=input_height, input_width=input_width)
    model2 = resnet50_pspnet(n_classes=n_classes, input_height=input_height, input_width=input_width)

    # change layer names for both models
    for layer in model1.layers:
        layer.name = f"1_{layer.name}"

    for layer in model2.layers:
        layer.name = f"2_{layer.name}"

    # merge output of both previous models
    merged = concatenate([model1.output, model2.output])
    merged = Conv1D(2, kernel_size=1, activation='relu')(merged)

    # making new network which has two inputs and single merged output
    two_stream_network = Model(inputs=[model1.input, model2.input], outputs=merged)

    two_stream_network.model_name = 'two_stream_resnet50_pspnet'

    two_stream_network.output_width = model1.output_width
    two_stream_network.output_height = model1.output_height
    two_stream_network.n_classes = model1.n_classes
    two_stream_network.input_height = model1.input_height
    two_stream_network.input_width = model1.input_width

    two_stream_network.train = MethodType(train_two_stream, two_stream_network)
    two_stream_network.predict_segmentation = MethodType(two_stream_predict, two_stream_network)
    two_stream_network.predict_segmentation_with_segmentation_return = MethodType(
        two_stream_predict_with_segmentation_return, two_stream_network)
    two_stream_network.predict_multiple = MethodType(two_stream_predict_multiple, two_stream_network)
    two_stream_network.evaluate_segmentation = MethodType(evaluate, two_stream_network)

    return two_stream_network


def two_stream_resnet50_pspnet_average_merge(n_classes, input_height=384, input_width=576):
    # making two models from current pspnet_101 network
    model1 = resnet50_pspnet(n_classes=n_classes, input_height=input_height, input_width=input_width)
    model2 = resnet50_pspnet(n_classes=n_classes, input_height=input_height, input_width=input_width)

    # change layer names for both models
    for layer in model1.layers:
        layer.name = f"1_{layer.name}"

    for layer in model2.layers:
        layer.name = f"2_{layer.name}"

    # merge output of both previous models
    merged = average([model1.output, model2.output])

    # making new network which has two inputs and single merged output
    two_stream_network = Model(inputs=[model1.input, model2.input], outputs=merged)

    two_stream_network.model_name = 'two_stream_resnet50_pspnet_average_merge'

    two_stream_network.output_width = model1.output_width
    two_stream_network.output_height = model1.output_height
    two_stream_network.n_classes = model1.n_classes
    two_stream_network.input_height = model1.input_height
    two_stream_network.input_width = model1.input_width

    two_stream_network.train = MethodType(train_two_stream, two_stream_network)
    two_stream_network.predict_segmentation = MethodType(two_stream_predict, two_stream_network)
    two_stream_network.predict_segmentation_with_segmentation_return = MethodType(
        two_stream_predict_with_segmentation_return, two_stream_network)
    two_stream_network.predict_multiple = MethodType(two_stream_predict_multiple, two_stream_network)
    two_stream_network.evaluate_segmentation = MethodType(evaluate, two_stream_network)

    return two_stream_network
