# This code is proveded by Vladkryvoruchko and small modifications done by me .


from math import ceil
from keras import layers
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import BatchNormalization, Activation, Input, Dropout, ZeroPadding2D, Lambda, Conv3D, MaxPool3D
from keras.layers.merge import Concatenate, Add
from keras import backend as K
from keras.models import Model
from keras.optimizers import SGD

from keras.backend import tf as ktf
import tensorflow as tf

from .config import IMAGE_ORDERING
from .model_utils import get_segmentation_model, resize_image, get_segmentation_model_with_weighted_output, \
    get_segmentation_model_i3d_inception, get_temporal_segmentation_model_with_weighted_output
from types import MethodType

learning_rate = 1e-3  # Layer specific learning rate
# Weight decay not implemented


def BN(name=""):
    return BatchNormalization(momentum=0.95, name=name, epsilon=1e-5)


class Interp(layers.Layer):

    def __init__(self, new_size, **kwargs):
        self.new_size = new_size
        super(Interp, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Interp, self).build(input_shape)

    def call(self, inputs, **kwargs):
        new_height, new_width = self.new_size
        resized = ktf.image.resize_images(inputs, [new_height, new_width],
                                          align_corners=True)
        return resized

    def compute_output_shape(self, input_shape):
        return tuple([None, self.new_size[0], self.new_size[1], input_shape[3]])

    def get_config(self):
        config = super(Interp, self).get_config()
        config['new_size'] = self.new_size
        return config


# def Interp(x, shape):
#    new_height, new_width = shape
#    resized = ktf.image.resize_images(x, [new_height, new_width],
#                                      align_corners=True)
#    return resized


def residual_conv(prev, level, pad=1, lvl=1, sub_lvl=1, modify_stride=False):
    lvl = str(lvl)
    sub_lvl = str(sub_lvl)
    names = ["conv" + lvl + "_" + sub_lvl + "_1x1_reduce",
             "conv" + lvl + "_" + sub_lvl + "_1x1_reduce_bn",
             "conv" + lvl + "_" + sub_lvl + "_3x3",
             "conv" + lvl + "_" + sub_lvl + "_3x3_bn",
             "conv" + lvl + "_" + sub_lvl + "_1x1_increase",
             "conv" + lvl + "_" + sub_lvl + "_1x1_increase_bn"]
    if modify_stride is False:
        prev = Conv2D(64 * level, (1, 1), strides=(1, 1), name=names[0],
                      use_bias=False)(prev)
    elif modify_stride is True:
        prev = Conv2D(64 * level, (1, 1), strides=(2, 2), name=names[0],
                      use_bias=False)(prev)

    prev = BN(name=names[1])(prev)
    prev = Activation('relu')(prev)

    prev = ZeroPadding2D(padding=(pad, pad))(prev)
    prev = Conv2D(64 * level, (3, 3), strides=(1, 1), dilation_rate=pad,
                  name=names[2], use_bias=False)(prev)

    prev = BN(name=names[3])(prev)
    prev = Activation('relu')(prev)
    prev = Conv2D(256 * level, (1, 1), strides=(1, 1), name=names[4],
                  use_bias=False)(prev)
    prev = BN(name=names[5])(prev)
    return prev


def short_convolution_branch(prev, level, lvl=1, sub_lvl=1, modify_stride=False):
    lvl = str(lvl)
    sub_lvl = str(sub_lvl)
    names = ["conv" + lvl + "_" + sub_lvl + "_1x1_proj",
             "conv" + lvl + "_" + sub_lvl + "_1x1_proj_bn"]

    if modify_stride is False:
        prev = Conv2D(256 * level, (1, 1), strides=(1, 1), name=names[0],
                      use_bias=False)(prev)
    elif modify_stride is True:
        prev = Conv2D(256 * level, (1, 1), strides=(2, 2), name=names[0],
                      use_bias=False)(prev)

    prev = BN(name=names[1])(prev)
    return prev


def empty_branch(prev):
    return prev


def residual_short(prev_layer, level, pad=1, lvl=1, sub_lvl=1, modify_stride=False):
    prev_layer = Activation('relu')(prev_layer)
    block_1 = residual_conv(prev_layer, level,
                            pad=pad, lvl=lvl, sub_lvl=sub_lvl,
                            modify_stride=modify_stride)

    block_2 = short_convolution_branch(prev_layer, level,
                                       lvl=lvl, sub_lvl=sub_lvl,
                                       modify_stride=modify_stride)
    added = Add()([block_1, block_2])
    return added


def residual_empty(prev_layer, level, pad=1, lvl=1, sub_lvl=1):
    prev_layer = Activation('relu')(prev_layer)

    block_1 = residual_conv(prev_layer, level, pad=pad,
                            lvl=lvl, sub_lvl=sub_lvl)
    block_2 = empty_branch(prev_layer)
    added = Add()([block_1, block_2])
    return added


def ResNet(inp, layers):
    # Names for the first couple layers of model
    names = ["conv1_1_3x3_s2",
             "conv1_1_3x3_s2_bn",
             "conv1_2_3x3",
             "conv1_2_3x3_bn",
             "conv1_3_3x3",
             "conv1_3_3x3_bn"]

    # Short branch(only start of network)

    cnv1 = Conv2D(64, (3, 3), strides=(2, 2), padding='same', name=names[0],
                  use_bias=False)(inp)  # "conv1_1_3x3_s2"
    bn1 = BN(name=names[1])(cnv1)  # "conv1_1_3x3_s2/bn"
    relu1 = Activation('relu')(bn1)  # "conv1_1_3x3_s2/relu"

    cnv1 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', name=names[2],
                  use_bias=False)(relu1)  # "conv1_2_3x3"
    bn1 = BN(name=names[3])(cnv1)  # "conv1_2_3x3/bn"
    relu1 = Activation('relu')(bn1)  # "conv1_2_3x3/relu"

    cnv1 = Conv2D(128, (3, 3), strides=(1, 1), padding='same', name=names[4],
                  use_bias=False)(relu1)  # "conv1_3_3x3"
    bn1 = BN(name=names[5])(cnv1)  # "conv1_3_3x3/bn"
    relu1 = Activation('relu')(bn1)  # "conv1_3_3x3/relu"

    res = MaxPooling2D(pool_size=(3, 3), padding='same',
                       strides=(2, 2))(relu1)  # "pool1_3x3_s2"

    # ---Residual layers(body of network)

    """
    Modify_stride --Used only once in first 3_1 convolutions block.
    changes stride of first convolution from 1 -> 2
    """

    # 2_1- 2_3
    res = residual_short(res, 1, pad=1, lvl=2, sub_lvl=1)
    for i in range(2):
        res = residual_empty(res, 1, pad=1, lvl=2, sub_lvl=i + 2)

    # 3_1 - 3_3
    res = residual_short(res, 2, pad=1, lvl=3, sub_lvl=1, modify_stride=True)
    for i in range(3):
        res = residual_empty(res, 2, pad=1, lvl=3, sub_lvl=i + 2)
    if layers is 50:
        # 4_1 - 4_6
        res = residual_short(res, 4, pad=2, lvl=4, sub_lvl=1)
        for i in range(5):
            res = residual_empty(res, 4, pad=2, lvl=4, sub_lvl=i + 2)
    elif layers is 101:
        # 4_1 - 4_23
        res = residual_short(res, 4, pad=2, lvl=4, sub_lvl=1)
        for i in range(22):
            res = residual_empty(res, 4, pad=2, lvl=4, sub_lvl=i + 2)
    else:
        print("This ResNet is not implemented")

    # 5_1 - 5_3
    res = residual_short(res, 8, pad=4, lvl=5, sub_lvl=1)
    for i in range(2):
        res = residual_empty(res, 8, pad=4, lvl=5, sub_lvl=i + 2)

    res = Activation('relu')(res)
    return res


def ResNet_with_different_level_features(inp, layers):
    # Names for the first couple layers of model
    names = ["conv1_1_3x3_s2",
             "conv1_1_3x3_s2_bn",
             "conv1_2_3x3",
             "conv1_2_3x3_bn",
             "conv1_3_3x3",
             "conv1_3_3x3_bn"]

    layers_to_merge = []

    # Short branch(only start of network)

    cnv1 = Conv2D(64, (3, 3), strides=(2, 2), padding='same', name=names[0], use_bias=False)(inp)  # "conv1_1_3x3_s2"
    bn1 = BN(name=names[1])(cnv1)  # "conv1_1_3x3_s2/bn"
    relu1 = Activation('relu')(bn1)  # "conv1_1_3x3_s2/relu"

    cnv1 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', name=names[2], use_bias=False)(relu1)  # "conv1_2_3x3"
    bn1 = BN(name=names[3])(cnv1)  # "conv1_2_3x3/bn"
    relu1 = Activation('relu')(bn1)  # "conv1_2_3x3/relu"

    cnv1 = Conv2D(128, (3, 3), strides=(1, 1), padding='same', name=names[4], use_bias=False)(relu1)  # "conv1_3_3x3"
    bn1 = BN(name=names[5])(cnv1)  # "conv1_3_3x3/bn"
    relu1 = Activation('relu')(bn1)  # "conv1_3_3x3/relu"

    res = MaxPooling2D(pool_size=(3, 3), padding='same', strides=(2, 2))(relu1)  # "pool1_3x3_s2"

    layers_to_merge.append(res)

    # ---Residual layers(body of network)

    """
    Modify_stride --Used only once in first 3_1 convolutions block.
    changes stride of first convolution from 1 -> 2
    """

    # 2_1- 2_3
    res = residual_short(res, 1, pad=1, lvl=2, sub_lvl=1)
    for i in range(2):
        res = residual_empty(res, 1, pad=1, lvl=2, sub_lvl=i + 2)

    layers_to_merge.append(res)

    # 3_1 - 3_3
    res = residual_short(res, 2, pad=1, lvl=3, sub_lvl=1, modify_stride=True)
    for i in range(3):
        res = residual_empty(res, 2, pad=1, lvl=3, sub_lvl=i + 2)

    layers_to_merge.append(res)

    if layers is 50:
        # 4_1 - 4_6
        res = residual_short(res, 4, pad=2, lvl=4, sub_lvl=1)
        for i in range(5):
            res = residual_empty(res, 4, pad=2, lvl=4, sub_lvl=i + 2)
    elif layers is 101:
        # 4_1 - 4_23
        res = residual_short(res, 4, pad=2, lvl=4, sub_lvl=1)
        for i in range(22):
            res = residual_empty(res, 4, pad=2, lvl=4, sub_lvl=i + 2)
    else:
        print("This ResNet is not implemented")

    layers_to_merge.append(res)

    # 5_1 - 5_3
    res = residual_short(res, 8, pad=4, lvl=5, sub_lvl=1)
    for i in range(2):
        res = residual_empty(res, 8, pad=4, lvl=5, sub_lvl=i + 2)

    res = Activation('relu')(res)

    # applying activations to all the layers
    layers_to_merge[0] = Activation('relu')(layers_to_merge[0])
    layers_to_merge[1] = Activation('relu')(layers_to_merge[1])
    layers_to_merge[2] = Activation('relu')(layers_to_merge[2])
    layers_to_merge[3] = Activation('relu')(layers_to_merge[3])

    # resizing first two layers to match all the remaining
    layers_to_merge[0] = Interp([60, 60])(layers_to_merge[0])
    layers_to_merge[1] = Interp([60, 60])(layers_to_merge[1])

    layers_to_merge[2] = Conv2D(128, kernel_size=(1, 1))(layers_to_merge[2])

    # layers_to_merge[0] = Conv2D(filters=128, kernel_size=(3, 3), padding='same')(layers_to_merge[0])
    # layers_to_merge[1] = Conv2D(filters=128, kernel_size=(3, 3), padding='same')(layers_to_merge[1])
    # layers_to_merge[2] = Conv2D(filters=128, kernel_size=(3, 3), padding='same')(layers_to_merge[2])
    # layers_to_merge[3] = Conv2D(filters=128, kernel_size=(3, 3), padding='same')(layers_to_merge[3])
    #
    # layers_to_merge[0] = BN(name='diff_layer_merging_bn_0')(layers_to_merge[0])
    # layers_to_merge[1] = BN(name='diff_layer_merging_bn_1')(layers_to_merge[1])
    # layers_to_merge[2] = BN(name='diff_layer_merging_bn_2')(layers_to_merge[2])
    # layers_to_merge[3] = BN(name='diff_layer_merging_bn_3')(layers_to_merge[3])
    #
    # layers_to_merge[0] = Activation('relu')(layers_to_merge[0])
    # layers_to_merge[1] = Activation('relu')(layers_to_merge[1])
    # layers_to_merge[2] = Activation('relu')(layers_to_merge[2])
    # layers_to_merge[3] = Activation('relu')(layers_to_merge[3])

    layers_to_merge.append(res)
    final = Concatenate()(layers_to_merge)

    final = Conv2D(filters=2048, kernel_size=(1, 1))(final)

    return final


def ResNet_temporal_with_different_level_features(inp, layers):
    # Names for the first couple layers of model
    names = ["conv1_1_3x3_s2",
             "conv1_1_3x3_s2_bn",
             "conv1_2_3x3",
             "conv1_2_3x3_bn",
             "conv1_3_3x3",
             "conv1_3_3x3_bn"]

    layers_to_merge = []

    # Short branch(only start of network)

    cnv1 = Conv3D(64, (3, 3, 3), strides=(1, 2, 2), padding='same', name=names[0], use_bias=False)(
        inp)  # "conv1_1_3x3_s2"
    bn1 = BN(name=names[1])(cnv1)  # "conv1_1_3x3_s2/bn"
    relu1 = Activation('relu')(bn1)  # "conv1_1_3x3_s2/relu"

    cnv1 = Conv3D(64, (3, 3, 3), strides=(1, 1, 1), padding='same', name=names[2], use_bias=False)(
        relu1)  # "conv1_2_3x3"
    bn1 = BN(name=names[3])(cnv1)  # "conv1_2_3x3/bn"
    relu1 = Activation('relu')(bn1)  # "conv1_2_3x3/relu"

    cnv1 = Conv3D(128, (3, 3, 3), strides=(1, 1, 1), padding='same', name=names[4], use_bias=False)(
        relu1)  # "conv1_3_3x3"
    bn1 = BN(name=names[5])(cnv1)  # "conv1_3_3x3/bn"
    relu1 = Activation('relu')(bn1)  # "conv1_3_3x3/relu"

    res = MaxPool3D(pool_size=(3, 3, 3), padding='same', strides=(2, 2, 2))(relu1)  # "pool1_3x3_s2"
    res = Lambda(lambda x: K.mean(x, axis=1, keepdims=False))(res)

    layers_to_merge.append(res)

    # ---Residual layers(body of network)

    """
    Modify_stride --Used only once in first 3_1 convolutions block.
    changes stride of first convolution from 1 -> 2
    """

    # 2_1- 2_3
    res = residual_short(res, 1, pad=1, lvl=2, sub_lvl=1)
    for i in range(2):
        res = residual_empty(res, 1, pad=1, lvl=2, sub_lvl=i + 2)

    layers_to_merge.append(res)

    # 3_1 - 3_3
    res = residual_short(res, 2, pad=1, lvl=3, sub_lvl=1, modify_stride=True)
    for i in range(3):
        res = residual_empty(res, 2, pad=1, lvl=3, sub_lvl=i + 2)

    layers_to_merge.append(res)

    if layers is 50:
        # 4_1 - 4_6
        res = residual_short(res, 4, pad=2, lvl=4, sub_lvl=1)
        for i in range(5):
            res = residual_empty(res, 4, pad=2, lvl=4, sub_lvl=i + 2)
    elif layers is 101:
        # 4_1 - 4_23
        res = residual_short(res, 4, pad=2, lvl=4, sub_lvl=1)
        for i in range(22):
            res = residual_empty(res, 4, pad=2, lvl=4, sub_lvl=i + 2)
    else:
        print("This ResNet is not implemented")

    layers_to_merge.append(res)

    # 5_1 - 5_3
    res = residual_short(res, 8, pad=4, lvl=5, sub_lvl=1)
    for i in range(2):
        res = residual_empty(res, 8, pad=4, lvl=5, sub_lvl=i + 2)

    res = Activation('relu')(res)

    # applying activations to all the layers
    layers_to_merge[0] = Activation('relu')(layers_to_merge[0])
    layers_to_merge[1] = Activation('relu')(layers_to_merge[1])
    layers_to_merge[2] = Activation('relu')(layers_to_merge[2])
    layers_to_merge[3] = Activation('relu')(layers_to_merge[3])

    # resizing first two layers to match all the remaining
    layers_to_merge[0] = Interp([60, 60])(layers_to_merge[0])
    layers_to_merge[1] = Interp([60, 60])(layers_to_merge[1])

    layers_to_merge[2] = Conv2D(128, kernel_size=(1, 1))(layers_to_merge[2])

    # layers_to_merge[0] = Conv2D(filters=128, kernel_size=(3, 3), padding='same')(layers_to_merge[0])
    # layers_to_merge[1] = Conv2D(filters=128, kernel_size=(3, 3), padding='same')(layers_to_merge[1])
    # layers_to_merge[2] = Conv2D(filters=128, kernel_size=(3, 3), padding='same')(layers_to_merge[2])
    # layers_to_merge[3] = Conv2D(filters=128, kernel_size=(3, 3), padding='same')(layers_to_merge[3])
    #
    # layers_to_merge[0] = BN(name='diff_layer_merging_bn_0')(layers_to_merge[0])
    # layers_to_merge[1] = BN(name='diff_layer_merging_bn_1')(layers_to_merge[1])
    # layers_to_merge[2] = BN(name='diff_layer_merging_bn_2')(layers_to_merge[2])
    # layers_to_merge[3] = BN(name='diff_layer_merging_bn_3')(layers_to_merge[3])
    #
    # layers_to_merge[0] = Activation('relu')(layers_to_merge[0])
    # layers_to_merge[1] = Activation('relu')(layers_to_merge[1])
    # layers_to_merge[2] = Activation('relu')(layers_to_merge[2])
    # layers_to_merge[3] = Activation('relu')(layers_to_merge[3])

    layers_to_merge.append(res)
    final = Concatenate()(layers_to_merge)

    final = Conv2D(filters=2048, kernel_size=(1, 1))(final)

    return final


def interp_block(prev_layer, level, feature_map_shape, input_shape):
    if input_shape == (473, 473):
        kernel_strides_map = {1: 60,
                              2: 30,
                              3: 20,
                              6: 10}
    elif input_shape == (713, 713):
        kernel_strides_map = {1: 90,
                              2: 45,
                              3: 30,
                              6: 15}
    else:
        print("Pooling parameters for input shape ",
              input_shape, " are not defined.")
        exit(1)

    names = [
        "conv5_3_pool" + str(level) + "_conv",
        "conv5_3_pool" + str(level) + "_conv_bn"
    ]
    kernel = (kernel_strides_map[level], kernel_strides_map[level])
    strides = (kernel_strides_map[level], kernel_strides_map[level])
    prev_layer = AveragePooling2D(kernel, strides=strides)(prev_layer)
    prev_layer = Conv2D(512, (1, 1), strides=(1, 1), name=names[0],
                        use_bias=False)(prev_layer)
    prev_layer = BN(name=names[1])(prev_layer)
    prev_layer = Activation('relu')(prev_layer)
    # prev_layer = Lambda(Interp, arguments={
    #                    'shape': feature_map_shape})(prev_layer)
    prev_layer = Interp(feature_map_shape)(prev_layer)
    return prev_layer


def build_pyramid_pooling_module(res, input_shape):
    """Build the Pyramid Pooling Module."""
    # ---PSPNet concat layers with Interpolation
    feature_map_size = tuple(int(ceil(input_dim / 8.0))
                             for input_dim in input_shape)

    interp_block1 = interp_block(res, 1, feature_map_size, input_shape)
    interp_block2 = interp_block(res, 2, feature_map_size, input_shape)
    interp_block3 = interp_block(res, 3, feature_map_size, input_shape)
    interp_block6 = interp_block(res, 6, feature_map_size, input_shape)

    # concat all these layers. resulted
    # shape=(1,feature_map_size_x,feature_map_size_y,4096)
    res = Concatenate()([res,
                         interp_block6,
                         interp_block3,
                         interp_block2,
                         interp_block1])
    return res


def build_pyramid_pooling_module_background_subtraction(res, input_shape):
    """Build the Pyramid Pooling Module."""
    # ---PSPNet concat layers with Interpolation
    feature_map_size = tuple(int(ceil(input_dim / 8.0)) for input_dim in input_shape)

    interp_block1 = interp_block(res, 1, feature_map_size, input_shape)
    interp_block2 = interp_block(res, 2, feature_map_size, input_shape)
    interp_block3 = interp_block(res, 3, feature_map_size, input_shape)
    interp_block6 = interp_block(res, 6, feature_map_size, input_shape)

    res_channels_down_sample = layers.Conv2D(filters=512, kernel_size=(1, 1), padding='same', activation='relu')(res)

    # interp_block1_more_filters = layers.Conv2D(filters=2048, kernel_size=(1, 1), padding='same')(interp_block1)
    interp_block1_background_subtraction = layers.Subtract()([interp_block1, res_channels_down_sample])
    interp_block1_background_subtraction = layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same',
                                                         activation='relu')(interp_block1_background_subtraction)

    # interp_block2_more_filters = layers.Conv2D(filters=2048, kernel_size=(1, 1), padding='same')(interp_block2)
    interp_block2_background_subtraction = layers.Subtract()([interp_block2, res_channels_down_sample])
    interp_block2_background_subtraction = layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same',
                                                         activation='relu')(interp_block2_background_subtraction)

    # interp_block3_more_filters = layers.Conv2D(filters=2048, kernel_size=(1, 1), padding='same')(interp_block3)
    interp_block3_background_subtraction = layers.Subtract()([interp_block3, res_channels_down_sample])
    interp_block3_background_subtraction = layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same',
                                                         activation='relu')(interp_block3_background_subtraction)

    # interp_block6_more_filters = layers.Conv2D(filters=2048, kernel_size=(1, 1), padding='same')(interp_block6)
    interp_block6_background_subtraction = layers.Subtract()([interp_block6, res_channels_down_sample])
    interp_block6_background_subtraction = layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same',
                                                         activation='relu')(interp_block6_background_subtraction)

    # concat all these layers. resulted
    # shape=(1,feature_map_size_x,feature_map_size_y,4096)
    res = Concatenate()([res,
                         interp_block6,
                         interp_block3,
                         interp_block2,
                         interp_block1,
                         interp_block1_background_subtraction,
                         interp_block2_background_subtraction,
                         interp_block3_background_subtraction,
                         interp_block6_background_subtraction
                         ])
    return res


def build_pyramid_pooling_module_weighted(res, input_shape):
    """Build the Pyramid Pooling Module."""
    # ---PSPNet concat layers with Interpolation
    feature_map_size = tuple(int(ceil(input_dim / 8.0))
                             for input_dim in input_shape)

    interp_block1 = interp_block(res, 1, feature_map_size, input_shape)
    interp_block2 = interp_block(res, 2, feature_map_size, input_shape)
    interp_block3 = interp_block(res, 3, feature_map_size, input_shape)
    interp_block6 = interp_block(res, 6, feature_map_size, input_shape)

    res_less_channels = layers.Conv2D(filters=512, kernel_size=(1, 1), padding='same', activation='relu')(res)

    # interp_block1_more_filters = layers.Conv2D(filters=2048, kernel_size=(1, 1), padding='same')(interp_block1)
    interp_block1_weighted = layers.Subtract()([interp_block1, res_less_channels])
    interp_block1_weighted = layers.Conv2D(filters=1024, kernel_size=(3, 3), padding='same', activation='relu')(
        interp_block1_weighted)
    interp_block1_weighted = layers.Conv2D(filters=512, kernel_size=(1, 1), padding='same', activation='sigmoid')(
        interp_block1_weighted)
    interp_block1_weighted = layers.Multiply()([interp_block1, interp_block1_weighted])

    # interp_block2_more_filters = layers.Conv2D(filters=2048, kernel_size=(1, 1), padding='same')(interp_block2)
    interp_block2_weighted = layers.Subtract()([interp_block2, res_less_channels])
    interp_block2_weighted = layers.Conv2D(filters=1024, kernel_size=(3, 3), padding='same', activation='relu')(
        interp_block2_weighted)
    interp_block2_weighted = layers.Conv2D(filters=512, kernel_size=(1, 1), padding='same', activation='sigmoid')(
        interp_block2_weighted)
    interp_block2_weighted = layers.Multiply()([interp_block2, interp_block2_weighted])

    # interp_block3_more_filters = layers.Conv2D(filters=2048, kernel_size=(1, 1), padding='same')(interp_block3)
    interp_block3_weighted = layers.Subtract()([interp_block3, res_less_channels])
    interp_block3_weighted = layers.Conv2D(filters=1024, kernel_size=(3, 3), padding='same', activation='relu')(
        interp_block3_weighted)
    interp_block3_weighted = layers.Conv2D(filters=512, kernel_size=(1, 1), padding='same', activation='sigmoid')(
        interp_block3_weighted)
    interp_block3_weighted = layers.Multiply()([interp_block3, interp_block3_weighted])

    # interp_block6_more_filters = layers.Conv2D(filters=2048, kernel_size=(1, 1), padding='same')(interp_block6)
    interp_block6_weighted = layers.Subtract()([interp_block6, res_less_channels])
    interp_block6_weighted = layers.Conv2D(filters=1024, kernel_size=(3, 3), padding='same', activation='relu')(
        interp_block6_weighted)
    interp_block6_weighted = layers.Conv2D(filters=512, kernel_size=(1, 1), padding='same', activation='sigmoid')(
        interp_block6_weighted)
    interp_block6_weighted = layers.Multiply()([interp_block6, interp_block6_weighted])

    # concat all these layers. resulted
    # shape=(1,feature_map_size_x,feature_map_size_y,4096)
    res = Concatenate()([res,
                         interp_block6,
                         interp_block3,
                         interp_block2,
                         interp_block1,
                         interp_block1_weighted,
                         interp_block2_weighted,
                         interp_block3_weighted,
                         interp_block6_weighted
                         ])
    return res


def i3d_build_pyramid_pooling_module(res, input_shape):
    """Build the Pyramid Pooling Module."""
    # ---PSPNet concat layers with Interpolation
    feature_map_size = tuple(int(ceil(input_dim / 8.0))
                             for input_dim in input_shape)

    interp_block1 = interp_block(res, 1, feature_map_size, input_shape)
    interp_block2 = interp_block(res, 2, feature_map_size, input_shape)
    interp_block3 = interp_block(res, 3, feature_map_size, input_shape)
    interp_block6 = interp_block(res, 6, feature_map_size, input_shape)

    # concat all these layers. resulted
    # shape=(1,feature_map_size_x,feature_map_size_y,4096)
    res = Concatenate()([res,
                         interp_block6,
                         interp_block3,
                         interp_block2,
                         interp_block1])
    return res


def _build_pspnet(nb_classes, resnet_layers, input_shape, activation='softmax'):
    assert IMAGE_ORDERING == 'channels_last'

    inp = Input((input_shape[0], input_shape[1], 3))

    res = ResNet(inp, layers=resnet_layers)

    psp = build_pyramid_pooling_module(res, input_shape)

    x = Conv2D(512, (3, 3), strides=(1, 1), padding="same", name="conv5_4",
               use_bias=False)(psp)
    x = BN(name="conv5_4_bn")(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)

    x = Conv2D(nb_classes, (1, 1), strides=(1, 1), name="conv6")(x)
    # x = Lambda(Interp, arguments={'shape': (
    #    input_shape[0], input_shape[1])})(x)
    x = Interp([input_shape[0], input_shape[1]])(x)

    model = get_segmentation_model(inp, x)

    return model


def feature_super_resolution(ending_layer):
    # out 1
    out1 = layers.Conv2DTranspose(128, (8, 8))(ending_layer)
    out1 = layers.Activation('relu')(out1)
    # out 2
    out2 = layers.UpSampling2D(2)(ending_layer)
    out2 = layers.Conv2D(1, (1, 1))(out2)
    out2 = layers.Activation('relu')(out2)
    # merging
    out = layers.Add()([out1, out2])
    out = BN('feature_super_resolution_BN_1')(out)

    # out 1
    out1 = layers.Conv2DTranspose(128, (15, 15))(out)
    out1 = layers.Activation('relu')(out1)
    # out 2
    out2 = layers.UpSampling2D(2)(out)
    out2 = layers.Conv2D(1, (1, 1))(out2)
    out2 = layers.Activation('relu')(out2)
    # merging
    out = layers.Add()([out1, out2])
    out = BN('feature_super_resolution_BN_2')(out)

    # out 1
    out1 = layers.Conv2DTranspose(128, (29, 29))(out)
    out1 = layers.Activation('relu')(out1)
    # out 2
    out2 = layers.UpSampling2D(2)(out)
    out2 = layers.Conv2D(1, (1, 1))(out2)
    out2 = layers.Activation('relu')(out2)
    # merging
    out = layers.Add()([out1, out2])
    out = BN('feature_super_resolution_BN_3')(out)

    return out


def _build_pspnet_inception(nb_classes, inception_net_input, inception_net_output, resnet_layers, input_shape,
                            activation='softmax'):
    assert IMAGE_ORDERING == 'channels_last'

    # feature level super resolution
    inception_net_output = feature_super_resolution(inception_net_output)

    inception_net_output = Conv2D(kernel_size=(3, 3), filters=2048, padding='same')(inception_net_output)
    inception_net_output = BN('inception_to_psp_BN')(inception_net_output)
    inception_net_output = Interp([60, 60])(inception_net_output)

    psp = build_pyramid_pooling_module(inception_net_output, (473, 473))

    x = Conv2D(512, (3, 3), strides=(1, 1), padding="same", name="conv5_4",
               use_bias=False)(psp)
    x = BN(name="conv5_4_bn")(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)

    x = Conv2D(nb_classes, (1, 1), strides=(1, 1), name="conv6")(x)
    # x = Lambda(Interp, arguments={'shape': (
    #    input_shape[0], input_shape[1])})(x)
    x = Interp([input_shape[0], input_shape[1]])(x)

    model = get_segmentation_model_i3d_inception(inception_net_input, x)

    return model


def _build_pspnet_with_weighted_output(nb_classes, resnet_layers, input_shape, activation='softmax'):
    assert IMAGE_ORDERING == 'channels_last'

    inp = Input((input_shape[0], input_shape[1], 3))

    res = ResNet_with_different_level_features(inp, layers=resnet_layers)

    psp = build_pyramid_pooling_module(res, input_shape)

    xx = Conv2D(512, (3, 3), strides=(1, 1), padding="same", name="conv5_4",
               use_bias=False)(psp)
    xx = BN(name="conv5_4_bn")(xx)
    xx = Activation('relu')(xx)
    xx = Dropout(0.1)(xx)

    # for channel and pixel attention use
    # channel_pixel_weighting = BN(name="channel_pixel_attention_bn_1")(res)
    # channel_pixel_weighting = Activation('relu')(channel_pixel_weighting)
    # channel_pixel_weighting = Conv2D(512, (1, 1), padding='same', use_bias=False)(channel_pixel_weighting)
    # channel_pixel_weighting = BN(name="channel_pixel_attention_bn_2")(channel_pixel_weighting)
    # channel_pixel_weighting = Activation('relu')(channel_pixel_weighting)

    # channel weighting starts
    channel_weighting = MaxPooling2D((60, 60))(xx)
    channel_weighting = layers.Dense(256)(channel_weighting)
    channel_weighting = Activation('relu')(channel_weighting)
    channel_weighting = Dropout(0.1)(channel_weighting)
    channel_weighting = layers.Dense(512)(channel_weighting)
    channel_weighting = Activation('sigmoid')(channel_weighting)
    # channel_weighting = Dropout(0.1)(channel_weighting)
    xx = layers.Multiply()([xx, channel_weighting])
    # channel weighting ends

    # pixel attention starts
    pixel_weighting = Conv2D(1024, (3, 3), strides=(1, 1), padding="same", use_bias=False)(xx)
    pixel_weighting = BN(name="pixel_attention_conv_bn_1")(pixel_weighting)
    pixel_weighting = Activation('relu')(pixel_weighting)
    pixel_weighting = Dropout(0.1)(pixel_weighting)
    pixel_weighting = Conv2D(1024, (3, 3), strides=(1, 1), padding="same", use_bias=False)(pixel_weighting)
    pixel_weighting = BN(name="pixel_attention_conv_bn_2")(pixel_weighting)
    pixel_weighting = Activation('relu')(pixel_weighting)
    pixel_weighting = Conv2D(512, (1, 1), strides=(1, 1), padding="same", use_bias=False)(pixel_weighting)
    pixel_weighting = BN(name="pixel_attention_conv_bn_3")(pixel_weighting)
    pixel_weighting = Activation('relu')(pixel_weighting)
    pixel_weighting = Dropout(0.1)(pixel_weighting)
    pixel_weighting = Conv2D(2, (1, 1))(pixel_weighting)

    # pixel_weighting_multiply = Activation('sigmoid')(pixel_weighting)
    # pixel_weighting_multiply = Lambda(lambda val: val[:, :, :, 1])(pixel_weighting_multiply)
    # pixel_weighting_multiply = layers.Reshape((60, 60, 1))(pixel_weighting_multiply)
    # x = layers.Multiply()([x, pixel_weighting_multiply])
    pixel_weighting_multiply = Activation('relu')(pixel_weighting)
    pixel_weighting_multiply = Conv2D(1, (1, 1))(pixel_weighting_multiply)
    pixel_weighting_multiply = Activation('sigmoid')(pixel_weighting_multiply)
    xx = layers.Multiply()([xx, pixel_weighting_multiply])
    xx = layers.Add()([xx, pixel_weighting_multiply])

    pixel_weighting_output = Interp([input_shape[0], input_shape[1]])(pixel_weighting)
    # pixel attention ends

    x = Conv2D(nb_classes, (1, 1), strides=(1, 1), name="conv6")(xx)
    # x = Lambda(Interp, arguments={'shape': (
    #    input_shape[0], input_shape[1])})(x)
    x = Interp([input_shape[0], input_shape[1]])(x)

    model = get_segmentation_model_with_weighted_output(inp, x, pixel_weighting_output)

    return model


def _build_pspnet_temporal_with_weighted_output(nb_classes, resnet_layers, input_shape, activation='softmax',
                                                volume_frames=3):

    assert IMAGE_ORDERING == 'channels_last'

    inp = Input((volume_frames, input_shape[0], input_shape[1], 3))

    res = ResNet_temporal_with_different_level_features(inp, layers=resnet_layers)

    psp = build_pyramid_pooling_module(res, input_shape)

    xx = Conv2D(512, (3, 3), strides=(1, 1), padding="same", name="conv5_4",
                use_bias=False)(psp)
    xx = BN(name="conv5_4_bn")(xx)
    xx = Activation('relu')(xx)
    xx = Dropout(0.1)(xx)

    # for channel and pixel attention use
    # channel_pixel_weighting = BN(name="channel_pixel_attention_bn_1")(res)
    # channel_pixel_weighting = Activation('relu')(channel_pixel_weighting)
    # channel_pixel_weighting = Conv2D(512, (1, 1), padding='same', use_bias=False)(channel_pixel_weighting)
    # channel_pixel_weighting = BN(name="channel_pixel_attention_bn_2")(channel_pixel_weighting)
    # channel_pixel_weighting = Activation('relu')(channel_pixel_weighting)

    # channel weighting starts
    channel_weighting = MaxPooling2D((60, 60))(xx)
    channel_weighting = layers.Dense(256)(channel_weighting)
    channel_weighting = Activation('relu')(channel_weighting)
    channel_weighting = Dropout(0.1)(channel_weighting)
    channel_weighting = layers.Dense(512)(channel_weighting)
    channel_weighting = Activation('sigmoid')(channel_weighting)
    # channel_weighting = Dropout(0.1)(channel_weighting)
    xx = layers.Multiply()([xx, channel_weighting])
    # channel weighting ends

    # pixel attention starts
    pixel_weighting = Conv2D(1024, (3, 3), strides=(1, 1), padding="same", use_bias=False)(xx)
    pixel_weighting = BN(name="pixel_attention_conv_bn_1")(pixel_weighting)
    pixel_weighting = Activation('relu')(pixel_weighting)
    pixel_weighting = Dropout(0.1)(pixel_weighting)
    pixel_weighting = Conv2D(1024, (3, 3), strides=(1, 1), padding="same", use_bias=False)(pixel_weighting)
    pixel_weighting = BN(name="pixel_attention_conv_bn_2")(pixel_weighting)
    pixel_weighting = Activation('relu')(pixel_weighting)
    pixel_weighting = Conv2D(512, (1, 1), strides=(1, 1), padding="same", use_bias=False)(pixel_weighting)
    pixel_weighting = BN(name="pixel_attention_conv_bn_3")(pixel_weighting)
    pixel_weighting = Activation('relu')(pixel_weighting)
    pixel_weighting = Dropout(0.1)(pixel_weighting)
    pixel_weighting = Conv2D(2, (1, 1))(pixel_weighting)

    # pixel_weighting_multiply = Activation('sigmoid')(pixel_weighting)
    # pixel_weighting_multiply = Lambda(lambda val: val[:, :, :, 1])(pixel_weighting_multiply)
    # pixel_weighting_multiply = layers.Reshape((60, 60, 1))(pixel_weighting_multiply)
    # x = layers.Multiply()([x, pixel_weighting_multiply])
    pixel_weighting_multiply = Activation('relu')(pixel_weighting)
    pixel_weighting_multiply = Conv2D(1, (1, 1))(pixel_weighting_multiply)
    pixel_weighting_multiply = Activation('sigmoid')(pixel_weighting_multiply)
    xx = layers.Multiply()([xx, pixel_weighting_multiply])
    xx = layers.Add()([xx, pixel_weighting_multiply])

    pixel_weighting_output = Interp([input_shape[0], input_shape[1]])(pixel_weighting)
    # pixel attention ends

    x = Conv2D(nb_classes, (1, 1), strides=(1, 1), name="conv6")(xx)
    # x = Lambda(Interp, arguments={'shape': (
    #    input_shape[0], input_shape[1])})(x)
    x = Interp([input_shape[0], input_shape[1]])(x)

    model = get_temporal_segmentation_model_with_weighted_output(inp, x, pixel_weighting_output)

    return model


def _build_pspnet_element_weighting(nb_classes, resnet_layers, input_shape, activation='softmax'):
    def second_part(net):
        # starting the other side which will be multiplied
        x2 = Conv2D(nb_classes, (1, 1), strides=(1, 1), padding='same', name="conv_1_1x1_multiply_side",
                    use_bias=False)(net)
        x2 = BN(name="bn_1_multiply_side")(x2)
        x2 = Activation('relu')(x2)
        x2 = Conv2D(nb_classes, (3, 3), strides=(1, 1), padding='same', name="conv_2_3x3_multiply_side",
                    use_bias=False)(x2)
        x2 = BN(name="bn_2_multiply_side")(x2)
        x2 = Activation('relu')(x2)
        x2 = Conv2D(nb_classes, (3, 3), strides=(1, 1), padding='same', name="conv_3_3x3_multiply_side",
                    use_bias=False)(x2)
        x2 = BN(name="bn_3_multiply_side")(x2)
        x2 = Activation('sigmoid')(x2)
        return x2

    assert IMAGE_ORDERING == 'channels_last'

    inp = Input((input_shape[0], input_shape[1], 3))

    res = ResNet(inp, layers=resnet_layers)

    psp = build_pyramid_pooling_module(res, input_shape)

    x2 = second_part(psp)

    x = Conv2D(512, (3, 3), strides=(1, 1), padding="same", name="conv5_4",
               use_bias=False)(psp)
    x = BN(name="conv5_4_bn")(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)

    x = Conv2D(nb_classes, (1, 1), strides=(1, 1), name="conv6")(x)

    x = layers.Multiply()([x, x2])

    # x = Lambda(Interp, arguments={'shape': (
    #    input_shape[0], input_shape[1])})(x)
    x = Interp([input_shape[0], input_shape[1]])(x)

    model = get_segmentation_model(inp, x)

    return model


def _build_pspnet_resnet_diff_level_features(nb_classes, resnet_layers, input_shape, activation='softmax'):
    assert IMAGE_ORDERING == 'channels_last'

    inp = Input((input_shape[0], input_shape[1], 3))

    res = ResNet_with_different_level_features(inp, layers=resnet_layers)

    # psp = build_pyramid_pooling_module(res, input_shape)
    psp = build_pyramid_pooling_module_weighted(res, input_shape)

    x = Conv2D(512, (3, 3), strides=(1, 1), padding="same", name="conv5_4",
               use_bias=False)(psp)
    x = BN(name="conv5_4_bn")(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)

    x = Conv2D(nb_classes, (1, 1), strides=(1, 1), name="conv6")(x)
    # x = Lambda(Interp, arguments={'shape': (
    #    input_shape[0], input_shape[1])})(x)
    x = Interp([input_shape[0], input_shape[1]])(x)

    model = get_segmentation_model(inp, x)

    return model


def _build_pspnet_background_subtraction(nb_classes, resnet_layers, input_shape, activation='softmax'):
    assert IMAGE_ORDERING == 'channels_last'

    inp = Input((input_shape[0], input_shape[1], 3))

    res = ResNet(inp, layers=resnet_layers)

    psp = build_pyramid_pooling_module_background_subtraction(res, input_shape)

    x = Conv2D(512, (3, 3), strides=(1, 1), padding="same", name="conv5_4",
               use_bias=False)(psp)
    x = BN(name="conv5_4_bn")(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)

    x = Conv2D(nb_classes, (1, 1), strides=(1, 1), name="conv6")(x)
    # x = Lambda(Interp, arguments={'shape': (
    #    input_shape[0], input_shape[1])})(x)
    x = Interp([input_shape[0], input_shape[1]])(x)

    model = get_segmentation_model(inp, x)

    return model

def _build_pspnet_weighted(nb_classes, resnet_layers, input_shape, activation='softmax'):
    assert IMAGE_ORDERING == 'channels_last'

    inp = Input((input_shape[0], input_shape[1], 3))

    res = ResNet(inp, layers=resnet_layers)

    psp = build_pyramid_pooling_module_weighted(res, input_shape)

    x = Conv2D(512, (3, 3), strides=(1, 1), padding="same", name="conv5_4",
               use_bias=False)(psp)
    x = BN(name="conv5_4_bn")(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)

    x = Conv2D(nb_classes, (1, 1), strides=(1, 1), name="conv6")(x)
    # x = Lambda(Interp, arguments={'shape': (
    #    input_shape[0], input_shape[1])})(x)
    x = Interp([input_shape[0], input_shape[1]])(x)

    model = get_segmentation_model(inp, x)

    return model


def _i3d_build_pspnet(nb_classes, input_shape, activation='softmax'):
    assert IMAGE_ORDERING == 'channels_last'

    inp = Input((14, 14, 480))
    for_psp = Interp([60, 60])(inp)

    for_psp = Conv2D(filters=1024, kernel_size=(3, 3), padding='same')(for_psp)
    for_psp = BN(name='i3d_bn_1')(for_psp)
    for_psp = Activation('relu')(for_psp)

    for_psp = Conv2D(filters=1024, kernel_size=(3, 3), padding='same')(for_psp)
    for_psp = BN(name='i3d_bn_2')(for_psp)
    for_psp = Activation('relu')(for_psp)

    for_psp = Conv2D(filters=2048, kernel_size=(3, 3), padding='same')(for_psp)
    for_psp = BN(name='i3d_bn_3')(for_psp)
    for_psp = Activation('relu')(for_psp)

    for_psp = Conv2D(filters=1024, kernel_size=(1, 1), padding='same')(for_psp)
    for_psp = BN(name='i3d_bn_4')(for_psp)
    for_psp = Activation('relu')(for_psp)

    for_psp = Conv2D(filters=2048, kernel_size=(3, 3), padding='same')(for_psp)
    for_psp = BN(name='i3d_bn_5')(for_psp)
    for_psp = Activation('relu')(for_psp)

    psp = i3d_build_pyramid_pooling_module(for_psp, (473, 473))

    x = Conv2D(512, (3, 3), strides=(1, 1), padding="same", name="conv5_4",
               use_bias=False)(psp)
    x = BN(name="conv5_4_bn")(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)

    # channel weighting starts
    channel_weighting = MaxPooling2D((60, 60))(x)
    channel_weighting = layers.Dense(256)(channel_weighting)
    channel_weighting = Activation('relu')(channel_weighting)
    channel_weighting = Dropout(0.1)(channel_weighting)
    channel_weighting = layers.Dense(512)(channel_weighting)
    channel_weighting = Activation('sigmoid')(channel_weighting)
    # channel_weighting = Dropout(0.1)(channel_weighting)
    x = layers.Multiply()([x, channel_weighting])
    # channel weighting ends

    # pixel attention starts
    pixel_weighting = Conv2D(1024, (3, 3), strides=(1, 1), padding="same", use_bias=False)(x)
    pixel_weighting = BN(name="pixel_attention_conv_bn_1")(pixel_weighting)
    pixel_weighting = Activation('relu')(pixel_weighting)
    pixel_weighting = Dropout(0.1)(pixel_weighting)
    pixel_weighting = Conv2D(1024, (3, 3), strides=(1, 1), padding="same", use_bias=False)(pixel_weighting)
    pixel_weighting = BN(name="pixel_attention_conv_bn_2")(pixel_weighting)
    pixel_weighting = Activation('relu')(pixel_weighting)
    pixel_weighting = Conv2D(512, (1, 1), strides=(1, 1), padding="same", use_bias=False)(pixel_weighting)
    pixel_weighting = BN(name="pixel_attention_conv_bn_3")(pixel_weighting)
    pixel_weighting = Activation('relu')(pixel_weighting)
    pixel_weighting = Dropout(0.1)(pixel_weighting)
    pixel_weighting = Conv2D(2, (1, 1))(pixel_weighting)

    # pixel_weighting_multiply = Activation('sigmoid')(pixel_weighting)
    # pixel_weighting_multiply = Lambda(lambda val: val[:, :, :, 1])(pixel_weighting_multiply)
    # pixel_weighting_multiply = layers.Reshape((60, 60, 1))(pixel_weighting_multiply)
    # x = layers.Multiply()([x, pixel_weighting_multiply])
    pixel_weighting_multiply = Activation('relu')(pixel_weighting)
    pixel_weighting_multiply = Conv2D(1, (1, 1))(pixel_weighting_multiply)
    pixel_weighting_multiply = Activation('sigmoid')(pixel_weighting_multiply)
    x = layers.Multiply()([x, pixel_weighting_multiply])
    x = layers.Add()([x, pixel_weighting_multiply])

    pixel_weighting_output = Interp([input_shape[0], input_shape[1]])(pixel_weighting)
    # pixel attention ends

    x = Conv2D(nb_classes, (1, 1), strides=(1, 1), name="conv6")(x)
    # x = Lambda(Interp, arguments={'shape': (
    #    input_shape[0], input_shape[1])})(x)
    x = Interp([input_shape[0], input_shape[1]])(x)

    model = get_segmentation_model_with_weighted_output(inp, x, pixel_weighting_output)

    from ..train import train_i3d
    from ..predict import i3d_predict, predict_with_segmentation_return_i3d

    model.train = MethodType(train_i3d, model)
    model.predict_segmentation = MethodType(i3d_predict, model)
    model.predict_segmentation_with_segmentation_return = MethodType(predict_with_segmentation_return_i3d, model)

    return model
