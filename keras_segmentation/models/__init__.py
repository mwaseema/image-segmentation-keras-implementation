model_from_name = {}

from . import fcn
from . import segnet
from . import unet
from . import pspnet
from . import two_stream_network
from . import IoU_network
from . import bounding_box_based_network
from . import bounding_box_iou_based_network

model_from_name["fcn_8"] = fcn.fcn_8
model_from_name["fcn_32"] = fcn.fcn_32
model_from_name["fcn_8_vgg"] = fcn.fcn_8_vgg
model_from_name["fcn_32_vgg"] = fcn.fcn_32_vgg
model_from_name["fcn_8_resnet50"] = fcn.fcn_8_resnet50
model_from_name["fcn_32_resnet50"] = fcn.fcn_32_resnet50
model_from_name["fcn_8_mobilenet"] = fcn.fcn_8_mobilenet
model_from_name["fcn_32_mobilenet"] = fcn.fcn_32_mobilenet

model_from_name["pspnet"] = pspnet.pspnet
model_from_name["vgg_pspnet"] = pspnet.vgg_pspnet
model_from_name["resnet50_pspnet"] = pspnet.resnet50_pspnet

model_from_name["vgg_pspnet"] = pspnet.vgg_pspnet
model_from_name["resnet50_pspnet"] = pspnet.resnet50_pspnet

model_from_name["pspnet_50"] = pspnet.pspnet_50
model_from_name["pspnet_50_with_weighted_output"] = pspnet.pspnet_50_with_weighted_output
model_from_name["pspnet_50_temporal_with_weighted_output"] = pspnet.pspnet_50_temporal_with_weighted_output
model_from_name["pspnet_50_i3d_inception"] = pspnet.pspnet_50_i3d_inception
model_from_name["pspnet_50_background_subtraction"] = pspnet.pspnet_50_background_subtraction
model_from_name["pspnet_50_weighted"] = pspnet.pspnet_50_weighted
model_from_name["pspnet_50_resnet_diff_level_features"] = pspnet.pspnet_50_resnet_diff_level_features
model_from_name["pspnet_50_element_weighting"] = pspnet.pspnet_50_element_weighting
model_from_name["i3d_pspnet"] = pspnet.i3d_pspnet
model_from_name["pspnet_101"] = pspnet.pspnet_101
model_from_name["two_stream_pspnet_101"] = two_stream_network.two_stream_pspnet_101
model_from_name["two_stream_pspnet_101_average_merge"] = two_stream_network.two_stream_pspnet_101_average_merge
model_from_name["two_stream_pspnet_50"] = two_stream_network.two_stream_pspnet_50
model_from_name["two_stream_pspnet_50_average_merge"] = two_stream_network.two_stream_pspnet_50_average_merge
model_from_name["two_stream_resnet50_pspnet"] = two_stream_network.two_stream_resnet50_pspnet
model_from_name["two_stream_resnet50_pspnet_average_merge"] = two_stream_network.two_stream_resnet50_pspnet_average_merge

model_from_name["IoU_network_pspnet50"] = IoU_network.IoU_network_pspnet50

model_from_name["bounding_box_based_network_model"] = bounding_box_based_network.bounding_box_based_network_model

model_from_name["bounding_box_iou_based_network"] = bounding_box_iou_based_network.bounding_box_iou_based_network

# model_from_name["mobilenet_pspnet"] = pspnet.mobilenet_pspnet


model_from_name["unet_mini"] = unet.unet_mini
model_from_name["unet"] = unet.unet
model_from_name["vgg_unet"] = unet.vgg_unet
model_from_name["resnet50_unet"] = unet.resnet50_unet
model_from_name["mobilenet_unet"] = unet.mobilenet_unet

model_from_name["segnet"] = segnet.segnet
model_from_name["vgg_segnet"] = segnet.vgg_segnet
model_from_name["resnet50_segnet"] = segnet.resnet50_segnet
model_from_name["mobilenet_segnet"] = segnet.mobilenet_segnet
