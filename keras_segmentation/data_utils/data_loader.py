
import numpy as np
import cv2
import glob
import itertools
import os
from tqdm import tqdm

from .frames_data import get_frame_and_ground_truth_crop, get_video_wise_list
from .standalone_IoU_model_libs import get_bounding_boxes
from ..data_utils.bounding_box_based_network_utils import get_im_patch
from ..models.config import IMAGE_ORDERING
from .augmentation import augment_seg, two_stream_augment_seg
from . import standalone_IoU_model_libs
import random

random.seed(0)
class_colors = [  ( random.randint(0,255),random.randint(0,255),random.randint(0,255)   ) for _ in range(5000)  ]




def get_pairs_from_paths( images_path , segs_path ):
	images = glob.glob( os.path.join(images_path,"*.jpg")  ) + glob.glob( os.path.join(images_path,"*.png")  ) +  glob.glob( os.path.join(images_path,"*.jpeg")  )
	segmentations  =  glob.glob( os.path.join(segs_path,"*.png")  )

	segmentations_d = dict( zip(segmentations,segmentations ))

	ret = []

	for im in images:
		seg_bnme = os.path.basename(im).replace(".jpg" , ".png").replace(".jpeg" , ".png")
		seg = os.path.join( segs_path , seg_bnme  )
		assert ( seg in segmentations_d ),  (im + " is present in "+images_path +" but "+seg_bnme+" is not found in "+segs_path + " . Make sure annotation image are in .png"  )
		ret.append((im , seg) )

	return ret


def get_pairs_from_paths_i3d(features_path, segs_path):
	i3d_feature_paths = glob.glob(os.path.join(features_path, '*.npy'))
	ret = []
	for i3d_feature_path in i3d_feature_paths:
		filename = os.path.basename(i3d_feature_path)
		filename_wo_ext, _ = os.path.splitext(filename)
		segmentation_path = os.path.join(segs_path, f'{filename_wo_ext}.png')
		ret.append((i3d_feature_path, segmentation_path))
	return ret


def two_stream_get_pairs_from_paths(images_path, flows_path, segs_path):
	images = glob.glob(os.path.join(images_path, "*.jpg")) + glob.glob(os.path.join(images_path, "*.png")) + glob.glob(
		os.path.join(images_path, "*.jpeg"))
	segmentations = glob.glob(os.path.join(segs_path, "*.png"))
	flows = glob.glob(os.path.join(flows_path, "*.png"))

	segmentations_d = dict(zip(segmentations, segmentations))
	flows_d = dict(zip(flows, flows))

	ret = []

	for im in images:
		seg_bnme = os.path.basename(im).replace(".jpg", ".png").replace(".jpeg", ".png")
		seg = os.path.join(segs_path, seg_bnme)
		flw = os.path.join(flows_path, seg_bnme)
		assert (seg in segmentations_d), (
					im + " is present in " + images_path + " but " + seg_bnme + " is not found in " + segs_path + " . Make sure annotation image are in .png")
		assert (flw in flows_d), (
				im + " is present in " + images_path + " but " + seg_bnme + " is not found in " + flows_path + " . Make sure flow image are in .png")
		ret.append((im, flw, seg))

	return ret




def get_image_arr( path , width , height , imgNorm="sub_mean" , odering='channels_first' ):


	if type( path ) is np.ndarray:
		img = path
	else:
		img = cv2.imread(path, 1)

	if imgNorm == "sub_and_divide":
		img = np.float32(cv2.resize(img, ( width , height ))) / 127.5 - 1
	elif imgNorm == "sub_mean":
		img = cv2.resize(img, ( width , height ))
		img = img.astype(np.float32)
		img[:,:,0] -= 103.939
		img[:,:,1] -= 116.779
		img[:,:,2] -= 123.68
		img = img[ : , : , ::-1 ]
	elif imgNorm == "divide":
		img = cv2.resize(img, ( width , height ))
		img = img.astype(np.float32)
		img = img/255.0

	if odering == 'channels_first':
		img = np.rollaxis(img, 2, 0)
	return img






def get_segmentation_arr( path , nClasses ,  width , height , no_reshape=False ):

	seg_labels = np.zeros((  height , width  , nClasses ))

	if type( path ) is np.ndarray:
		img = path
	else:
		img = cv2.imread(path, 1)

	img = cv2.resize(img, ( width , height ) , interpolation=cv2.INTER_NEAREST )
	img = img[:, : , 0]

	for c in range(nClasses):
		seg_labels[: , : , c ] = (img == c ).astype(int)



	if no_reshape:
		return seg_labels

	seg_labels = np.reshape(seg_labels, ( width*height , nClasses ))
	return seg_labels




def verify_segmentation_dataset( images_path , segs_path , n_classes ):

	img_seg_pairs = get_pairs_from_paths( images_path , segs_path )

	assert len(img_seg_pairs)>0 , "Dataset looks empty or path is wrong "

	for im_fn , seg_fn in tqdm(img_seg_pairs) :
		img = cv2.imread( im_fn )
		seg = cv2.imread( seg_fn )

		assert ( img.shape[0]==seg.shape[0] and img.shape[1]==seg.shape[1] ) , "The size of image and the annotation does not match or they are corrupt "+ im_fn + " " + seg_fn
		assert ( np.max(seg[:,:,0]) < n_classes) , "The pixel values of seg image should be from 0 to "+str(n_classes-1) + " . Found pixel value "+str(np.max(seg[:,:,0]))

	print("Dataset verified! ")


def two_stream_verify_segmentation_dataset(images_path, flows_path, segs_path, n_classes):
	img_seg_pairs = two_stream_get_pairs_from_paths(images_path, flows_path, segs_path)

	assert len(img_seg_pairs) > 0, "Dataset looks empty or path is wrong "

	for im_fn, flow_fn, seg_fn in tqdm(img_seg_pairs):
		img = cv2.imread(im_fn)
		flw = cv2.imread(flow_fn)
		seg = cv2.imread(seg_fn)

		assert (img.shape[0] == seg.shape[0] and img.shape[1] == seg.shape[
			1]), "The size of image and the annotation does not match or they are corrupt " + im_fn + " " + seg_fn

		assert (img.shape[0] == flw.shape[0] and img.shape[1] == flw.shape[
			1]), "The size of image and the flow does not match or they are corrupt " + im_fn + " " + flow_fn

		assert (np.max(seg[:, :, 0]) < n_classes), "The pixel values of seg image should be from 0 to " + str(
			n_classes - 1) + " . Found pixel value " + str(np.max(seg[:, :, 0]))

	print("Dataset verified! ")


def image_segmentation_generator( images_path , segs_path ,  batch_size,  n_classes , input_height , input_width , output_height , output_width  , do_augment=False ):


	img_seg_pairs = get_pairs_from_paths( images_path , segs_path )
	random.shuffle( img_seg_pairs )
	zipped = itertools.cycle( img_seg_pairs  )

	while True:
		X = []
		Y = []
		for _ in range(batch_size):
			im, seg = next(zipped)

			im = cv2.imread(im, 1)
			seg = cv2.imread(seg, 1)

			if do_augment:
				img, seg[:, :, 0] = augment_seg(img, seg[:, :, 0])

			X.append(get_image_arr(im, input_width, input_height, odering=IMAGE_ORDERING))
			Y.append(get_segmentation_arr(seg, n_classes, output_width, output_height))

		yield np.array(X), np.array(Y)


def image_segmentation_generator_i3d_inception(features_folder, segmentation_folder, batch_size, n_classes,
											   input_height, input_width, output_height, output_width):
	feature_seg_pairs = get_pairs_from_paths_i3d(features_folder, segmentation_folder)
	random.shuffle(feature_seg_pairs)
	zipped = itertools.cycle(feature_seg_pairs)

	while True:
		X = []
		Y = []
		for _ in range(batch_size):
			feature_volume, seg = next(zipped)

			feature_volume = np.load(feature_volume)
			seg = cv2.imread(seg, 1)

			for frame_number in range(feature_volume.shape[0]):
				feature_volume[frame_number] = get_image_arr(feature_volume[frame_number], input_width, input_height, odering=IMAGE_ORDERING)

			X.append(feature_volume)
			Y.append(get_segmentation_arr(seg, n_classes, output_width, output_height))

		yield np.array(X), np.array(Y)


def image_segmentation_generator_with_weighted_output(images_path, segs_path, batch_size, n_classes, input_height,
													  input_width, output_height, output_width, do_augment=False):
	img_seg_pairs = get_pairs_from_paths(images_path, segs_path)
	random.shuffle(img_seg_pairs)
	zipped = itertools.cycle(img_seg_pairs)

	while True:
		X = []
		Y = []
		for _ in range(batch_size):
			im, seg = next(zipped)

			im = cv2.imread(im, 1)
			seg = cv2.imread(seg, 1)

			if do_augment:
				img, seg[:, :, 0] = augment_seg(im, seg[:, :, 0])

			X.append(get_image_arr(im, input_width, input_height, odering=IMAGE_ORDERING))
			Y.append(get_segmentation_arr(seg, n_classes, output_width, output_height))

		yield np.array(X), {"main_output_activation": np.array(Y), "second_output_activation": np.array(Y)}


def image_segmentation_temporal_generator_with_weighted_output(images_path, segs_path, batch_size, n_classes,
															   input_height,
															   input_width, output_height, output_width,
															   do_augment=False):
	frames_in_cuboid = 3
	middle_number = 1

	image_paths = glob.glob(os.path.join(images_path, '*.png')) + glob.glob(os.path.join(images_path, '*.jpg'))
	video_wise = get_video_wise_list(image_paths)
	video_wise = [video_wise[k] for k in video_wise.keys()]
	random.shuffle(video_wise)
	video_wise_paths = itertools.cycle(video_wise)

	X = []
	Y = []
	while True:
		video_frames = next(video_wise_paths)

		for frame_number in range(len(video_frames) - frames_in_cuboid):
			__frame_paths = video_frames[frame_number:frame_number + frames_in_cuboid]

			__frames = [cv2.imread(__frame_path) for __frame_path in __frame_paths]
			__fname = os.path.splitext(os.path.basename(__frame_paths[middle_number]))[0]
			gt = cv2.imread(os.path.join(segs_path, __fname + '.png'))

			frame_patches = [[] for i in range(9)]
			for __frame in __frames:
				patches = get_frame_and_ground_truth_crop(__frame)
				for i in range(len(patches)):
					frame_patches[i].append(
						get_image_arr(patches[i], input_width, input_height, odering=IMAGE_ORDERING))
			frame_patches = [np.array(frame_patch) for frame_patch in frame_patches]
			gt_patches = get_frame_and_ground_truth_crop(gt)

			for frame_patch, gt_patch in zip(frame_patches, gt_patches):
				X.append(frame_patch)
				Y.append(get_segmentation_arr(gt_patch, n_classes, output_width, output_height))

				if len(X) == batch_size:
					yield np.array(X), {"main_output_activation": np.array(Y), "second_output_activation": np.array(Y)}
					X = []
					Y = []


def image_segmentation_generator_bounding_box_based_network(images_path, segs_path, batch_size, n_classes, input_height,
															input_width, output_height, output_width, do_augment=False):
	img_seg_pairs = get_pairs_from_paths(images_path, segs_path)
	random.shuffle(img_seg_pairs)
	zipped = itertools.cycle(img_seg_pairs)

	X = []
	Y = []

	while True:
		im, seg = next(zipped)

		im = cv2.imread(im, 1)
		seg = cv2.imread(seg, 1)

		# get height and width of the image
		height, width, _ = seg.shape

		# get bounding boxes
		bounding_boxes = get_bounding_boxes(seg)

		for bounding_box in bounding_boxes:
			# get coords
			x1 = bounding_box['x1']
			x2 = bounding_box['x2']
			y1 = bounding_box['y1']
			y2 = bounding_box['y2']
			w = x2 - x1
			h = y2 - y1
			bbox_coords = (x1, y1, w, h)

			# get patch from image
			im_patch, _ = get_im_patch(im, bounding_box)

			X.append(get_image_arr(im_patch, input_width, input_height, odering=IMAGE_ORDERING))
			Y.append(bbox_coords)

			# check if batch is filled
			if len(X) == batch_size or len(Y) == batch_size:
				yield np.array(X), np.array(Y)
				X = []
				Y = []


def image_segmentation_generator_bounding_box_iou_based_network(images_path, segs_path, batch_size, n_classes, input_height,
															input_width, output_height, output_width, do_augment=False):
	img_seg_pairs = get_pairs_from_paths(images_path, segs_path)
	random.shuffle(img_seg_pairs)
	zipped = itertools.cycle(img_seg_pairs)

	X = []
	Y = []

	while True:
		im, seg = next(zipped)

		im = cv2.imread(im, 1)
		seg = cv2.imread(seg, 1)

		# get height and width of the image
		height, width, _ = seg.shape

		# get bounding boxes
		bounding_boxes = get_bounding_boxes(seg)

		for bounding_box in bounding_boxes:
			# get coords
			x1 = bounding_box['x1']
			x2 = bounding_box['x2']
			y1 = bounding_box['y1']
			y2 = bounding_box['y2']
			w = x2 - x1
			h = y2 - y1
			bbox_coords = (x1, y1, w, h)

			# get patch from image
			im_patch, _ = get_im_patch(im, bounding_box)

			X.append(get_image_arr(im_patch, input_width, input_height, odering=IMAGE_ORDERING))
			Y.append(bbox_coords)

			# check if batch is filled
			if len(X) == batch_size or len(Y) == batch_size:
				yield np.array(X), np.array(Y).astype(np.float64)
				X = []
				Y = []


def IoU_network_image_segmentation_generator(images_path, segs_path, batch_size, n_classes, input_height, input_width,
											 output_height, output_width, do_augment=False):
	img_seg_pairs = get_pairs_from_paths(images_path, segs_path)
	random.shuffle(img_seg_pairs)
	zipped = itertools.cycle(img_seg_pairs)

	X = []
	Y = []
	while True:
		im, seg = next(zipped)

		im = cv2.imread(im, 1)
		seg = cv2.imread(seg, 1)

		bounding_boxes = standalone_IoU_model_libs.get_bounding_boxes(seg)
		# neglecting any image that has more than one bounding box
		if len(bounding_boxes) > 1:
			for bounding_box in bounding_boxes:
				for new_mask, iou_score, bbox_coords in standalone_IoU_model_libs.generate_augmentations(seg,
																										 bounding_box,
																										 100, 0.1):
					if do_augment:
						im, seg[:, :, 0] = augment_seg(im, seg[:, :, 0])

					im_patch = standalone_IoU_model_libs.get_patch_of_image_with_bounding_box_coords(im, bbox_coords)

					X.append(get_image_arr(im_patch, input_width, input_height, odering=IMAGE_ORDERING))
					# Y.append(get_segmentation_arr(seg, n_classes, output_width, output_height))
					Y.append(iou_score)

					if len(X) == batch_size:
						yield np.array(X), np.array(Y)
						X = []
						Y = []
		else:
			if do_augment:
				im, seg[:, :, 0] = augment_seg(im, seg[:, :, 0])

			X.append(get_image_arr(im, input_width, input_height, odering=IMAGE_ORDERING))
			# Y.append(get_segmentation_arr(seg, n_classes, output_width, output_height))
			Y.append(0)

			if len(X) == batch_size:
				yield np.array(X), np.array(Y)
				X = []
				Y = []


def image_segmentation_generator_i3d(images_path, segs_path, batch_size, n_classes, input_height, input_width,
								 output_height, output_width, do_augment=False):
	img_seg_pairs = get_pairs_from_paths_i3d(images_path, segs_path)
	random.shuffle(img_seg_pairs)
	zipped = itertools.cycle(img_seg_pairs)

	while True:
		X = []
		Y = []
		for _ in range(batch_size):
			im, seg = next(zipped)

			im = np.load(im)
			seg = cv2.imread(seg, 1)

			# if do_augment:
			# 	img, seg[:, :, 0] = augment_seg(img, seg[:, :, 0])

			X.append(im)
			Y.append(get_segmentation_arr(seg, n_classes, output_width, output_height))

		yield np.array(X), {"main_output_activation": np.array(Y), "second_output_activation": np.array(Y)}


def two_stream_image_segmentation_generator(images_path, flows_path, segs_path, batch_size, n_classes, input_height,
											input_width, output_height, output_width, do_augment=False):
	img_seg_pairs = two_stream_get_pairs_from_paths(images_path, flows_path, segs_path)
	random.shuffle(img_seg_pairs)
	zipped = itertools.cycle(img_seg_pairs)

	while True:
		X = []
		flows_arr = []
		Y = []
		for _ in range(batch_size):
			im, flw, seg = next(zipped)

			im = cv2.imread(im, 1)
			flw = cv2.imread(flw, 1)
			seg = cv2.imread(seg, 1)

			if do_augment:
				# This was original
				# img, flow, seg[:, :, 0] = two_stream_augment_seg(img, flw, seg[:, :, 0])
				# but changed to this one, i think that was a typo
				im, flw, seg[:, :, 0] = two_stream_augment_seg(im, flw, seg[:, :, 0])

			X.append(get_image_arr(im, input_width, input_height, odering=IMAGE_ORDERING))
			flows_arr.append(get_image_arr(flw, input_width, input_height, odering=IMAGE_ORDERING))
			Y.append(get_segmentation_arr(seg, n_classes, output_width, output_height))

		yield [np.array(X), np.array(flows_arr)], np.array(Y)
