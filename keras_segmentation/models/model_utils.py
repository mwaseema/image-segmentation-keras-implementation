from keras.models import *
from keras.layers import *

import keras.backend as K
from types import MethodType


from .config import IMAGE_ORDERING
from ..train import train, train_with_weighted_output, train_i3d_inception, train_temporal_with_weighted_output
from ..predict import predict, predict_multiple, evaluate, predict_with_segmentation_return, \
	predict_with_segmentation_and_probabilities_return, predict_with_weighted_output, predict_i3d_inception, \
	predict_temporal_with_weighted_output

from tqdm import tqdm

# source m1 , dest m2
def transfer_weights( m1 , m2 , verbose=True ):

	assert len( m1.layers ) == len(m2.layers) , "Both models should have same number of layers"

	nSet = 0
	nNotSet = 0

	if verbose:
		print("Copying weights ")
		bar = tqdm(zip( m1.layers, m2.layers))
	else:
		bar = zip( m1.layers, m2.layers)

	for l , ll  in bar:

		if not any([w.shape != ww.shape for w, ww in zip(list(l.weights), list(ll.weights))]):
			if len(list(l.weights)) > 0:
				ll.set_weights(l.get_weights())
				nSet += 1
		else:
			nNotSet += 1

	if verbose:
		print("Copied weights of %d layers and skipped %d layers" % (nSet, nNotSet))


def transfer_weights_zipped_layers(m1, m2, verbose=True):
	nSet = 0
	nNotSet = 0

	if verbose:
		print("Copying weights ")
		bar = tqdm(zip(m1.layers, m2.layers))
	else:
		bar = zip(m1.layers, m2.layers)

	for l, ll in bar:

		if not any([w.shape != ww.shape for w, ww in zip(list(l.weights), list(ll.weights))]):
			if len(list(l.weights)) > 0:
				ll.set_weights(l.get_weights())
				nSet += 1
		else:
			nNotSet += 1

	if verbose:
		print("Copied weights of %d layers and skipped %d layers" % (nSet, nNotSet))


def transfer_weights_single_to_two_stream(m1, m2, verbose=True):
	assert len(m1.layers) == len(m2.layers[:-2]) // 2, "Both models should have same number of layers"

	# copying weights to first stream
	nSet = 0
	nNotSet = 0

	bar = zip(m1.layers, m2.layers[0::2])
	if verbose:
		print("Copying weights to first stream")
		bar = tqdm(bar)

	for l, ll in bar:

		if not any([w.shape != ww.shape for w, ww in zip(list(l.weights), list(ll.weights))]):
			if len(list(l.weights)) > 0:
				ll.set_weights(l.get_weights())
				nSet += 1
		else:
			nNotSet += 1

	if verbose:
		print("Copied weights of %d layers and skipped %d layers for first stream" % (nSet, nNotSet))

	# copying weights to second stream
	nSet = 0
	nNotSet = 0

	bar = zip(m1.layers, m2.layers[1::2])
	if verbose:
		print("Copying weights to second stream")
		bar = tqdm(bar)

	for l, ll in bar:

		if not any([w.shape != ww.shape for w, ww in zip(list(l.weights), list(ll.weights))]):
			if len(list(l.weights)) > 0:
				ll.set_weights(l.get_weights())
				nSet += 1
		else:
			nNotSet += 1

	if verbose:
		print("Copied weights of %d layers and skipped %d layers for first stream" % (nSet, nNotSet))


def transfer_weights_single_to_two_stream_average_merge(m1, m2, verbose=True):
	assert len(m1.layers) == len(m2.layers[:-1])//2, "Both models should have same number of layers"

	# copying weights to first stream
	nSet = 0
	nNotSet = 0

	bar = zip(m1.layers, m2.layers[0::2])
	if verbose:
		print("Copying weights to first stream")
		bar = tqdm(bar)

	for l, ll in bar:

		if not any([w.shape != ww.shape for w, ww in zip(list(l.weights), list(ll.weights))]):
			if len(list(l.weights)) > 0:
				ll.set_weights(l.get_weights())
				nSet += 1
		else:
			nNotSet += 1

	if verbose:
		print("Copied weights of %d layers and skipped %d layers for first stream" % (nSet, nNotSet))

	# copying weights to second stream
	nSet = 0
	nNotSet = 0

	bar = zip(m1.layers, m2.layers[1::2])
	if verbose:
		print("Copying weights to second stream")
		bar = tqdm(bar)

	for l, ll in bar:

		if not any([w.shape != ww.shape for w, ww in zip(list(l.weights), list(ll.weights))]):
			if len(list(l.weights)) > 0:
				ll.set_weights(l.get_weights())
				nSet += 1
		else:
			nNotSet += 1

	if verbose:
		print("Copied weights of %d layers and skipped %d layers for second stream" % (nSet, nNotSet))


def resize_image( inp ,  s , data_format ):

	try:
		
		return Lambda( lambda x: K.resize_images(x, 
			height_factor=s[0], 
			width_factor=s[1], 
			data_format=data_format , 
			interpolation='bilinear') )( inp )

	except Exception as e:

		# if keras is old , then rely on the tf function ... sorry theono/cntk users . 
		assert data_format == 'channels_last'
		assert IMAGE_ORDERING == 'channels_last'

		import tensorflow as tf

		return Lambda( 
			lambda x: tf.image.resize_images(
				x , ( K.int_shape(x)[1]*s[0] ,K.int_shape(x)[2]*s[1] ))  
			)( inp )


def get_segmentation_model( input , output ):

	img_input = input
	o = output

	o_shape = Model(img_input , o ).output_shape
	i_shape = Model(img_input , o ).input_shape

	if IMAGE_ORDERING == 'channels_first':
		output_height = o_shape[2]
		output_width = o_shape[3]
		input_height = i_shape[2]
		input_width = i_shape[3]
		n_classes = o_shape[1]
		o = (Reshape((  -1  , output_height*output_width   )))(o)
		o = (Permute((2, 1)))(o)
	elif IMAGE_ORDERING == 'channels_last':
		output_height = o_shape[1]
		output_width = o_shape[2]
		input_height = i_shape[1]
		input_width = i_shape[2]
		n_classes = o_shape[3]
		o = (Reshape((   output_height*output_width , -1    )))(o)

	o = (Activation('softmax'))(o)
	model = Model(img_input, o)
	model.output_width = output_width
	model.output_height = output_height
	model.n_classes = n_classes
	model.input_height = input_height
	model.input_width = input_width
	model.model_name = ""

	model.train = MethodType(train, model)
	model.predict_segmentation = MethodType(predict, model)
	model.predict_segmentation_with_segmentation_return = MethodType(predict_with_segmentation_return, model)
	model.predict_with_segmentation_and_probabilities_return = MethodType(
		predict_with_segmentation_and_probabilities_return, model)
	model.predict_multiple = MethodType(predict_multiple, model)
	model.evaluate_segmentation = MethodType(evaluate, model)

	return model


def get_segmentation_model_i3d_inception(input, output):
	img_input = input
	o = output

	o_shape = Model(img_input, o).output_shape
	i_shape = Model(img_input, o).input_shape

	if IMAGE_ORDERING == 'channels_first':
		output_height = o_shape[2]
		output_width = o_shape[3]
		input_height = i_shape[3]
		input_width = i_shape[4]
		n_classes = o_shape[1]
		o = (Reshape((-1, output_height * output_width)))(o)
		o = (Permute((2, 1)))(o)
	elif IMAGE_ORDERING == 'channels_last':
		output_height = o_shape[1]
		output_width = o_shape[2]
		input_height = i_shape[2]
		input_width = i_shape[3]
		n_classes = o_shape[3]
		o = (Reshape((output_height * output_width, -1)))(o)

	o = (Activation('softmax'))(o)
	model = Model(img_input, o)
	model.output_width = output_width
	model.output_height = output_height
	model.n_classes = n_classes
	model.input_height = input_height
	model.input_width = input_width
	model.model_name = ""

	model.train = MethodType(train_i3d_inception, model)
	model.predict_segmentation = MethodType(predict_i3d_inception, model)

	return model


def get_segmentation_model_with_weighted_output(input, output, output2):
	img_input = input
	o = output

	o_shape = Model(img_input, o).output_shape
	i_shape = Model(img_input, o).input_shape

	if IMAGE_ORDERING == 'channels_first':
		output_height = o_shape[2]
		output_width = o_shape[3]
		input_height = i_shape[2]
		input_width = i_shape[3]
		n_classes = o_shape[1]
		o = (Reshape((-1, output_height * output_width)))(o)
		o = (Permute((2, 1)))(o)

		# for output2
		output2 = (Reshape((-1, output_height * output_width)))(output2)
		output2 = (Permute((2, 1)))(output2)
	elif IMAGE_ORDERING == 'channels_last':
		output_height = o_shape[1]
		output_width = o_shape[2]
		input_height = i_shape[1]
		input_width = i_shape[2]
		n_classes = o_shape[3]
		o = (Reshape((output_height * output_width, -1)))(o)

		# for output2
		output2 = (Reshape((output_height * output_width, -1)))(output2)

	o = (Activation('softmax', name="main_output_activation"))(o)
	output2 = (Activation('softmax', name="second_output_activation"))(output2)
	model = Model(img_input, [o, output2])
	model.output_width = output_width
	model.output_height = output_height
	model.n_classes = n_classes
	model.input_height = input_height
	model.input_width = input_width
	model.model_name = ""

	model.train = MethodType(train_with_weighted_output, model)
	model.predict_segmentation = MethodType(predict_with_weighted_output, model)
	model.predict_segmentation_with_segmentation_return = MethodType(predict_with_segmentation_return, model)
	model.predict_with_segmentation_and_probabilities_return = MethodType(
		predict_with_segmentation_and_probabilities_return, model)
	model.predict_multiple = MethodType(predict_multiple, model)
	model.evaluate_segmentation = MethodType(evaluate, model)

	return model


def get_temporal_segmentation_model_with_weighted_output(input, output, output2):
	img_input = input
	o = output

	o_shape = Model(img_input, o).output_shape
	i_shape = Model(img_input, o).input_shape

	if IMAGE_ORDERING == 'channels_first':
		output_height = o_shape[2]
		output_width = o_shape[3]
		input_height = i_shape[3]
		input_width = i_shape[4]
		n_classes = o_shape[1]
		o = (Reshape((-1, output_height * output_width)))(o)
		o = (Permute((2, 1)))(o)

		# for output2
		output2 = (Reshape((-1, output_height * output_width)))(output2)
		output2 = (Permute((2, 1)))(output2)
	elif IMAGE_ORDERING == 'channels_last':
		output_height = o_shape[1]
		output_width = o_shape[2]
		input_height = i_shape[2]
		input_width = i_shape[3]
		n_classes = o_shape[3]
		o = (Reshape((output_height * output_width, -1)))(o)

		# for output2
		output2 = (Reshape((output_height * output_width, -1)))(output2)

	o = (Activation('softmax', name="main_output_activation"))(o)
	output2 = (Activation('softmax', name="second_output_activation"))(output2)
	model = Model(img_input, [o, output2])
	model.output_width = output_width
	model.output_height = output_height
	model.n_classes = n_classes
	model.input_height = input_height
	model.input_width = input_width
	model.model_name = ""

	model.train = MethodType(train_temporal_with_weighted_output, model)
	model.predict_segmentation = MethodType(predict_temporal_with_weighted_output, model)

	# model.predict_segmentation_with_segmentation_return = MethodType(predict_with_segmentation_return, model)
	# model.predict_with_segmentation_and_probabilities_return = MethodType(
	# 	predict_with_segmentation_and_probabilities_return, model)
	# model.predict_multiple = MethodType(predict_multiple, model)
	# model.evaluate_segmentation = MethodType(evaluate, model)

	return model
