import tensorflow as tf
import numpy as np
from model_utils import stack_bottleneck, skip_block, AnchorBoxes
from tensorflow import keras
from tensorflow.keras import layers
from config import CONFIG

def panoramic_blitznet_body(images):
	""" Creates the resnet body composed by the features extractor,
	the downsampling stream and the upsampling stream with skip connections.
	Args:
		images: Input tensor to the CNN with shape [batch_size, height, width, channels].	
	Returns:
		An array of every feature map in the upsampling stream to feed both detection and segmentation heads.
	"""
	# Feature extractor (ResNet50)
	base = tf.keras.applications.resnet50.ResNet50(
		include_top=False, weights='imagenet', input_tensor=images,
    	input_shape=(512, 1024, 3), pooling=None, classes=None)
	base.trainable = False

	# Downsapmling stream
	conv6 = stack_bottleneck(base.output, 512, 2, name = 'conv6')
	conv7 = stack_bottleneck(conv6, 512, 2, name = 'conv7')
	conv8 = stack_bottleneck(conv7, 512, 2, name = 'conv8')
	conv9 = stack_bottleneck(conv8, 512, 2, name = 'conv9')

	# Upsampling stream
	rev_conv9 = layers.Conv2D(512, 1, strides = 1, padding = 'same', name = 'rev_conv9')(conv9)
	rev_conv8 = skip_block(rev_conv9, conv8, 128, name = 'rev_conv8') 
	rev_conv7 = skip_block(rev_conv8, conv7, 128, name = 'rev_conv7')
	rev_conv6 = skip_block(rev_conv7, conv6, 128, name = 'rev_conv6')
	rev_conv5 = skip_block(rev_conv6, base.output, 128, name = 'rev_conv5')
	rev_conv4 = skip_block(rev_conv5, base.get_layer('conv4_block6_out').output, 128, name = 'rev_conv4')
	rev_conv3 = skip_block(rev_conv4, base.get_layer('conv3_block4_out').output, 128, name = 'rev_conv3')
	rev_conv2 = skip_block(rev_conv3, base.get_layer('conv2_block3_out').output, 128, name = 'rev_conv2')

	features = [rev_conv2, rev_conv3, rev_conv4, rev_conv5, rev_conv6, rev_conv7, rev_conv8, rev_conv9]
	return features

def multibox_head(features):
	""" Creates the detection head.
	Args:
		features: Input array of features maps.
	Returns:	
		Output tensor of confidences and localizations for every anchor box generated.
	"""	
	localizations = []
	confidences = []
	anchorboxes = []
	
	# Get the number of anchor boxes per cell on each feature map
	n_boxes = 2 * len(CONFIG['aspect_ratios']) # 2 scales for every anchor box

	# Define two scales for each feature map between the minimum and the maximum scale
	scales = np.linspace(CONFIG['min_scale'], CONFIG['max_scale'], (len(features) -1))
	step = (CONFIG['max_scale'] - CONFIG['min_scale']) / (len(features) - 2)
	scales = np.insert(scales, 0, 0.02)
	scales = np.append(scales, (scales[-1] + step))
	
	for i in range(len(features)):

		# Predict 4 bounding box coordinates for each box, therefore the shape of localization layers is [batch_size, fm_height, fm_width, n_boxes * 4]
		loc = layers.Conv2D(n_boxes * 4, (3, 3), padding = 'same', kernel_initializer = tf.keras.initializers.VarianceScaling(scale=0.1, mode='fan_in', distribution='uniform'),  kernel_regularizer = tf.keras.regularizers.L2(0.0005), name = 'rev_conv' + str(i + 2) + '_loc')(features[i])

		# Predict n_classes confidence values for each box, so the shape of confidence layers is [batch_size, fm_height, fm_width, n_boxes * n_classes]
		conf = layers.Conv2D(n_boxes * CONFIG['num_classes'], (3, 3), padding = 'same', kernel_initializer = tf.keras.initializers.VarianceScaling(scale=0.1, mode='fan_in', distribution='uniform'),  kernel_regularizer = tf.keras.regularizers.L2(0.0005), name = 'rev_conv' + str(i + 2) + '_conf')(features[i])

		# Generate the anchor boxes. The shape of each layer is [batch_size, fm_height, fm_width, n_boxes, 4]
		anchorbox = AnchorBoxes(CONFIG['img_height'], CONFIG['img_width'], scales[i], scales[i + 1], CONFIG['aspect_ratios'], name = 'rev_conv' + str(i + 2) + '_anchor')(loc)

		# Reshape each layer to shape [batch_size, fm_height * fm_width * n_boxes, 4]
		loc = layers.Reshape((-1, 4), name = 'rev_conv' + str(i + 2) + '_loc_reshape')(loc)
		localizations.append(loc)

		# Reshape each layer to shape [batch_size, fm_height * fm_width * n_boxes, n_classes]
		conf = layers.Reshape((-1, CONFIG['num_classes']), name = 'rev_conv' + str(i + 2) + '_conf_reshape')(conf)
		confidences.append(conf)

		# Reshape each layer to shape [batch_size, fm_height * fm_width * n_boxes, 4]
		anchorbox = layers.Reshape((-1, 4), name = 'rev_conv' + str(i + 2) + '_anchor_reshape')(anchorbox)
		anchorboxes.append(anchorbox)
	
	# Concatenate every prediction from every layer
	mbox_loc = layers.Concatenate(axis = 1, name = 'concat_localizations')(localizations)
	mbox_conf = layers.Concatenate(axis = 1, name = 'concat_confidences')(confidences)
	mbox_conf = layers.Softmax(name = 'confidence_softmax')(mbox_conf)
	mbox_anchors = layers.Concatenate(axis = 1, name = 'concat_anchors')(anchorboxes)
	
	predictions = layers.Concatenate(axis = 2, name = 'detection')([mbox_conf, mbox_loc, mbox_anchors])
	return predictions		

def segmentation_head(features):
	""" Creates the segmentation head.
	Args:
		features: Input array of features maps.
	Returns:	
		segmentation: Output segmentation map with shape [batch_size, img_height/4, img_width/4, num_classes].
		Each channel contains the odds of every pixel to belong to the corresponding class (including backgorund).
	"""
	# Get the shape of the upper feature map to reshape every feature map to this size
	height, width = features[0].get_shape()[1], features[0].get_shape()[2] 

	segmentation = []
	for i in range(len(features)):
		# Map each feature map to an intermediate representation
		seg = layers.Conv2D(64, 1, padding = 'same', activation = 'relu', kernel_regularizer = tf.keras.regularizers.L2(0.0005), name = 'rev_conv' + str(i + 2) + '_seg')(features[i])
		
		# Resize each feature map to the final size
		seg = tf.image.resize(seg, [height, width], method = 'nearest', name = 'rev_conv' + str(i + 2) + '_seg_reshape')
		segmentation.append(seg)
	
	# Concatenate every segmentation layer output
	segmentation = layers.Concatenate(axis = -1, name = 'concat_segmentation')(segmentation)
	
	# Predict the class probabililty for each pixel. The output shape is [batch_size, fm_height, fm_width, num_classes]
	segmentation = layers.Conv2D(CONFIG['num_classes'], (3, 3), padding = 'same', kernel_regularizer = tf.keras.regularizers.L2(0.0005), name = 'segmentation_logits')(segmentation)
	return segmentation
	
def PanoramicBlitzNet(images):
	""" Creates the net model.
	Args:
		images: Input tensor to the CNN with shape [batch_size, height, width, channels].
	Returns:	
		Output tensor of the net (detection head + segmentation head outputs).
	"""
	features = panoramic_blitznet_body(images)
	predictions = multibox_head(features)
	segmentation = segmentation_head(features)
	return keras.Model(inputs = images, outputs = [predictions, segmentation])