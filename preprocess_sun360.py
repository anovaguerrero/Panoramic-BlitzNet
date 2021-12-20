import tensorflow_datasets as tfds
import tensorflow as tf

from config import CONFIG
from data_augmentation import random_augmentation
from utils import resize_and_rescale

def deconstruct(record):
	""" Splits the dataset dictionary into separated tensors.
	Args:
		record: TFDS dataset.
	Returns:
		img: Image tensor of shape [image_height, image_width, channels].
		bboxes_list: Bounding boxes tensor of shape [num_objects, 2, 4].
		label: Label tensor of shape [num_objects].
		segmentation_mask: Segmentation map tensor of shape [image_height, image_width, channels].
	"""
	img = record['image']
	bboxes_list = record['objects']['bboxes_list']
	bboxes_list = bboxes_list.to_tensor(shape=[None, None, 4]) # In order to work with ragged tensors in an easier way, bboxes_list is converted into a tensor object
	label = record['objects']['label']
	segmentation_mask = record['segmentation_mask']
	return img, bboxes_list, label, segmentation_mask

def process_data(*args):
	""" Applies the required transformations to each feature tensor.
	Args:
		img: Image tensor of shape [image_height, image_width, channels].
		bboxes_list: Bounding boxes tensor of shape [num_objects, 2, 4].
		label: Label tensor of shape [num_objects].
		segmentation_mask: Segmentation map tensor of shape [image_height, image_width, channels].
	Returns:
		img: Output transformed image tensor of shape [image_height, image_width, channels].
		labels: Output transformed labels tensor of shape [num_objects, 2, 5]. Contains the 4 bounding box coordinate and the class for each object.
		segmentation_mask: Output transformed segmentation map tensor of shape [image_height, image_width, channels].
	"""
	img, bboxes_list, label, segmentation_mask = args

	# Resize and rescale the input images
	img = resize_and_rescale(img)

	# Remove the object outlines and resize and rescale the segmentation masks
	cond = segmentation_mask < 255
	segmentation_mask = tf.where(cond, segmentation_mask, 0)
	segmentation_mask = resize_and_rescale(segmentation_mask)
	
	# For feeding the model it is necessary to have the bounding box coordinates and the corresponding class concatenated together
	# Since there are 2 bounding boxes per object and only one class per object it is necessary to duplicate the label tensor
	label = tf.reshape(label, [label.shape[0], 1])
	label = tf.tile(label, [1, 2])
	label = tf.reshape(label, [label.shape[0], 2, 1])

	# Cast the label tensor from int32 to float32 in order to concatenate with the bounding boxes that are in format float32
	label = tf.cast(label, tf.float32)

	# Concatenate the bounding boxes and the labels
	labels = tf.concat([bboxes_list, label], axis = -1) 
	return img, labels, segmentation_mask

def apply_augmentation(*args):
	""" Applies the data augmentation to the images, bounding boxes and segmentation masks.
	Args:
		img: Image tensor of shape [image_height, image_width, channels].
		labels: Label tensor of shape [num_objects, 2, 5]. Contains the 4 bounding box coordinate and the class for each object.
		segmentation_mask: Segmentation map tensor of shape [image_height, image_width, channels].
	Returns:
		img: Output transformed image tensor of shape [image_height, image_width, channels].
		labels: Output transformed labels tensor of shape [num_objects, 2, 5]. Contains the 4 bounding box coordinate and the class for each object.
		segmentation_mask: Output transformed segmentation map tensor of shape [image_height, image_width, channels].
	"""				
	img, labels, segmentation_mask = args
	img, segmentation_mask, labels = random_augmentation(img, segmentation_mask, labels)

	return img, labels, segmentation_mask

def create_tuple(img, labels, segmentation_mask):
	""" Creates a tuple from each tensor to feed the model.
	Args:
		img: Image tensor of shape [image_height, image_width, channels].
		labels: Labels tensor of shape [num_objects, 2, 5] containing the bounding boxes and the category for each object.
		segmentation_mask: Segmentation map tensor of shape [image_height, image_width, channels].
	Returns:
		Tuple of tensors necessaries for feeding the model.
	"""
	# Set the shape of every tensor
	img.set_shape((CONFIG['img_height'], CONFIG['img_width'], 3))
	labels.set_shape((None, 2, 5))
	segmentation_mask.set_shape((CONFIG['img_height'], CONFIG['img_width'], 1))

	# Reshape labels from [num_objects, 2, 5] to [num_objects, 5] to remove the double bbox for every object and get a 1-D vector of non-zero bboxes
	labels = tf.reshape(labels, [-1, 5])

	# Mask for removing the zero value bboxes
	labels_mask = labels[..., -1] > 0
	labels = tf.boolean_mask(labels, labels_mask)
	mask = tf.reduce_sum(labels[..., :-1], axis = -1) > 0.0
	labels = tf.boolean_mask(labels, mask)	
	
	# Resize segmentation ground truth to a smaller size and one-hot encoding for computing the loss
	segmentation_mask = tf.image.resize(segmentation_mask, (CONFIG['img_height']//4, CONFIG['img_width']//4), method = 'nearest')
	segmentation_mask = tf.squeeze(segmentation_mask, axis = -1)
	segmentation_mask = tf.cast(segmentation_mask*255.0, tf.uint8)	
	segmentation_mask = tf.one_hot(segmentation_mask, depth = CONFIG['num_classes'])
	return (img), (labels, segmentation_mask)

def create_tuple_test(img, labels, segmentation_mask):
	""" Creates a tuple from each tensor to feed the model.
	Args:
		img: Image tensor of shape [image_height, image_width, channels].
		labels: Labels tensor of shape [num_objects, 2, 5] containing the bounding boxes and the category for each object.
		segmentation_mask: Segmentation map tensor of shape [image_height, image_width, channels].
	Returns:
		Tuple of tensors necessaries for feeding the model.
	"""
	# Set the shape of every tensor
	img.set_shape((CONFIG['img_height'], CONFIG['img_width'], 3))
	labels.set_shape((None, 2, 5))
	segmentation_mask.set_shape((CONFIG['img_height'], CONFIG['img_width'], 1))
	
	# Reshape labels from [num_objects, 2, 5] to [num_objects, 5] to remove the double bbox for every object and get a 1-D vector of non-zero bboxes
	labels = tf.reshape(labels, [-1, 5])

	# Mask for removing the zero value bboxes
	labels_mask = labels[..., -1] > 0
	labels = tf.boolean_mask(labels, labels_mask)
	mask = tf.reduce_sum(labels[..., :-1], axis = -1) > 0.0
	labels = tf.boolean_mask(labels, mask)	
	
	# Resize segmentation ground truth to a smaller size and one-hot encoding for computing the loss
	segmentation_mask = tf.image.resize(segmentation_mask, (CONFIG['img_height']//4, CONFIG['img_width']//4), method = 'nearest')
	segmentation_mask = tf.squeeze(segmentation_mask, axis = -1)
	segmentation_mask = tf.cast(segmentation_mask*255.0, tf.uint8)	
	segmentation_mask = tf.one_hot(segmentation_mask, depth = CONFIG['num_classes'])	
	return (img), (labels, segmentation_mask)

def prepare_ds(training = True):
	""" Loads and preprocess the dataset before feeding the model.
	Args:
		training: Whether it is training or inference mode.
	Returns:
		Training and validation dataset when training mode or test dataset when inference mode.
	"""
	if training:
		# Load training and validation (if required in training) datasets and preprocess both to feed the model
		train_ds = tfds.load('sun360', split = 'train')
		
		if CONFIG['use_val'] == True:
			val_ds = train_ds.skip(int((1.0 - CONFIG['val_perc']) * len(train_ds)))
			train_ds = train_ds.take(int((1.0 - CONFIG['val_perc']) * len(train_ds)))
			val_ds = val_ds.cache()
			val_ds = val_ds.map(deconstruct)
			val_ds = val_ds.map(lambda *args: tf.py_function(process_data, args, [tf.float32, tf.float32, tf.float32]))
			val_ds = val_ds.map(create_tuple)
			val_ds = val_ds.shuffle(CONFIG['buffer_size'])
			val_ds = val_ds.padded_batch(CONFIG['batch_size'], drop_remainder=True)
			val_ds = val_ds.prefetch(tf.data.experimental.AUTOTUNE)

		train_ds = train_ds.cache()
		train_ds = train_ds.map(deconstruct)
		train_ds = train_ds.map(lambda *args: tf.py_function(process_data, args, [tf.float32, tf.float32, tf.float32]))
		train_ds = train_ds.map(lambda *args: tf.py_function(apply_augmentation, args, [tf.float32, tf.float32, tf.float32]))
		train_ds = train_ds.map(create_tuple)
		train_ds = train_ds.shuffle(CONFIG['buffer_size'])
		train_ds = train_ds.padded_batch(CONFIG['batch_size'], drop_remainder=True)
		train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)

		if CONFIG['use_val'] == True:
			return train_ds, val_ds
		else:
			return train_ds

	if not training:
		# Load test dataset and preprocess it to feed the model
		test_ds = tfds.load('sun360', split = 'test')
		test_ds = test_ds.cache()
		test_ds = test_ds.map(deconstruct)
		test_ds = test_ds.map(lambda *args: tf.py_function(process_data, args, [tf.float32, tf.float32, tf.float32]))
		test_ds = test_ds.map(create_tuple_test)
		test_ds = test_ds.padded_batch(CONFIG['batch_size'])
		test_ds = test_ds.prefetch(tf.data.experimental.AUTOTUNE)
		return test_ds