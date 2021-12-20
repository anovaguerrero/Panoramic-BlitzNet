# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import numpy as np
import tensorflow as tf
from config import CONFIG
import itertools
import io
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
import matplotlib.pyplot as plt
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import confusion_matrix

def resize_and_rescale(image):
	""" Resizes and rescales an image.
	Args:
		image: Image tensor of shape [image_height, image_width, channels].
	Returns:
		image: Transformed image tensor of shape [resized_height, resized_width, channels].
	"""
	image = tf.cast(image, tf.float32)
	image = tf.image.resize(image, [CONFIG['img_height'], CONFIG['img_width']])
	image = (image / 255.0)
	return image

def draw_bounding_box_on_image(image, ymin, xmin, ymax, xmax, color, font, thickness=2, display_str_list=()):
	"""Adds a bounding box to an image. Used just for Tensorboard"""
	draw = ImageDraw.Draw(image)
	im_width, im_height = image.size
	(left, right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
	draw.line([(left, top), (left, bottom), (right, bottom), (right, top), (left, top)], width=thickness, fill=color)

	# If the total height of the display strings added to the top of the bounding
	# box exceeds the top of the image, stack the strings below the bounding box
	# instead of above.
	display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]	

	# Each display_str has a top and bottom margin of 0.05x.
	total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)	
	if top > total_display_str_height:
		text_bottom = top
	else:
		text_bottom = top + total_display_str_height

	# Reverse list and print from bottom to top.
	for display_str in display_str_list[::-1]:
		text_width, text_height = font.getsize(display_str)
		margin = np.ceil(0.05 * text_height)
		draw.rectangle([(left, text_bottom - text_height - 2 * margin), (left + text_width, text_bottom)], fill=color)
		draw.text((left + margin, text_bottom - text_height - margin), display_str, fill="black", font=font)
		text_bottom -= text_height - 2 * margin

def draw_boxes(image, boxes, class_names, scores, ID_class, max_boxes=10, min_score=0.1):
	"""Overlay labeled boxes on an image with formatted scores and label names."""
	colors = list(ImageColor.colormap.values())
	font = ImageFont.load_default()
	ID_class = tf.cast(ID_class, tf.int32)
	for i in range(min(boxes.shape[0], max_boxes)):
		if scores[i] >= min_score:
			ymin, xmin, ymax, xmax = tuple(boxes[i])
			ind = ID_class[i]
			display_str = "{}: {}%".format(class_names[ind], int(100 * scores[i]))
			color = colors[hash(class_names[ind]) % len(colors)]
			image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
			draw_bounding_box_on_image(image_pil, ymin, xmin, ymax, xmax, color, font, display_str_list=[display_str])
			np.copyto(image, np.array(image_pil))
	return image

def draw_boxes_batch(image, nms_output):
	output_images = tf.TensorArray(tf.uint8, CONFIG['batch_size'])
	for i in range(CONFIG['batch_size']):
		nms_item = nms_output[i]
		image_item = image[i]
		image_and_boxes = draw_boxes(image_item, nms_item[..., 2:], CONFIG['class_names'], nms_item[..., 1], nms_item[..., 0], max_boxes = 200)
		output_images = output_images.write(i, image_and_boxes)		

	output_images = output_images.concat()
	output_images = tf.reshape(output_images, [CONFIG['batch_size'], CONFIG['img_height'], CONFIG['img_width'], 3])   
	return output_images                    

def plot_confusion_matrix(cm, class_names):
	"""Returns a matplotlib figure containing the plotted confusion matrix.
	Args:
		cm (array, shape = [n, n]): a confusion matrix of integer classes
		class_names (array, shape = [n]): String names of the integer classes
	"""
	figure = plt.figure(figsize=(8, 8))
	norm_cm = cm / tf.math.reduce_sum(cm, axis = 1)[:, np.newaxis]
	plt.imshow(norm_cm, vmin = 0.0, vmax = 1.0, interpolation='nearest', cmap=plt.cm.Blues)
	plt.title("Confusion matrix")
	plt.colorbar()
	tick_marks = np.arange(len(class_names))
	plt.xticks(tick_marks, class_names, rotation=45)
	plt.yticks(tick_marks, class_names)

	# Compute the labels from the normalized confusion matrix.
	labels = np.around(cm / tf.math.reduce_sum(cm, axis = 1)[:, np.newaxis], decimals=2)

	# Use white text if squares are dark; otherwise black.
	threshold = 0.75
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		color = "white" if norm_cm[i, j] > threshold else "black"
		plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	return figure

def plot_to_image(figure):
	"""Converts the matplotlib plot specified by 'figure' to a PNG image and
	   returns it. The supplied figure is closed and inaccessible after this call.
	"""
	# Save the plot to a PNG in memory.
	buf = io.BytesIO()
	plt.savefig(buf, format='png')

	# Closing the figure prevents it from being displayed directly inside
	# the notebook.
	plt.close(figure)
	buf.seek(0)

	# Convert PNG buffer to TF image
	image = tf.image.decode_png(buf.getvalue(), channels=4)

	# Add the batch dimension
	image = tf.expand_dims(image, 0)
	return image  

def calculate_confusion_matrix(logits, y_true):
	# Use the model to predict the values from the validation dataset.
	predictions = np.argmax(logits, axis = -1)
	predictions = np.reshape(predictions, [-1])
	y_true = np.argmax(y_true, axis = -1)
	y_true = np.reshape(y_true, [-1])

	# Calculate the confusion matrix.
	cm = confusion_matrix.confusion_matrix(y_true, predictions, CONFIG['num_classes'], dtype = dtypes.float64)
	return cm

def calculate_IoU(cm):
	"""Calculate Intersection over Union for a given confusion matrix
	Args:
		cm: Confusion matrix of shape [num_classes, num_classes].
	"""
	# Get the diagonal of the confusion matrix which represents the true positives
	true_positives = tf.cast(tf.linalg.tensor_diag_part(cm), tf.float32)

	# Get the sum along the predicted set dimension
	pred_sum = tf.cast(tf.math.reduce_sum(cm, axis = 0), tf.float32)

	# Get the sum along the ground truth set dimension
	gt_sum = tf.cast(tf.math.reduce_sum(cm, axis = 1), tf.float32)

	# Compute the denominator
	denominator = pred_sum + gt_sum - true_positives

	# Compute the IoU
	iou = tf.math.divide_no_nan(true_positives, denominator)
	return iou

def plot_iou(iou, class_names):
	"""Returns a matplotlib figure containing the plotted confusion matrix.
	Args:
		cm (array, shape = [n, n]): a confusion matrix of integer classes
		class_names (array, shape = [n]): String names of the integer classes
	"""
	figure = plt.figure(figsize=(8, 8))
	iou_img = tf.reshape(iou, [15,1])
	plt.imshow(iou_img, interpolation='nearest', cmap=plt.cm.Blues)
	plt.title("IoU")
	plt.colorbar()
	tick_marks = np.arange(len(class_names))
	plt.yticks(tick_marks, class_names)

	# Compute the labels from the normalized confusion matrix.
	labels = np.around(iou, decimals=2)

	# Use white text if squares are dark; otherwise black.
	threshold = tf.math.reduce_max(iou) / 2.
	for i in range(iou.shape[0]):
		color = "white" if iou[i] > threshold else "black"
		plt.text(0, i, labels[i], horizontalalignment="center", color=color)

	plt.tight_layout()
	plt.ylabel('Class label')
	return figure  