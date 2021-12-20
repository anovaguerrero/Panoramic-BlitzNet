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
import tensorflow as tf
import numpy as np

from tensorflow.keras import layers
from tensorflow.keras import backend
from config import CONFIG

def bottleneck(x, filters, kernel_size = 3, stride = 1, name = None):
	""" A residual block.
	Args:
		x: Input tensor.
		filters: Integer, filters of the bottleneck layer.
		kernel_size: Default 3, kernel size of the bottleneck layer.
		stride: Default 1, stride of the first layer.
		name: String, block label.
	Returns:
		Output tensor for the residual block.
	"""
	if stride == 1:
		shortcut = x
	else:
		shortcut = layers.MaxPooling2D([1, 1], strides = stride, name = name + '_shortcut')(x)

	x = layers.Conv2D(filters, 1, strides = stride, padding = 'same', activation = 'relu', name = name + '_1_conv')(x)
	x = layers.Conv2D(filters, kernel_size, strides = 1, padding = 'same', activation = 'relu', name = name + '_2_conv')(x)
	x = layers.Conv2D(4 * filters, 1, strides = 1, padding = 'same', name = name + '_3_conv')(x)
	x = layers.Add(name = name + '_add')([shortcut, x])
	x = layers.Activation('relu', name = name + '_out')(x)
	return x

def stack_bottleneck(x, filters, blocks, stride1 = 2, name = None):
	""" A set of stacked residual blocks.
	Args:
		x: Input tensor.
		filters: Integer, filters of the bottleneck layer in a block.
		blocks: Integer, blocks in the stacked blocks.
		stride1: Default 2, stride of the first layer in the first block.
		name: String, stack label.
	Returns:
		Output tensor for the stacked blocks.
	"""
	x = bottleneck(x, filters, stride = stride1, name = name + '_block1')
	for i in range(2, blocks + 1):
		x = bottleneck(x, filters, name = name + '_block' + str(i))
	return x

def skip_block(inputs, skip_downstream, filters, name = None):
	""" A residual block with skip connection between upscale stream input and the corresponding downscale stream output.
	Args:
		x: Input tensor.
		skip_downstream: Output from the downscale stream
		filters: Integer, filters of the bottleneck layer in a block.
		kernel_size: Default 3, kernel size of the residual skip layer.
		stride: Deafult 1, stride of each residual skip layer.
		first_block: Default False, indicates the first block of the upscale stream. 
		name: String, block label.
	Returns:
		Output tensor for the stacked blocks.
	"""
	resized_input = tf.image.resize(inputs, tf.shape(skip_downstream)[1:3], method = 'bilinear', name = name + 'resize')

	x = tf.concat([resized_input, skip_downstream], 3, name = name + '_concat')
	x = layers.Conv2D(filters, 1, strides = 1, padding = 'same', activation = 'relu', name = name + '_1_conv')(x)
	x = layers.Conv2D(filters, 3, strides = 1, padding = 'same', activation = 'relu', name = name + '_2_conv')(x)
	x = layers.Conv2D(4 * filters, 1, strides = 1, padding = 'same', name = name + '_3_conv')(x)
	x = layers.Add(name = name + '_out')([resized_input, x])
	x = layers.Activation('relu', name = name + '_relu')(x)
	return x

class AnchorBoxes(layers.Layer):
	""" A Keras layer to create an output tensor containing anchor box coordinates based on the input tensor and the passed arguments.
    A set of anchor boxes of different aspect ratios is created for each spatial unit of
    the input feature map. The number of anchor boxes created per unit depends on the arguments
    aspect_ratios (by default this number is 5). The boxes are parameterized by the coordinate tuple [cx, cy, w, h].	
	"""
	def __init__(self, img_height, img_width, this_scale, next_scale, aspect_ratios, name = None):
		super(AnchorBoxes, self).__init__(name = name)
		self.img_height = img_height
		self.img_width = img_width
		self.this_scale = this_scale
		self.next_scale = next_scale
		self.aspect_ratios = aspect_ratios
		self.n_boxes = 2 * len(self.aspect_ratios)
		"""
		Args:
			img_height: The height of the input feature map.
			img_width: The width of the input feature map.
			this_scale: The scale of the actual feature map for computing the size of the generated anchor boxes.
			next_scale: The scale of the next feature map for computing the size of the generated anchor boxes.
			aspect_ratios: The list of aspect ratios of the anchor boxes to be generated for each layer.
			n_boxes: Number of generated anchor boxes per cell in each feature map.
		"""
	def call(self, x):
		# Compute the box widths and and heights for all scales and aspect ratios
		wh_list = []
		for ar in self.aspect_ratios:
			box_height = self.this_scale / np.sqrt(ar)
			box_width = self.this_scale * np.sqrt(ar)
			wh_list.append((box_width, box_height))
			box_height = np.sqrt(self.next_scale * self.this_scale) / np.sqrt(ar)
			box_width = np.sqrt(self.next_scale * self.this_scale) * np.sqrt(ar)
			wh_list.append((box_width, box_height))			
		wh_list = np.array(wh_list)

		# It is necessary the shape of the input tensor
		feature_map_height, feature_map_width = x.get_shape()[1], x.get_shape()[2]

		# Compute the grid of box center points. They are identical for all aspect ratios	
		# Compute the offsets, i.e. at what pixel values the first anchor box point will be from the top and from the left of the image	
		offset_height = 0.5
		offset_width = 0.5
		
		# Compute the grid of anchor box center points
		cy = np.linspace(offset_height / feature_map_height, 1 - offset_height / feature_map_height, feature_map_height, endpoint = True) # Relative coordinates in range [0, 1]
		cx = np.linspace(offset_width / feature_map_width, 1 - offset_width / feature_map_width, feature_map_width, endpoint = True) # Relative coordinates in range [0, 1]
		x_grid, y_grid = np.meshgrid(cx, cy)
		x_grid = np.expand_dims(x_grid, -1) # This is necessary for np.tile() further down
		y_grid = np.expand_dims(y_grid, -1) # This is necessary for np.tile() further down

		# Create a 4D tensor template of shape [feature_map_height, feature_map_width, n_boxes, 4] where the last dimension will contain [cx, cy, w, h]
		boxes_tensor = np.zeros((feature_map_height, feature_map_width, self.n_boxes, 4))

		boxes_tensor[:, :, :, 0] = np.tile(x_grid, (1, 1, self.n_boxes)) # Set cx
		boxes_tensor[:, :, :, 1] = np.tile(y_grid, (1, 1, self.n_boxes)) # Set cy
		boxes_tensor[:, :, :, 2] = wh_list[:, 0] # Set w
		boxes_tensor[:, :, :, 3] = wh_list[:, 1] # Set h

		# Prepend one dimension to boxes_tensor to account for the batch size and tile it along
		# The result will be a 5D tensor of shape [batch_size, feature_map_height, feature_map_width, n_boxes, 4]
		boxes_tensor = np.expand_dims(boxes_tensor, axis = 0)
		boxes_tensor = backend.tile(backend.constant(boxes_tensor, dtype = 'float32'), (backend.shape(x)[0], 1, 1, 1, 1))
		return boxes_tensor
		
def batch_iou(gt_bboxes, anchor_bboxes):
	""" Computes the IoU between each anchor bounding box and ground truth bounding box.
	Args:
		gt_bboxes: Tensor of ground truth bounding boxes with shape [num_gt_bboxes, 4]. Coordinates are [ymin, xmin, ymax, xmax].
		anchor_bboxes: Tensor of anchor bounding boxes with shape [num_anchor_bboxes, 4]. Coordinates are [cx, cy, w, h].
	Returns:
		iou: Tensor of IoU between every bounding box with shape [num_anchor_bboxes, num_gt_bboxes].
	"""
	n_anchor_boxes = tf.shape(anchor_bboxes)[0]

	# First of all reshape and transpose gt_bboxes tensor and anchor_bboxes tensor to create the iou tensor with the correct shape
	gt_bboxes = tf.reshape(tf.transpose(gt_bboxes), [4, 1, -1])
	anchor_bboxes = tf.reshape(tf.transpose(anchor_bboxes), [4, -1, 1])
		
	# Transform anchor_bboxes coordinates from centroids to corners
	a_ymin = anchor_bboxes[1] - anchor_bboxes[3] / 2
	a_xmin = anchor_bboxes[0] - anchor_bboxes[2] / 2
	a_ymax = anchor_bboxes[1] + anchor_bboxes[3] / 2
	a_xmax = anchor_bboxes[0] + anchor_bboxes[2] / 2
		
	# Compute the intersection corners
	ymin = tf.math.maximum(gt_bboxes[0], a_ymin)
	xmin = tf.math.maximum(gt_bboxes[1], a_xmin)
	ymax = tf.math.minimum(gt_bboxes[2], a_ymax)
	xmax = tf.math.minimum(gt_bboxes[3], a_xmax)
		
	# Calculate the intersection area 
	intersection_width = tf.math.maximum(0.0, (xmax - xmin))
	intersection_height = tf.math.maximum(0.0, (ymax - ymin))
	intersection = intersection_width * intersection_height
		
	# Calculate the union area between boxes and the IoU 
	union = (gt_bboxes[2] - gt_bboxes[0]) * (gt_bboxes[3] - gt_bboxes[1]) + anchor_bboxes[2] * anchor_bboxes[3] - intersection
	iou = intersection / union
	iou = tf.reshape(iou, [n_anchor_boxes, -1])	
	return iou

def decode_bboxes(localization, anchors):
	""" Converts anchor box offsets to predicted bounding box coordinates.
	Args:
		localization: Tensor of localization coordinates with shape [batch_size, num_gt_bboxes, 4]. Coordinates are [cx, cy, w, h].
		anchors: Tensor of anchor bounding boxes with shape [batch_size, num_anchor_bboxes, 4]. Coordinates are [cx, cy, w, h].
	Returns:
		Output tensor of converted coordinates from anchor boxes offsets to predicted bounding boxes coordinates with shape [batch_size, num_anchor_bboxes, 4].
		Coordinates are [ymin, xmin, ymax, xmax]
	"""
	var_cx, var_cy, var_w, var_h = CONFIG['variances']

	# Get prediction offsets
	l_cx = localization[:, :, 0]*var_cx
	l_cy = localization[:, :, 1]*var_cy
	l_w = localization[:, :, 2]*var_w
	l_h = localization[:, :, 3]*var_h

	# Get anchor boxes coordinates
	a_cx = anchors[:, :, 0]
	a_cy = anchors[:, :, 1]
	a_w = anchors[:, :, 2]
	a_h = anchors[:, :, 3]

	# Calculate predicted boxes coordinates
	cx = l_cx*a_w + a_cx
	cy = l_cy*a_h + a_cy
	w = tf.exp(l_w)*a_w
	h = tf.exp(l_h)*a_h

	xmin = tf.math.maximum(0., cx - w/2)
	ymin = tf.math.maximum(0., cy - h/2)
	xmax = tf.math.minimum(1., w + xmin)
	ymax = tf.math.minimum(1., h + ymin)
	return tf.stack([ymin, xmin, ymax, xmax], axis = -1)

def NMS_fc(predictions):
	""" Performs Non-Maximum Supression to predictions to obtain final bounding boxes during inference.
	Args:
		predictions: Tensor of predicted data with shape [batch_size, num_anchor_bboxes, num_classes + 8]. Last axis parameters are the odds for each anchor box to belong to 
		each class, 4 parameters from localization and 4 parameters from anchor boxes.
	Returns:
		Output tensor of final predictions with shape [batch_size, num_detected_bboxes, 6]. Last axis parameters are the following: 1 from class ID, 1 from class odds
		and 4 from bounding box coordinates [ymin, xmin, ymax, xmax].
	"""	
	converted_bboxes = decode_bboxes(predictions[..., -8:-4], predictions[..., -4:])
	predictions = tf.concat([predictions[...,:-8], converted_bboxes], axis=-1)

	def filter_predictions(batch_item):
		def filter_single_class(index):
			# From a tensor of shape (n_boxes, n_classes + 4 coordinates) extract a tensor of shape (n_boxes, 1 + 4 coordinates) that contains the
			# confidence values for just one class, determined by `index`
			confidences = tf.expand_dims(batch_item[..., index], axis = -1)
			class_id = tf.fill(tf.shape(confidences), tf.cast(index, tf.float32))
			box_coordinates = batch_item[..., -4:]

			single_class = tf.concat([class_id, confidences, box_coordinates], axis= - 1)

			# Apply confidence thresholding with respect to the class defined by `index`
			threshold_mask = single_class[:, 1] > CONFIG['confidence_th']
			single_class = tf.boolean_mask(single_class, threshold_mask)

			k = tf.math.minimum(tf.shape(single_class)[0], CONFIG['top_k_nms'])
			_, top_k_inds = tf.math.top_k(single_class[..., 1], k)
			top_single_class = tf.gather(single_class, top_k_inds)
				
			# If any boxes made the threshold, perform NMS
			def perform_nms(): 
				scores = top_single_class[..., 1]
				boxes = top_single_class[..., -4:]

				maxima_indices = tf.image.non_max_suppression(boxes, 
															scores,
															max_output_size = CONFIG['nms_max_output_size'], 
															iou_threshold = CONFIG['nms_th'])
				maxima = tf.gather(top_single_class, maxima_indices, axis = 0)
				return maxima
				
			def no_confident_predictions():
				return tf.constant(0.0, shape = (1, 6))

			single_class_nms = tf.cond(tf.math.equal(tf.size(top_single_class), 0), no_confident_predictions, perform_nms)
		
			padded_single_class = tf.pad(single_class_nms,
										paddings=[[0, CONFIG['nms_max_output_size'] - tf.shape(single_class_nms)[0]], [0, 0]],
										mode = 'constant', 
										constant_values = 0.0)
			return padded_single_class
			
		filtered_single_classes = tf.nest.map_structure(tf.stop_gradient, tf.map_fn(lambda i: filter_single_class(i),
											tf.range(1, CONFIG['num_classes']),
											dtype = tf.float32))
		filtered_predictions = tf.reshape(filtered_single_classes, [-1, 6])
		return filtered_predictions
		
	output_tensor = tf.nest.map_structure(tf.stop_gradient, tf.map_fn(lambda x: filter_predictions(x), predictions))
	return output_tensor	

def compute_offsets(gt_bboxes, anchors, CONFIG):
	""" Computes the offstes between matched anchor boxes and ground truth bounding boxes.
	Args:
		gt_bboxes: A tensor of ground truth bounding boxes with shape [batch_size, num_anchor_bboxes, 4].
		anchors: Tensor of anchor boxes with shape [batch_size, num_anchor_bboxes, 4].
	Returns:	
		offsets: A tensor of ground truth offsets with shape [batch_size, num_anchor_bboxes, 4].
	"""	
	# Get the parameters from the ground truth bounding boxes	
	gt_ymin = gt_bboxes[:, :, 0]
	gt_xmin = gt_bboxes[:, :, 1]
	gt_ymax = gt_bboxes[:, :, 2]
	gt_xmax = gt_bboxes[:, :, 3]
	
	# Get the parameters from the anchor boxes
	anchor_cx = anchors[:, :, 0]
	anchor_cy = anchors[:, :, 1]
	anchor_w = anchors[:, :, 2]
	anchor_h = anchors[:, :, 3]
	
	# Compute for each parameter the difference between anchor and ground truth
	obs_cx = (gt_xmin + 0.5 * (gt_xmax - gt_xmin) - anchor_cx) / anchor_w # cx
	obs_cy = (gt_ymin + 0.5 * (gt_ymax - gt_ymin) - anchor_cy) / anchor_h # cy
	
	gt_w = gt_xmax - gt_xmin
	gt_w = tf.math.maximum(gt_w, 1e-8)
	obs_w = tf.math.log(gt_w / anchor_w) # w

	gt_h = gt_ymax - gt_ymin
	gt_h = tf.math.maximum(gt_h, 1e-8)
	obs_h = tf.math.log(gt_h / anchor_h) # h

	var_cx, var_cy, var_w, var_h = CONFIG['variances']
	offsets = tf.stack([obs_cx / var_cx, obs_cy / var_cy, obs_w / var_w, obs_h / var_h], axis = -1)
	return offsets	

def matching_fc(gt_data, anchor_boxes):
	""" Matches anchor bboxes with ground truth bboxes.
	Args:
		gt_data: Tensor of ground truth data with shape [batch_size, num_gt_bboxes, 5]. It contains bounding boxes coordinates and categories.
		anchor_bboxes: Tensor of anchor boxes with shape [batch_size, num_anchor_bboxes, 4].
	Returns:
		matched_data: Output tensor with shape [batch_size, num_anchor_bboxes, 6]. First 4 parameters from ground truth offset, 1 parameter from category for each anchor box
		and 1 parameter indicating whether the matched anchor box is positive or negative.
	"""		
	# Get bounding boxes and categories from 'gt_data'
	gt_bboxes, gt_cats = gt_data[..., :-1], gt_data[..., -1]

	pos_mask = tf.TensorArray(tf.float32, CONFIG['batch_size'])
	bboxes = tf.TensorArray(tf.float32, CONFIG['batch_size'])
	cats = tf.TensorArray(tf.float32, CONFIG['batch_size'])	
	for i in range(CONFIG['batch_size']):
		num_gt = tf.cast(tf.shape(gt_cats[i])[0], tf.int32)
		num_anchors = anchor_boxes[i].get_shape()[0]

		# source for tiling if nothing matches
		positive_matches_mask = tf.zeros(num_anchors, dtype=tf.float32)
		cats_gather = tf.zeros(num_anchors, dtype=tf.float32)
		gt_bboxes_gather = tf.ones((num_anchors, 4), dtype=tf.float32)

		if num_gt > 0: # Avoid no gt data
			# Calculate th IoU between anchor boxes and gt bboxes
			iou = batch_iou(gt_bboxes[i], anchor_boxes[i])

			# Calculate for each anchor box the ground truth bounding box that matches the best
			best_gt_inds = tf.cast(tf.math.argmax(iou, axis = 1), tf.int32)
			anchor_inds = tf.range(num_anchors)
			gt_inds = tf.range(num_gt)
			scatter_inds = tf.stack([anchor_inds, best_gt_inds], axis = 1)
			scatter_values = tf.ones_like(best_gt_inds, dtype = tf.int32)	
			pos_matches_anchor_gt = tf.scatter_nd(scatter_inds, scatter_values, tf.shape(iou)) * tf.cast(iou >= CONFIG['iou_threshold'], tf.int32) # Remove false positives

			# Calculate for each ground truth bounding box the anchor box that matches the best
			best_anchor_inds = tf.cast(tf.math.argmax(iou, axis = 0), tf.int32)
			scatter_inds = tf.stack([best_anchor_inds, gt_inds], axis = 1)
			scatter_values = tf.ones_like(best_anchor_inds, dtype = tf.int32)				
			pos_matches_gt_anchor = tf.scatter_nd(scatter_inds, scatter_values, tf.shape(iou)) * tf.cast(iou > 0.0, tf.int32) # In case there is any gt with value (0.0, 0.0, 0.0, 0.0)

			# By doing this matching it is possible that one anchor box is being matched with more than 1 gt bbox so we need to let at maximum 1 match per anchor box
			# We just select the first matched gt bbox regardless which match has higher IoU
			unique_gt_inds = tf.cast(tf.math.argmax(pos_matches_gt_anchor, axis = 1), tf.int32)

			# Get a mask with positive matches
			unique_matches_mask = tf.math.reduce_sum(pos_matches_gt_anchor, axis = 1) > 0

			# Apply the mask and get the unique positive matches
			unique_gt_inds = tf.boolean_mask(unique_gt_inds, unique_matches_mask)
			unique_anchor_inds = tf.boolean_mask(anchor_inds, unique_matches_mask)
			scatter_inds = tf.stack([unique_anchor_inds, unique_gt_inds], axis = 1)
			scatter_values = tf.ones_like(unique_gt_inds)
			pos_matches_unique_gt_anchor = tf.scatter_nd(scatter_inds, scatter_values, tf.shape(iou))

			# For computing the final matching matrix first we use the matched gt bboxes with their respective anchor box and then the remaining anchor boxes with their 
			# respective gt bbox
			cond = tf.math.reduce_sum(pos_matches_gt_anchor, axis = 1) > 0
			cond = tf.reshape(cond, [tf.shape(cond)[0], 1])
			positive_matches = tf.where(cond, pos_matches_unique_gt_anchor, pos_matches_anchor_gt)	
			
			# For computing the output, it is necesary to get the ground truth index that matches with each anchor box and which anchor boxes are positive matches
			gt_matches_inds = tf.math.argmax(positive_matches, axis = 1)
			positive_matches_mask = tf.cast((tf.math.reduce_sum(positive_matches, axis = 1) > 0), tf.float32)

			# Get ground truth bounding boxes and category for every matched anchor box 
			gt_bboxes_gather = tf.gather(params = gt_bboxes[i], indices = gt_matches_inds)
			cats_gather = tf.gather(params = gt_cats[i], indices = gt_matches_inds)
			
			# To calculate confidence loss, negative matches are used with their odds to be 'background' so in those cases we match them with category 0 (background)
			# to make easier the calculation of loss function
			cats_gather = tf.where(tf.cast(positive_matches_mask, tf.bool), cats_gather, 0.0)		

		pos_mask = pos_mask.write(i, positive_matches_mask)
		cats = cats.write(i, cats_gather)
		bboxes = bboxes.write(i, gt_bboxes_gather)

	pos_mask = pos_mask.concat()
	bboxes = bboxes.concat()
	cats = cats.concat()	

	# Reshape outputs to a correct shape
	pos_match_mask = tf.reshape(pos_mask, [CONFIG['batch_size'], -1, 1])
	bboxes = tf.reshape(bboxes, [CONFIG['batch_size'], -1, 4])
	cats = tf.reshape(cats, [CONFIG['batch_size'], -1, 1])			

	# Calculate the ground truth offsets relative to anchor boxes
	gt_offsets = compute_offsets(bboxes, anchor_boxes, CONFIG)

	# Concatenate the outputs
	matched_data = tf.concat([gt_offsets, cats, pos_match_mask], axis = 2)	
	return matched_data