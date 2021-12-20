import tensorflow as tf
import tensorflow_addons as tfa

from config import CONFIG

def random_h_flip(image, gt_sgm_map, gt_bboxes):
	""" Applies random horizontal flip.
	Args:
		image: An image tensor of shape [image_heigth, image_width, channels].
		gt_seg_map: A segmentation map tensor of shape [image_heigth, image_width, channels].
		gt_bboxes: A bounding boxes tensor of shape [num_bboxes, 2, 4].
	Returns:
		image: Output flipped image tensor of shape [image_heigth, image_width, channels].
		gt_seg_map: Output flipped segmentation map tensor of shape [image_heigth, image_width, channels].
		gt_bboxes: Output flipped bounding boxes tensor of shape [num_bboxes, 2, 4].
	"""
	image = tf.image.flip_left_right(image)
	gt_sgm_map = tf.image.flip_left_right(gt_sgm_map)

	# Unstack bboxes_list tensor of shape (None, 2, 4) to obtain 2 bboxes tensors of shape (None, 4)
	bbox_1, bbox_2 = tf.unstack(gt_bboxes, axis = 1)

	# Unstack bbox_1 tensor to obtain the coordiantes of each object bbox
	y1min, x1min, y1max, x1max = tf.unstack(bbox_1, axis = 1) 

	# Unstack bbox_2 tensor just like bbox_1
	y2min, x2min, y2max, x2max = tf.unstack(bbox_2, axis = 1) 

	# In this case the main bbox is stored in bbox_1 and it corresponds with the bbox nearest to the left side, hence for keeping this, 
	# the flipped bbox_1 will be nearest to the right side and will be stored in bbox_2 in every case the object has two bboxes. If not, it will be stored in bbox_1
	# Convert xmin and xmax tensors to tf.Variable in order to calculate the flipped bboxes
	x1min, x1max = tf.Variable(x1min), tf.Variable(x1max) 
	x2min, x2max = tf.Variable(x2min), tf.Variable(x2max)
	for i in range(len(gt_bboxes)):
		if x2min[i] != x2max[i]: # Only the objects with two bboxes need to be flipped. The rest of obejcts have a second bbox with values (0.0, 0.0, 0.0, 0.0)
			aux_x2max = 1 - x1min[i] # Auxiliar variable
			aux_x2min = 1 - x1max[i]
			
			x1min[i].assign(1 - x2max[i])
			x1max[i].assign(1 - x2min[i])
			x2min[i].assign(aux_x2min)
			x2max[i].assign(aux_x2max)				
			
		else:
			aux_x1max = 1 - x1min[i] # Auxiliar variable
			aux_x1min = 1 - x1max[i]			
			x1min[i].assign(aux_x1min)
			x1max[i].assign(aux_x1max)

	# Convert tf.Variable back to tensor 			
	x1min = tf.convert_to_tensor(x1min)  
	x1max = tf.convert_to_tensor(x1max)
	x2min = tf.convert_to_tensor(x2min)
	x2max = tf.convert_to_tensor(x2max)

	# Stack the rotated coordinates to obtain a bbox tensor of shape (None, 4)	
	bbox_1 = tf.stack([y1min, x1min, y1max, x1max], axis = 1)
	bbox_2 = tf.stack([y2min, x2min, y2max, x2max], axis = 1)
	
	# Stack the two bboxes tensors to obtain a bboxes_list tensor of shape (None, 2, 4) as the original one
	gt_bboxes = tf.stack([bbox_1, bbox_2], axis = 1)
	return image, gt_sgm_map, gt_bboxes

def random_color(image, color_ordering):
	""" Applies random color changes to an image.
	Args:
		image: An image tensor of shape [image_heigth, image_width, channels].
		
	Returns:
		image: Output changed image tensor of shape [image_heigth, image_width, channels].
	"""
	if color_ordering == 0:
		image = tf.image.random_brightness(image, CONFIG['max_brightness'])
		image = tf.image.random_saturation(image, CONFIG['lower_sat'], CONFIG['upper_sat'])
		image = tf.image.random_hue(image, CONFIG['max_hue'])		
		image = tf.image.random_contrast(image, CONFIG['lower_contrast'], CONFIG['upper_contrast'])
	elif color_ordering == 1:
		image = tf.image.random_saturation(image, CONFIG['lower_sat'], CONFIG['upper_sat'])
		image = tf.image.random_brightness(image, CONFIG['max_brightness'])	
		image = tf.image.random_contrast(image, CONFIG['lower_contrast'], CONFIG['upper_contrast'])
		image = tf.image.random_hue(image, CONFIG['max_hue'])	
	elif color_ordering == 2:
		image = tf.image.random_contrast(image, CONFIG['lower_contrast'], CONFIG['upper_contrast'])
		image = tf.image.random_hue(image, CONFIG['max_hue'])
		image = tf.image.random_brightness(image, CONFIG['max_brightness'])
		image = tf.image.random_saturation(image, CONFIG['lower_sat'], CONFIG['upper_sat'])
	elif color_ordering == 3:	
		image = tf.image.random_hue(image, CONFIG['max_hue'])
		image = tf.image.random_saturation(image, CONFIG['lower_sat'], CONFIG['upper_sat'])
		image = tf.image.random_contrast(image, CONFIG['lower_contrast'], CONFIG['upper_contrast'])
		image = tf.image.random_brightness(image, CONFIG['max_brightness'])
	return image

def h_shift(image, gt_sgm_map, gt_bboxes):
	""" Applies horizontal rotation on the sphere with a random angle.
	Args:
		image: An image tensor of shape [image_heigth, image_width, channels].
		gt_seg_map: A segmentation map tensor of shape [image_heigth, image_width, channels].
		gt_bboxes: A bounding boxes tensor of shape [num_bboxes, 2, 4].
	Returns:
		image: Output rotated image tensor of shape [image_heigth, image_width, channels].
		gt_seg_map: Output rotated segmentation map tensor of shape [image_heigth, image_width, channels].
		gt_bboxes: Output rotated bounding boxes tensor of shape [num_bboxes, 2, 4].
	"""
	# Get the number of pixels to rotate the image depending the rotation angle
	h_shift_angle = tf.random.uniform([], 0, 360, dtype = tf.int64)
	h_shift_len = int(tf.math.round(h_shift_angle * CONFIG['img_width'] / 360))
	
	# Divide image in two sub-images and reassemble in order to get horizontal rotation
	left_img = tf.slice(image, [0, 0, 0], [CONFIG['img_height'], h_shift_len, 3])
	right_img = tf.slice(image, [0, h_shift_len, 0], [CONFIG['img_height'], CONFIG['img_width'] - h_shift_len, 3])
	img = tf.concat([right_img, left_img], axis = 1)
	
	left_sgm_map = tf.slice(gt_sgm_map, [0, 0, 0], [CONFIG['img_height'], h_shift_len, 1])
	right_sgm_map = tf.slice(gt_sgm_map, [0, h_shift_len, 0], [CONFIG['img_height'], CONFIG['img_width'] - h_shift_len, 1])
	gt_sgm_map = tf.concat([right_sgm_map, left_sgm_map], axis = 1)
	
	# Unstack bboxes_list tensor of shape (None, 2, 4) to obtain 2 bboxes tensors of shape (None, 4)
	bbox_1, bbox_2 = tf.unstack(gt_bboxes, axis = 1)

	# Unstack bbox_1 tensor to obtain the coordiantes of each object bbox
	y1min, x1min, y1max, x1max = tf.unstack(bbox_1, axis = 1)

	# Unstack bbox_2 tensor just like bbox_1	
	y2min, x2min, y2max, x2max = tf.unstack(bbox_2, axis = 1)
	
	# Calculate the rotated first bbox for every object
	n_x1min = x1min - h_shift_len / CONFIG['img_width'] 
	n_x1max = x1max - h_shift_len / CONFIG['img_width']	

	# Create a copy of the second bbox for calculating the rotated bbox
	n_x2min = x2min 
	n_x2max = x2max

	# Convert bboxes coordinates tensors to tf.Variable in order to calculate the rotated bboxes	
	x2min, x2max = tf.Variable(x2min), tf.Variable(x2max)
	y1min, y1max = tf.Variable(y1min), tf.Variable(y1max)
	y2min, y2max = tf.Variable(y2min), tf.Variable(y2max)
	# Convert rotated xmin and xmax tensors to tf.Variable in order to compare with the original bboxes and define the new bbox or new bboxes for every object
	n_x1min, n_x1max = tf.Variable(n_x1min), tf.Variable(n_x1max)
	n_x2min, n_x2max = tf.Variable(n_x2min), tf.Variable(n_x2max)

	for i in range(len(gt_bboxes)):
		# Calculate the rotated second bbox for the objects that have it (Objects with no second bbox have a (0, 0, 0, 0) vector indicating it)
		if x2min[i] != x2max[i]: 
			n_x2min[i].assign(x2min[i] - h_shift_len/ CONFIG['img_width'])
			n_x2max[i].assign(x2max[i] - h_shift_len/ CONFIG['img_width'])

		# If any rotated coordinate is negative it is converted to positive	
		if n_x1min[i] < 0.0:
			n_x1min[i].assign(1 + n_x1min[i])
		if n_x1max[i] < 0.0:
			n_x1max[i].assign(1 + n_x1max[i])
		if n_x2min[i] < 0.0:
			n_x2min[i].assign(1 + n_x2min[i])
		if n_x2max[i] < 0.0:
			n_x2max[i].assign(1 + n_x2max[i])
	
	# There are 4 different cases: 1) Object not cropped both before and after rotation, 2) Object not cropped before rotation but cropped after it,
	# 3) Object cropped both before and after rotation, 4) Object cropped before rotation but not cropped after it.
	# In this case for an object with two different bboxes, the one that is nearest to the left side of the image belongs to bbox_1 and 
	# the other one belongs to bbox_2 so the calculations will be done according to this
	for i in range(len(gt_bboxes)):
		# Case 1)
		if x2min[i] == x2max[i] and n_x1min[i] < n_x1max[i]: 
			pass

		# Case 2) From the original bbox it is obtained a second bbox with the same 'y' coordinates than the original	
		elif x2min[i] == x2max[i] and n_x1min[i] > n_x1max[i]: 
			n_x2min[i].assign(n_x1min[i])
			n_x2max[i].assign(1.0)
			y2min[i].assign(y1min[i])
			y2max[i].assign(y1max[i])
			n_x1min[i].assign(0.0)
			
		# Case 3a) Bbox_1 ends up on the left of bbox_2
		elif x2min[i] != x2max[i] and n_x1min[i] > n_x1max[i]: 
			n_x1min[i].assign(0.0)
			n_x2max[i].assign(1.0)
		
		# Case 3b) Bbox_1 ends up on the right of bbox_2
		elif x2min[i] != x2max[i] and n_x2min[i] > n_x2max[i]: 
			n_x1min[i].assign(0.0)
			n_x2max[i].assign(1.0)

		# Case 4) From two bboxes it is obtained just one bbox so it is necessary to remove the original second one				
		elif x2min[i] != x2max[i] and n_x1min[i] < n_x1max[i]:
			n_x1min[i].assign(n_x2min[i])
			n_x2min[i].assign(0.0)
			n_x2max[i].assign(0.0)
			y2min[i].assign(0.0)
			y2max[i].assign(0.0)
	
	# Convert tf.Variable back to tensor
	n_x1min = tf.convert_to_tensor(n_x1min)
	n_x1max = tf.convert_to_tensor(n_x1max)
	y1min = tf.convert_to_tensor(y1min)
	y1max = tf.convert_to_tensor(y1max)
	n_x2min = tf.convert_to_tensor(n_x2min) 
	n_x2max = tf.convert_to_tensor(n_x2max)
	y2min = tf.convert_to_tensor(y2min)
	y2max = tf.convert_to_tensor(y2max)
	
	# Stack the rotated coordinates to obtain a bbox tensor of shape (None, 4)
	bbox_1 = tf.stack([y1min, n_x1min, y1max, n_x1max], axis = 1) 
	bbox_2 = tf.stack([y2min, n_x2min, y2max, n_x2max], axis = 1)
	
	# Stack the two bboxes tensors to obtain a bboxes_list tensor of shape (None, 2, 4) as the original one
	gt_bboxes = tf.stack([bbox_1, bbox_2], axis = 1)
	return img, gt_sgm_map, gt_bboxes

def clip_by_value(image, gt_sgm_map):
	""" Clips pixels value between 0 and 1.
	Args:
		image: An image tensor of shape [image_heigth, image_width, channels].
		gt_seg_map: A segmentation map tensor of shape [image_heigth, image_width, channels].
	Returns:
		image: Output clipped image tensor of shape [image_heigth, image_width, channels].
		gt_seg_map: Output clipped segmentation map tensor of shape [image_heigth, image_width, channels].
	"""
	image = tf.clip_by_value(image, 0., 1.)
	gt_sgm_map = tf.clip_by_value(gt_sgm_map, 0., 1.)
	return image, gt_sgm_map

def random_augmentation(image, gt_sgm_map, labels):
	""" Applies random data agumentation to each image.
	Args:
		image: An image tensor of shape [image_heigth, image_width, channels].
		gt_seg_map: A segmentation map tensor of shape [image_heigth, image_width, channels].
		labels: Labels tensor of shape [num_objects, 2, 5] containing the bounding boxes and the category for each object.	
	Returns:
		image: Output augmentated image tensor of shape [image_heigth, image_width, channels].
		gt_seg_map: Output augmentated segmentation map tensor of shape [image_heigth, image_width, channels].
		labels: Output augmentated labels tensor of shape [num_bboxes, 2, 5] containing the bounding boxes and the category for each object.
	"""
	# Get sliced gt bounding boxes and categories for every object
	gt_bboxes = labels[:, :, :-1]
	gt_cats = tf.reshape(labels[:, :, -1], [labels.shape[0], labels.shape[1], 1])

	# Perform data augmentation
	rng = tf.random.uniform(shape = [], minval = 0., maxval = 1.)
	color_ordering = tf.random.uniform(shape = [], minval = 0, maxval = 4, dtype = tf.int32)
	if rng <= CONFIG['color_th'] and CONFIG['color_change']:
		image = random_color(image, color_ordering)	
	if rng <= CONFIG['h_flip_th'] and CONFIG['h_flip']:
		image, gt_sgm_map, gt_bboxes = random_h_flip(image, gt_sgm_map, gt_bboxes)
	if rng <= CONFIG['h_rot_th'] and CONFIG['h_rot']:
		image, gt_sgm_map, gt_bboxes = h_shift(image, gt_sgm_map, gt_bboxes)
	image, gt_sgm_map = clip_by_value(image, gt_sgm_map)

	# Concatenate bounding boxes an categories in one tensor.
	labels = tf.concat([gt_bboxes, gt_cats], axis = -1)
	return image, gt_sgm_map, labels
