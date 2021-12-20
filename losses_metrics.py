import tensorflow as tf
from config import CONFIG

def detection_objective(matched_predictions, raw_predictions):
	""" Computes the detection loss.
	Args:
		raw_predictions: A tensor of predicted data of shape [batch_size, num_anchor_bboxes, num_classes + 8], where the last axis parameters are the following:
		first 'num_classes' parameters from confidence, 4 parameters from localizations and 4 parameters from anchor boxes.
		matched_predictions: A tensor of shape [batch_size, num_anchor_bboxes, 6], where the last axis parameters are the following:
		first 4 parameters from ground truth bboxes offsets calculated during matching strategy, 1 parameter from the category from each ground truth bbox and
		1 parameter from '1' or '0' indicator if matched anchor box is positive or negative.
	Returns:
		detection_loss: A float representing the detection_loss.
		localization_loss: A float representing the localization_loss.
		confidence_loss: A float representing the confidence_loss.
		train_acc: A float representing the metric used to evaluate detection during training.
	"""
	# Get the necessary parameters from matched_predictions and raw_predictions
	confidence = raw_predictions[:, :, :-8]
	localization = raw_predictions[:, :, -8:-4]
	positive_matches_mask = matched_predictions[:, :, -1]
	gt_offsets = matched_predictions[:, :, :-2]
	gt_cats = matched_predictions[:, :, -2]
	# Since gt_cats are concatenated with gt_offstes in matched_predictions is necessary reconvert it from float to integer	
	gt_cats = tf.cast(gt_cats, tf.int32)
	
	# # Get the ground truth bboxes offsets and categories for the matched anchor boxes
	gt_offsets = tf.reshape(gt_offsets, [CONFIG['batch_size'], -1, 4])
	gt_cats = tf.reshape(gt_cats, [CONFIG['batch_size'], -1])
	cats_one_hot = tf.one_hot(tf.reshape(gt_cats, [CONFIG['batch_size'], -1]), CONFIG['num_classes'])

	# Compute the confidence loss
	# Calculate the number of positive matches
	number_of_positives = tf.cast(tf.math.reduce_sum(positive_matches_mask), tf.int32)

	# Cast the positive matches mask from int32 to bool
	positive_matches_mask = tf.reshape(tf.cast(positive_matches_mask, tf.bool), [CONFIG['batch_size'], -1])

	# Compute the negative matches mask
	negative_matches_mask = tf.reshape(tf.math.logical_not(positive_matches_mask), [CONFIG['batch_size'], -1])

	# Get the number of negative matches to compute the loss. In this case, for not creating a huge inbalance between positive and negative matches,
	# the number of negatives to use is the minimum between the true number of negative matches and 3 times the number of positive matches
	number_of_negatives = tf.math.minimum(tf.size(negative_matches_mask) - number_of_positives, 3 * number_of_positives)

	# Calculate the softmax cross entropy between confidence and ground truth data
	total_confidence_loss = tf.keras.losses.categorical_crossentropy(cats_one_hot, confidence)

	if CONFIG['use_w_det'] == True:
		sample_weights = tf.reduce_max(tf.math.multiply(cats_one_hot, CONFIG['object_weights']), axis = -1)
		total_confidence_loss = tf.math.multiply(sample_weights, total_confidence_loss)

	# Get the top k matches with largest confidence loss, being k the number of negative matches calculated previously, to use in the final confidence loss
	top_negative_values, top_negative_inds = tf.math.top_k(tf.boolean_mask(total_confidence_loss, negative_matches_mask), number_of_negatives)
	
	# Calculate the confidence loss associated with the negative matches
	negative_confidence_loss = tf.math.reduce_sum(top_negative_values) * tf.cast(tf.greater(number_of_negatives, 0), tf.float32)

	# Calculate the confidence loss associated with the positive matches
	positive_confidence_loss = tf.math.reduce_sum(tf.boolean_mask(total_confidence_loss, positive_matches_mask))

	# Cast the number of positive matches from int32 to float32 in order to calculate the loss
	number_of_positives = tf.cast(number_of_positives, tf.float32)
	number_of_negatives = tf.cast(number_of_negatives, tf.float32)

	# Calculate the final confidence loss
	confidence_loss = tf.math.divide_no_nan((positive_confidence_loss + negative_confidence_loss), number_of_positives)

	# Calculate the smooth L1 between localization and the difference calculated previously
	total_localization_loss = tf.keras.losses.huber(localization, gt_offsets)

	# Apply the positive matches mask to get just the loss concerning the positive matches 
	positive_localization_loss = tf.boolean_mask(total_localization_loss, tf.reshape(positive_matches_mask, [CONFIG['batch_size'], -1]))

	# Calculate the final localization loss through calculating the mean of all positive matches
	localization_loss = tf.cond(tf.equal(number_of_positives, 0.0), 
						lambda: 0.0,
						lambda: tf.math.reduce_mean(positive_localization_loss))

	# Compute the detection loss
	alpha = 1.0
	detection_loss = confidence_loss + alpha * localization_loss

	# Calculate an accuracy metric for detection task
	normalizer = tf.cast(tf.add(number_of_positives, number_of_negatives), tf.float32)
	inferred_class = tf.cast(tf.argmax(confidence, -1), tf.int32)
	positive_matches = tf.cast(tf.equal(tf.boolean_mask(inferred_class, positive_matches_mask), tf.boolean_mask(gt_cats, positive_matches_mask)), tf.float32)
	hard_matches = tf.equal(tf.boolean_mask(inferred_class, negative_matches_mask), tf.boolean_mask(gt_cats, negative_matches_mask))
	hard_matches = tf.cast(tf.gather(hard_matches, top_negative_inds), tf.float32)
	train_acc = tf.math.divide_no_nan((tf.reduce_sum(positive_matches) + tf.reduce_sum(hard_matches)), normalizer)
	return detection_loss, localization_loss, confidence_loss, train_acc

def segmentation_loss(seg_gt, seg_pred):
	""" Computes the segmentation loss.
	Args:
		seg_gt: A tensor of predicted data of shape [batch_size, fm_height, fm_width, num_classes].
		seg_pred: A tensor of shape [batch_size, img_height, img_width, 3].
	Returns:
		seg_loss: A float representing the segmentation_loss.
	"""
	# Calculate the unweighted softmax crossentropy
	seg_loss_local = tf.keras.losses.categorical_crossentropy(seg_gt, seg_pred, from_logits = True)

	if CONFIG['use_w_seg'] == True:	
		# Calculate the weights for de actual segmentation map
		sample_weights = tf.reduce_max(tf.math.multiply(seg_gt, CONFIG['class_weights']), axis = -1)
		
		# Compute the weighted segmentation loss
		seg_loss = tf.reduce_mean(tf.math.multiply(sample_weights, seg_loss_local))
	else:
		seg_loss = tf.reduce_mean(seg_loss_local)
	return seg_loss