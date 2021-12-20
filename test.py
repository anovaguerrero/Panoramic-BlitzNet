"""Usage: test.py <experiment_name>

Options:
	-h --help  Show this screen.
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg') # Avoid RAM leaks during figures creation
import numpy as np
import os
import cv2
import time
import random
from PIL import Image
from config import CONFIG, check
from model_utils import NMS_fc
from pano_blitznet_model import PanoramicBlitzNet
from utils import resize_and_rescale
from docopt import docopt

def draw(img, dets, cats, scores, experiment_directory, name):
	"""Visualize objects detected by the network by putting bounding boxes"""
	img = tf.cast(img * 255.0, tf.uint8)
	nonzero_mask = cats > 0
	dets = tf.boolean_mask(dets, nonzero_mask)
	cats = tf.cast(tf.boolean_mask(cats, nonzero_mask), tf.uint8)
	scores = np.round(tf.boolean_mask(scores, nonzero_mask).numpy(), 3)

	# Convert predictions coordinates [cx, cy w, h] to [ymin, xmin, ymax, xmax].
	h, w = img.shape[1:3]
	ymin, xmin, ymax, xmax = dets[:, 0] * h, dets[:, 1] * w, dets[:, 2] * h, dets[:, 3] * w
	ymin = tf.expand_dims(ymin, axis = -1)
	xmin = tf.expand_dims(xmin, axis = -1)
	ymax = tf.expand_dims(ymax, axis = -1)
	xmax = tf.expand_dims(xmax, axis = -1)
	dets = tf.concat([ymin, xmin, ymax, xmax], axis = -1)

	plt.cla()
	plt.axis('off')
	plt.figure(figsize=(w / 10, h / 10))
	plt.imshow(img[0])
	for i in range(len(cats)):
		cat = cats[i]
		bbox = np.array(dets[i])
		color = (random.random(), random.random(), random.random())
		rect = plt.Rectangle((bbox[1], bbox[0]), (bbox[3] - bbox[1]), (bbox[2] - bbox[0]), fill = False, edgecolor = color, linewidth = 12.5)
		plt.gca().add_patch(rect)
		plt.gca().text(bbox[1], bbox[0], '{:s}'.format(CONFIG['class_names'][cat]), bbox = dict(facecolor = color, alpha = 0.5), fontsize = 96, color = 'white')

	plt.savefig(experiment_directory + '/%s_detection.jpg' % (name), bbox_inches = 'tight', dpi = 10)
	plt.clf()
	plt.close('all')

def draw_seg(img, logits, experiment_directory, name):
	"""Applies generated segmentation mask to an image"""
	palette = np.load('Extra/palette.npy').tolist()
	img = tf.cast(img*255.0, tf.uint8)
	img_size = (img.shape[1], img.shape[2])
	segmentation = tf.expand_dims(tf.math.argmax(logits, axis = -1), axis = -1)
	segmentation = tf.image.resize(segmentation, img_size, method = 'nearest')
	segmentation = tf.reshape(segmentation, img_size)	

	image = Image.fromarray(np.uint8(img[0].numpy())).convert('RGB')
	segmentation_draw = Image.fromarray(np.uint8(segmentation.numpy())).convert('P')
	segmentation_draw.putpalette(palette)
	segmentation_draw.save(experiment_directory + '/%s_segmentation.png' % (name), 'PNG')
	image.save(experiment_directory + '/%s.jpg' % (name), 'JPEG')

	seg = cv2.imread(experiment_directory + '/%s_segmentation.png' % (name))
	im = cv2.imread(experiment_directory + '/%s.jpg' % (name), 0)
	im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
	result = cv2.addWeighted(im, 0.5, seg, 0.5, 0)
	cv2.imwrite(experiment_directory + '/%s_segmentation.png' % (name), result)

def test(experiment_name):
	# Get the directory where the results will be stored	
	experiment_directory = check(os.path.join(CONFIG['RESULTS_ROOT'], '%s/' % experiment_name))

	# Generate the model
	images = tf.keras.Input(shape = (CONFIG['img_height'], CONFIG['img_width'], 3), batch_size = CONFIG['batch_size'], name = 'img_input')	
	blitznet = PanoramicBlitzNet(images)

	# Load net weights
	ckpt_dir = os.path.dirname(CONFIG['CKPT_ROOT'] + experiment_name + '/')
	ckpt_to_restore = tf.train.latest_checkpoint(ckpt_dir)
	print('[INFO]: Restoring %s' % ckpt_to_restore)
	blitznet.load_weights(ckpt_to_restore)
	print('[INFO]: Checkpoint restored')

	# Iterate over the test images
	images_path = os.path.join(CONFIG['EVAL_DIR'], 'dataset/test/rgb/')
	for filename in os.listdir(images_path):
		start_time = time.time()

		# Get the image and preprocess
		raw_img = tf.io.read_file(os.path.join(images_path, filename))
		image = tf.image.decode_png(raw_img, channels = 3)
		image = resize_and_rescale(image)
		image = tf.reshape(image, [1, CONFIG['img_height'], CONFIG['img_width'], 3])
		
		# Substract mean color before propagating the image through the model
		red, green, blue = tf.split(image, 3, axis = -1)
		input_image = 255.0 * (tf.concat([blue, green, red], axis = -1) - (CONFIG['MEAN_COLOR']))
		
		# Pass the image through the model and get the outputs
		predictions, logits = blitznet(input_image)
		nms_output = NMS_fc(predictions)

		# Save results
		draw_seg(image, logits, experiment_directory, os.path.splitext(filename)[0])
		draw(image, nms_output[0, :, -4:], nms_output[0, :, 0], nms_output[0, :, 1], experiment_directory, os.path.splitext(filename)[0])
		print("[INFO]: Time taken for image %s: %.2fs" % (filename, (time.time() - start_time)))

if __name__ == "__main__":
	args = docopt(__doc__)
	experiment_name = args['<experiment_name>']
	gpus = tf.config.experimental.list_physical_devices('GPU')
	for gpu in gpus:
		tf.config.experimental.set_memory_growth(gpu, True)   
	test(experiment_name)
