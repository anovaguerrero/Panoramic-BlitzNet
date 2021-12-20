import os

# Class names
class_names = ['background', 'painting', 'bed', 'table', 'mirror', 'window', 'curtain', 'chair', 'light', 'sofa', 'door', 'cabinet', 'bedside', 'tv', 'shelf']

# Mean color to substract from the image before propagating thorugh the model
MEAN_COLOR = [123.151630838/255.0, 115.902882574/255.0, 103.062623801/255.0] # Format BGR

# PATHS TO THE PROJECT MAIN FOLDERS 
def check(dirname):
	"""Creates a directory in case it does not exist"""
	if not os.path.exists(dirname):
		os.makedirs(dirname)
	return dirname

# Path to the project directory
EVAL_DIR = os.path.dirname(os.path.realpath(__file__))

# Path to the folder where the model checkpoints are stored
CKPT_ROOT = check(os.path.join(EVAL_DIR, 'Checkpoints/')) 

# Path to the folder where the evaluation and test results are stored
RESULTS_ROOT = check(os.path.join(EVAL_DIR, 'Results/'))

# GENERAL VARIABLES FOR DATASET PROCESSING
# Width of the model input
img_width = 1024

# Height of the model input
img_height = 512

# Batch size for training and evaluation
batch_size = 1

# Flag to use validation set during training
use_val = False

# Percentage of training set to use as validation set
val_perc = 0.1

# Buffer size for dataset preprocessing
buffer_size = 566 # Must be equal or greater than the number of images in the training set

# Number of classes to detect and segment
num_classes = 15 # 14 classes plus the background

# VARIABLES FOR DATA AUGMENTATION
# Flags to apply a transformation or not
h_flip = True # Horizontal flip
h_rot = True # Horizontal rotation on the sphere
color_change = True # Color changes

# Variables to change how the data augmentation works
max_hue = 0.07 # In range [0, 0.5]
lower_sat = 0.5 # In range [0, upper_sat)
upper_sat = 1.5 # Greater than 'lower_sat'
max_brightness = 0.125 # Greater than zero
lower_contrast = 0.5 # In range [0, upper_sat)
upper_contrast = 1.5 # Greater than 'lower_sat'

# Odds of applying horizontal flip to the images
h_flip_th = 0.5 # In range [0, 1]

# Odds of applying horizontal rotation on the sphere to the images
h_rot_th = 0.9 # In range [0, 1]

# Odds of applying color changes to the images
color_th = 0.5 # In range [0, 1]

# VARIABLES FOR THE MODEL TRAINING, EVALUATION AND TEST
# Whether to save a different checkpoint each time the loss improves or overwrite the last saved checkpoint to save disk memory
overwrite_ckpt = True

# Initial learining rate during training
initial_lr = 0.0001

# Variances for cx, cy, w and h in order to calculate the detection loss
variances = [0.1, 0.1, 0.2, 0.2] 

# Minimum IoU between a proposal and a ground truth bbox to be accepted as positive match during training
iou_threshold = 0.5

# The lower feature map scale for computing the bounding boxes to generate
min_scale = 0.08 # In range [0, max_scale)

# The upper feature map scale for computing the bounding boxes to generate
max_scale = 0.95 # In range (min_scale, 1]

# Aspect ratios for anchor boxes
aspect_ratios = [1.0/3.0, 0.5, 1.0, 2.0, 3.0]

# Class weights for segmentation task
class_weights = [0.085, 12.336, 2.114, 3.160, 16.111, 2.768, 2.411, 5.870, 25.316, 3.201, 1.651, 3.396, 41.038, 25.277, 27.338]

# Flag to use weights in segmentation loss
use_w_seg = False

# Class weights for detection task
object_weights = [1.0, 0.6281, 1.4668, 0.6005, 2.1641, 0.9340, 0.7479, 0.5752, 1.0575, 0.9264, 0.5377, 0.8820, 1.4401, 1.7107, 6.4396]

# Flag to use weights in detection loss
use_w_det = False

# Confidence threshold to filter predictions before NMS in inference mode
confidence_th = 0.8

# Number of predictions to pass through NMS algorithm for each class
top_k_nms = 400

# Number of NMS outputs for each class
nms_max_output_size = 50

# IoU threshold used for filtering proposals during NMS
nms_th = 0.1

# IoU used to evaluate mAP metric
iou_test = 0.5

# DICTIONARY TO USE IN THE ENTIRE PROJECT
CONFIG = {
	'class_names': class_names,
	'MEAN_COLOR': MEAN_COLOR,
	'EVAL_DIR': EVAL_DIR,
	'CKPT_ROOT': CKPT_ROOT,
	'RESULTS_ROOT': RESULTS_ROOT,
	'img_width': img_width, 
	'img_height': img_height,
	'batch_size': batch_size,
	'use_val': use_val,
	'val_perc':val_perc,
	'buffer_size': buffer_size,
	'num_classes': num_classes,
	'h_flip': h_flip, 
	'h_rot': h_rot, 
	'color_change': color_change, 
	'max_hue': max_hue, 
	'lower_sat': lower_sat, 
	'upper_sat': upper_sat, 
	'max_brightness': max_brightness, 
	'lower_contrast': lower_contrast, 
	'upper_contrast': upper_contrast, 
	'h_flip_th': h_flip_th,
	'h_rot_th': h_rot_th,
	'color_th': color_th,
	'overwrite_ckpt': overwrite_ckpt,
	'initial_lr': initial_lr,
	'variances': variances, 
	'iou_threshold': iou_threshold,
	'min_scale': min_scale,
	'max_scale': max_scale,
	'aspect_ratios': aspect_ratios,
	'class_weights': class_weights,
	'use_w_seg': use_w_seg,
	'object_weights': object_weights,
	'use_w_det': use_w_det,
	'confidence_th': confidence_th,
	'top_k_nms': top_k_nms,
	'nms_max_output_size': nms_max_output_size,
	'nms_th': nms_th, 
	'iou_test': iou_test}