"""Usage: evaluation.py <experiment_name>

Options:
	-h --help  Show this screen.
"""
import tensorflow as tf
import os
import time
from config import CONFIG, check
from model_utils import NMS_fc
from preprocess_sun360 import prepare_ds
from pano_blitznet_model import PanoramicBlitzNet
from utils import *
from docopt import docopt

def evaluation(experiment_name):
    """Calculates and store evaluation metrics for a given trained model.
    Args:
        experiment_name: The name of the experiment whose weights will be loaded to evaluate.
    """
    def feed_forward_eval(eval_dataset):
        """Pass a given dataset through the model.
        Args:
            eval_dataset: The dataset to be evaluated. It contains a tuple of images, ground truth bounding boxes and categories and 
            ground truth segmentation masks.
        """
        detections = []
        gt_labels = []
        segmentation_maps = []
        gt_segmentation = []
        n_imgs = 0.0 # For matching predictions to their respective image

        # Iterate over the batches of the dataset.
        for step, (x_batch, y_batch) in enumerate(eval_dataset):
            start_time = time.time()
            # Get the image and the ground truth for the mini batch
            x_image = x_batch
            y_label, y_seg_mask = y_batch

            # Substract mean color before propagating the image through the CNN
            red, green, blue = tf.split(x_image, 3, axis = -1)
            image = 255.0 * (tf.concat([blue, green, red], axis = -1) - CONFIG['MEAN_COLOR'])    

            # Pass the image through the model
            predictions, logits = blitznet(image)

            # Get the final predictions for detection task
            nms_output = NMS_fc(predictions)
            
            # For detection proposals remove zero-value detections and store each prediction with its respective image. Same for ground truth
            for i in range(len(nms_output)):
                # Get ground truth and predictions data
                img_gt_labels = y_label[i]
                img_detections = nms_output[i]

                # Remove zero-values from predicted bounding boxes (during NMS these are created in order to concatenate predictions for every class)
                nonzero_mask = img_detections[:, 0] > 0
                img_detections = tf.boolean_mask(img_detections, nonzero_mask)

                # Match each prediction with its respective image to evaluate AP later
                img_ind = tf.constant(n_imgs, shape = [tf.shape(img_detections)[0], 1], dtype = tf.float32)
                processed_detections = tf.concat([img_ind, img_detections], axis = -1)

                # Match each ground truth bounding box with its respective image to evaluate AP later
                img_ind = tf.constant(n_imgs, shape = [tf.shape(img_gt_labels)[0], 1], dtype = tf.float32)
                processed_labels = tf.concat([img_ind, img_gt_labels], axis = -1)
                detections.append(processed_detections)   
                gt_labels.append(processed_labels)
                n_imgs += 1.0
            gt_segmentation.append(y_seg_mask)    
            segmentation_maps.append(logits)
            print("[INFO]: Time taken for image %s: %.2fs" % (step, (time.time() - start_time)))

        # Get concatenated ground truth and predicted data for the entire dataset   
        detections = tf.concat(detections, axis = 0)
        gt_labels = tf.concat(gt_labels, axis = 0)
        segmentation_maps = tf.concat(segmentation_maps, axis = 0)
        gt_segmentation = tf.concat(gt_segmentation, axis = 0)        
        return detections, gt_labels, segmentation_maps, gt_segmentation 
 
    def eval_metrics(cid):
        """Calculates precision and recall metrics for a given class from the results obtained in 'feed_forward_eval' function.
        Args:
            cid: Category ID to evaluate.
        Returns:
            precision: Array of precision values for the given category ID.
            recall: Array of recall values for the given category ID.
            tp_value: Number of true positives for the given category ID.
            fp_value: Number of false positives for the given category ID.
            num_gt_class: Number of false positives plus false negatives for the given category ID.
        """
        # Initialize some variables
        tp = []
        fp = []
        num_gt_class = 0
        
        # Get ground truth data for the given category ID
        gt_class_mask = gt_labels[:, -1] == cid
        gt_class_labels = tf.boolean_mask(gt_labels, gt_class_mask)
        gt_class_bboxes = gt_class_labels[:, -5:-1]

        # Get the number of ground truth bboxes for the given category ID
        num_gt_class = tf.shape(gt_class_labels)[0].numpy()

        # Create an array to check when a ground truth has been matched to a prediction
        gt_matched = tf.constant(False, tf.bool, num_gt_class)

        # Get the predictions for the given category ID
        class_detections = detections[detections[:, -6] == cid]
        num_det = len(class_detections)

        # Sort the predictions for the given category ID in descending format to start the evaluation
        class_confidences = class_detections[:, -5]
        sorted_inds = tf.argsort(class_confidences, direction = 'DESCENDING')
        sorted_detections = tf.gather(class_detections, sorted_inds)

        # Iterate over the number of detections to match them with ground truth and calculate metrics
        for detection in range(num_det):
            if num_gt_class > 0:
                # Get the bounding boxes coordinates for every ground truth and actual detection
                det_bbox = sorted_detections[detection, -4:]
                det_img = sorted_detections[detection, 0]
                max_iou = 0.0

                # Calculate IoU between actual detection and every ground truth bounding box
                # Compute the intersection corners
                ymin = tf.math.maximum(gt_class_bboxes[:, 0], det_bbox[0])
                xmin = tf.math.maximum(gt_class_bboxes[:, 1], det_bbox[1])
                ymax = tf.math.minimum(gt_class_bboxes[:, 2], det_bbox[2])
                xmax = tf.math.minimum(gt_class_bboxes[:, 3], det_bbox[3])
                        
                # Calculate the intersection area 
                intersections_width = tf.math.maximum(0.0, (xmax - xmin))
                intersections_height = tf.math.maximum(0.0, (ymax - ymin))
                intersections = intersections_width * intersections_height

                # Calculate IoU and get the maximum IoU and the index of the matched ground truth
                unions = (gt_class_bboxes[:, 2] - gt_class_bboxes[:, 0]) * (gt_class_bboxes[:, 3] - gt_class_bboxes[:, 1]) + (det_bbox[2] - det_bbox[0]) * (det_bbox[3] - det_bbox[1]) - intersections
                ious = intersections / unions
                img_mask = gt_class_labels[:, 0] == det_img 
                ious = ious * tf.cast(img_mask, tf.float32)
                max_iou = tf.math.reduce_max(ious)
                ind_max_iou = tf.math.argmax(ious)

                # Check if matched prediction is true positive or false positive
                if max_iou > CONFIG['iou_test']:
                    # If matched ground truth has not been detected yet then remove from the list of matched ground truth
                    if gt_matched[ind_max_iou] == False:
                        tp.append(1.0)
                        fp.append(0.0)
                        gt_matched = tf.Variable(gt_matched)
                        gt_matched[ind_max_iou].assign(True)
                        gt_matched = tf.convert_to_tensor(gt_matched)

                    # Else actual detection is matching an already detected ground truth so is a false positive
                    else:
                        tp.append(0.0)
                        fp.append(1.0)
                else:
                    tp.append(0.0)
                    fp.append(1.0)
            else:
                tp.append(0.0)
                fp.append(1.0)
                
        # Get the cumulative TP and FP to calculate precision and recall points
        tp = tf.math.cumsum(tp)       
        fp = tf.math.cumsum(fp)
        if num_det > 0:
            # Get the total number of TP and FP and precision and recall points
            tp_value = tp[-1]
            fp_value = fp[-1]
            precision = tf.math.divide_no_nan(tp, (tp + fp))
            recall = tf.math.divide_no_nan(tp, num_gt_class)

        # If no detections then put every variable to zero
        else:  
            tp_value = 0.0
            fp_value = 0.0
            precision = 0.0
            recall = 0.0
        return precision, recall, tp_value, fp_value, num_gt_class

    def compute_ap(precision, recall):
        """Calculates average precision and weighted average precision metric for given precision and recall points.
        Args:
            precision: Array of precision points.
            recall: Array of recall points.
        Returns:
            ap: Average precision value.
            wap: Weighted average precision value.
        """
        precision = tf.Variable(precision)
        for i in range((tf.size(precision) - 1), 0, -1):
            precision[i - 1].assign(tf.math.maximum(precision[i], precision[i - 1]))
        precision = tf.convert_to_tensor(precision)
        w_ap = []
        for i in range(tf.size(recall) -1):
            w_ap.append((recall[i + 1] - recall[i]) * precision[i + 1])
        w_ap = tf.reduce_sum(w_ap)    
        ap = tf.math.reduce_mean(precision) 
        return ap, w_ap

    def make_iou_table(ious):
        """ Prints and saves a table of IoU values for each class in segmentation task.
        Args:
            ious: Array of IoU values for every class from segmentation.
        """
        with open(experiment_directory + 'conf-%.2f-iou-%.2f.txt' % (CONFIG['confidence_th'], CONFIG['iou_test']), 'a') as f:
            tf.print('\n\nTOTAL MeanIoU\n-----------------------')
            f.write('\n\nTOTAL MeanIoU\n-----------------------\n')
            for cid in range(len(ious)):
                cat_name = CONFIG['class_names'][cid]
                iou = ious[cid].numpy()
                if len(cat_name) >= 7:
                    tf.print(cat_name, '\t|', np.round(iou, 3))
                    f.write('%s \t| %s\n' % (cat_name, repr(np.round(iou, 3))))
                else:
                    tf.print(cat_name, '\t\t|', np.round(iou, 3))
                    f.write('%s \t\t| %s\n' % (cat_name, repr(np.round(iou, 3))))     
            tf.print('AVERAGE\t\t|', np.round(tf.math.reduce_mean(ious), 3))
            f.write('AVERAGE\t\t| %s' % (repr(np.round(tf.math.reduce_mean(ious), 3))))
            f.close()

    def make_ap_table(aps):
        """ Prints and saves a table of AP values for each class in detection task.
        Args:
            aps: Array of AP values for every class from detection.
        """
        with open(experiment_directory + 'conf-%.2f-iou-%.2f.txt' % (CONFIG['confidence_th'], CONFIG['iou_test']), 'a') as f:
            tf.print('\n\nTOTAL MeanAP\n-----------------------')
            f.write('\n\nTOTAL MeanAP\n-----------------------\n')
            for cid in range(len(aps)):
                cat_name = CONFIG['class_names'][cid + 1]
                ap = aps[cid].numpy()
                if len(cat_name) >= 7:
                    tf.print(cat_name, '\t|', np.round(ap, 3))
                    f.write('%s \t| %s\n' % (cat_name, repr(np.round(ap, 3))))
                else:
                    tf.print(cat_name, '\t\t|', np.round(ap, 3))
                    f.write('%s \t\t| %s\n' % (cat_name, repr(np.round(ap, 3))))
            tf.print('AVERAGE\t\t|', np.round(tf.math.reduce_mean(aps), 3))
            f.write('AVERAGE\t\t| %s' % (repr(np.round(tf.math.reduce_mean(aps), 3))))
            f.close()            

    def make_wap_table(w_aps):
        """ Prints and saves a table of Weighted AP values for each class in detection task.
        Args:
            w_aps: Array of Weighted AP values for every class from detection.
        """        
        with open(experiment_directory + 'conf-%.2f-iou-%.2f.txt' % (CONFIG['confidence_th'], CONFIG['iou_test']), 'a') as f:
            tf.print('\n\nTOTAL WEIGHTED MeanAP\n-----------------------')
            f.write('\n\nTOTAL WEIGHTED MeanAP\n-----------------------\n')
            for cid in range(len(w_aps)):
                cat_name = CONFIG['class_names'][cid + 1]
                ap = w_aps[cid].numpy()
                if len(cat_name) >= 7:
                    tf.print(cat_name, '\t|', np.round(ap, 3))
                    f.write('%s \t| %s\n' % (cat_name, repr(np.round(ap, 3))))
                else:
                    tf.print(cat_name, '\t\t|', np.round(ap, 3))    
                    f.write('%s \t\t| %s\n' % (cat_name, repr(np.round(ap, 3))))
            tf.print('AVERAGE\t\t|', np.round(tf.math.reduce_mean(w_aps), 3))  
            f.write('AVERAGE\t\t| %s' % (repr(np.round(tf.math.reduce_mean(w_aps), 3))))
            f.close() 

    def make_precision_table(tps, fps):
        """ Prints and saves a table of Precision values for each class in detection task.
        Args:
            tps: Array of TP values for every class from detection.
            fps: Array of FP values for every class from detection.
        """        
        with open(experiment_directory + 'conf-%.2f-iou-%.2f.txt' % (CONFIG['confidence_th'], CONFIG['iou_test']), 'a') as f:
            tf.print('\n\nTOTAL Precision\n-----------------------')
            f.write('\n\nTOTAL Precision\n-----------------------\n')
            for cid in range(len(tps)):
                cat_name = CONFIG['class_names'][cid + 1]
                tp = tps[cid].numpy()
                fp = fps[cid].numpy()
                precision = tf.math.divide_no_nan(tp, (tp + fp))
                if len(cat_name) >= 7:
                    tf.print(cat_name, '\t|', np.round(precision, 3))
                    f.write('%s \t| %s\n' % (cat_name, repr(np.round(precision, 3))))
                else:
                    tf.print(cat_name, '\t\t|', np.round(precision, 3))
                    f.write('%s \t\t| %s\n' % (cat_name, repr(np.round(precision, 3))))
            total_precision = tf.reduce_sum(tps) / (tf.reduce_sum(tps) + tf.reduce_sum(fps))
            tf.print('TOTAL\t\t|', np.round(total_precision, 3))
            f.write('TOTAL\t\t| %s' % (repr(np.round(total_precision.numpy(), 3 ))))
            f.close()   

    def make_recall_table(tps, num_gts):
        """ Prints and saves a table of Recall values for each class in detection task.
        Args:
            tps: Array of TP values for every class from detection.
            num_gts: Array of FP + FN values for every class from detection.
        """              
        with open(experiment_directory + 'conf-%.2f-iou-%.2f.txt' % (CONFIG['confidence_th'], CONFIG['iou_test']), 'a') as f:
            tf.print('\n\nTOTAL Recall\n-----------------------')
            f.write('\n\nTOTAL Recall\n-----------------------\n')
            for cid in range(len(tps)):
                cat_name = CONFIG['class_names'][cid + 1]
                tp = tps[cid].numpy()
                num_gt = tf.cast(num_gts[cid], tf.float32)
                recall = tp / num_gt
                if len(cat_name) >= 7:
                    tf.print(cat_name, '\t|', np.round(recall, 3))
                    f.write('%s \t| %s\n' % (cat_name, repr(np.round(recall, 3))))
                else:
                    tf.print(cat_name, '\t\t|', np.round(recall, 3))
                    f.write('%s \t\t| %s\n' % (cat_name, repr(np.round(recall, 3))))
            total_recall = tf.reduce_sum(tps) / tf.reduce_sum(tf.cast(num_gts, tf.float32))
            tf.print('TOTAL\t\t|', np.round(total_recall, 3))   
            f.write('TOTAL\t\t| %s' % (repr(np.round(total_recall.numpy(), 3))))
            f.close()           

    # Get the directory where the results will be stored
    experiment_directory = check(os.path.join(CONFIG['RESULTS_ROOT'], '%s/' % experiment_name))	

    # Generate the model
    images = tf.keras.Input(shape = (CONFIG['img_height'], CONFIG['img_width'], 3), batch_size = CONFIG['batch_size'], name = 'img_input')	
    blitznet = PanoramicBlitzNet(images)

    # Load model weights
    ckpt_dir = os.path.dirname(CONFIG['CKPT_ROOT'] + experiment_name + '/')
    ckpt_to_restore = tf.train.latest_checkpoint(ckpt_dir)
    print('[INFO]: Restoring %s' % ckpt_to_restore)
    blitznet.load_weights(ckpt_to_restore)
    print('[INFO]: Checkpoint restored')

    # Get the test set and pass it through the model
    eval_dataset = prepare_ds(training = False)
    detections, gt_labels, segmentation_maps, gt_segmentation = feed_forward_eval(eval_dataset)

    aps, w_aps = [], []
    tps, fps, num_gts = [], [], []

    # Get the evaluation metrics for each category ID
    for cid in range(1, CONFIG['num_classes']):
        precision, recall, tp, fp, num_gt = eval_metrics(cid)
        ap, w_ap = compute_ap(precision, recall)
        aps.append(ap)
        w_aps.append(w_ap)
        tps.append(tp)
        fps.append(fp)
        num_gts.append(num_gt)
    aps = tf.concat(aps, axis = 0)
    w_aps = tf.concat(w_aps, axis = 0)  
    tps = tf.concat(tps, axis = 0)  
    fps = tf.concat(fps, axis = 0)

    # Save into a txt file each table
    with open(experiment_directory + 'conf-%.2f-iou-%.2f.txt' % (CONFIG['confidence_th'], CONFIG['iou_test']), 'w') as f:
        f.write('%s Results, Confidence = %s, IoU = %s\n' % (experiment_name, repr(CONFIG['confidence_th']), repr(CONFIG['iou_test'])))
        f.close()
    
    make_ap_table(aps)
    make_wap_table(w_aps)    
    make_precision_table(tps, fps)
    make_recall_table(tps, num_gts)

    confusion_matrix = calculate_confusion_matrix(segmentation_maps, gt_segmentation)
    iou = calculate_IoU(confusion_matrix)
    make_iou_table(iou)

if __name__ == "__main__":
    args = docopt(__doc__)
    experiment_name = args['<experiment_name>']
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)    
    evaluation(experiment_name)