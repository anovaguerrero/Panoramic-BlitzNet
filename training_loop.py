"""Usage: training_loop.py <experiment_name> <epochs> [--restore_ckpt] 

Options:
	-h --help  Show this screen.
    --restore_ckpt  Restore weights from previous training to continue with the training.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # Avoid information logs during execution
import tensorflow as tf
import numpy as np
import time
import datetime
from preprocess_sun360 import prepare_ds
from pano_blitznet_model import PanoramicBlitzNet
from losses_metrics import detection_objective, segmentation_loss
from config import CONFIG
from utils import *
from model_utils import NMS_fc, matching_fc
from docopt import docopt

def training_loop(experiment_name, epochs, restore_ckpt = False):
    """ Trains the model for a number of epochs.
    Args:
        experiment_name: Name for the actual training experiment.
        epochs: Number of epochs to train the model.
        restore_ckpt: Whether to restore weights to continue a training.
    Returns: 
        Trained model with saved final weights.
    """    
    def update_lr(epoch):
        """ Returns a custom learning rate that decreases as epochs progress.
        Args:
            epoch: The current epoch along the training process.
        Returns: 
            learning_rate: The float value corresponding to the learning rate to use in the current epoch of the training process.
        """
        initial_lr = CONFIG['initial_lr']
        if epoch < 90:
            optimizer.lr = initial_lr
        elif epoch < 130:
            optimizer.lr = initial_lr*0.1          
        else:
            optimizer.lr = initial_lr*0.01

    @tf.function(input_signature=[tf.TensorSpec(shape=[CONFIG['batch_size'], CONFIG['img_height'], CONFIG['img_width'], 3]), tf.TensorSpec(shape=[CONFIG['batch_size'], None, 5]), tf.TensorSpec(shape=[CONFIG['batch_size'], CONFIG['img_height']//4, CONFIG['img_width']//4, CONFIG['num_classes']])])
    def train_step(x_image, y_label, y_seg_mask):
        with tf.GradientTape() as tape:
            # Run the forward pass of the layer.
            # The operations that the layer applies to its inputs are going to be recorded on the GradientTape.
            predictions, logits = blitznet(x_image)  # Predictions and logits for this minibatch

            # Match anchor boxes to ground truth for training
            matched_predictions = matching_fc(y_label, predictions[:, :, -4:])

            # Compute the loss value for this minibatch.
            det_loss, loc_loss, conf_loss, acc = detection_objective(matched_predictions, predictions)
            seg_loss = segmentation_loss(y_seg_mask, logits)
            loss_value = det_loss + seg_loss

        # Use the gradient tape to automatically retrieve the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss_value, blitznet.trainable_weights)

        # Run one step of gradient descent by updating the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(grads, blitznet.trainable_weights))

        # Update training metric.
        segmentation_metric.update_state(tf.argmax(y_seg_mask, axis = -1), tf.argmax(logits, axis = -1))
        return loss_value, det_loss, loc_loss, conf_loss, seg_loss, acc, logits, predictions

    # Generate the net model
    images = tf.keras.Input(shape = (CONFIG['img_height'], CONFIG['img_width'], 3), batch_size = CONFIG['batch_size'], name = 'img_input')	
    blitznet = PanoramicBlitzNet(images)

    # Restore a model checkpoint to continue a training if it proceeds
    if restore_ckpt == True:
        ckpt_dir = os.path.dirname(CONFIG['CKPT_ROOT'] + experiment_name + '/')
        ckpt_to_restore = tf.train.latest_checkpoint(ckpt_dir)
        print('[INFO]: Restoring %s' % ckpt_to_restore)
        blitznet.load_weights(ckpt_to_restore)
        print('[INFO]: Checkpoint restored')

    # Define the optimizer and training metrics
    optimizer = tf.optimizers.Adam()
    segmentation_metric = tf.keras.metrics.MeanIoU(num_classes = CONFIG['num_classes'])
    best_loss = float('inf')

    # Create Tensorboard Logs
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'Logs/' + experiment_name + '/' + current_time + '/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    if CONFIG['use_val']:
        val_log_dir = 'Logs/' + experiment_name + '/' + current_time + '/val'
        val_summary_writer = tf.summary.create_file_writer(val_log_dir)
        
        # Get the training and validation datasets
        train_dataset, val_dataset = prepare_ds(training = True)
    else:
        train_dataset = prepare_ds(training = True)     

    # Start the training loop
    for epoch in range(epochs):
        print("\n[INFO]: Start of epoch %d" % (epoch))
        update_lr(epoch)
        print("[INFO]: Setting learning rate in %f" % (optimizer.lr))
        print("\n[INFO]: Computing training set")
        
        # Initialize some variables for training set
        start_time = time.time()
        acc_loss = 0.0
        acc_seg_loss = 0.0
        acc_det_loss = 0.0
        acc_loc_loss = 0.0
        acc_conf_loss = 0.0
        acc_accuracy = 0.0
        acc_cm = np.zeros(shape = (CONFIG['num_classes'], CONFIG['num_classes']))

        # Iterate over the batches of the dataset.
        for step, (x_batch, y_batch) in enumerate(train_dataset):
            # Get the image and the ground truth for the mini batch
            x_image = x_batch
            y_label, y_seg_mask = y_batch
            # x_gt_bbox = y_label[..., :-1] # Used just for Tensorboard
            
            # Substract mean color before propagating the image through the CNN
            red, green, blue = tf.split(x_image, 3, axis = -1)
            image = 255.0 * (tf.concat([blue, green, red], axis = -1) - CONFIG['MEAN_COLOR'])

            # Calculate losses and metrics for the mini batch
            loss_value, det_loss, loc_loss, conf_loss, seg_loss, acc, logits, predictions = train_step(image, y_label, y_seg_mask)
            cm = calculate_confusion_matrix(logits, y_seg_mask)
            acc_cm += cm
            acc_loss += loss_value
            acc_det_loss += det_loss
            acc_loc_loss += loc_loss
            acc_conf_loss += conf_loss
            acc_seg_loss += seg_loss
            acc_accuracy += acc

        # Calculate accumulate losses and metrics for the epoch
        mean_loss = acc_loss / (step + 1)
        mean_det_loss = acc_det_loss / (step + 1)
        mean_loc_loss = acc_loc_loss / (step + 1)
        mean_conf_loss = acc_conf_loss / (step + 1)
        mean_seg_loss = acc_seg_loss / (step + 1)
        mean_accuracy = acc_accuracy / (step + 1)

        # Print the results after the epoch
        mIoU = segmentation_metric.result()
        template = '[INFO]: Epoch {}, Loss: {}, Det Loss: {}, Loc Loss: {}, Conf Loss: {}, Accuracy: {}, Seg Loss: {}, mIoU: {}'
        print (template.format(epoch,
                                mean_loss, 
                                mean_det_loss,
                                mean_loc_loss, 
                                mean_conf_loss,
                                mean_accuracy,
                                mean_seg_loss,
                                mIoU))    
        segmentation_metric.reset_states()

        # Get some stuff to represent in Tensorboard. TODO Uncomment the data you want to represent

        # figure = plot_confusion_matrix(acc_cm, class_names = CONFIG['class_names'])
        # cm_image = plot_to_image(figure)
        # iou = calculate_IoU(acc_cm)
        # figure = plot_iou(iou, class_names = CONFIG['class_names'])
        # iou_image = plot_to_image(figure)            
        seg_map = tf.cast(tf.reshape(tf.math.argmax(logits, axis = -1), [CONFIG['batch_size'], CONFIG['img_height']//4, CONFIG['img_width']//4, 1]), tf.uint8)
        # y_seg_map = tf.cast(tf.reshape(tf.math.argmax(y_seg_mask, axis = -1), [CONFIG['batch_size'], CONFIG['img_height']//4, CONFIG['img_width']//4, 1]), tf.uint8)

        # nms = NMS_fc(predictions)
        # image = tf.cast(x_image*255.0, tf.uint8)
        # det_img = draw_boxes_batch(image.numpy(), nms)
        # colors = np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]) # 2 colors needed for this function to represent the bboxes
        # img = tf.image.draw_bounding_boxes(x_image, tf.reshape(x_gt_bbox, [CONFIG['batch_size'], -1, 4]), colors)

        # Save data to Tensorboard       
        with train_summary_writer.as_default():
            # tf.summary.image('detection', tf.reshape(det_img, [CONFIG['batch_size'], CONFIG['img_height'], CONFIG['img_width'], 3]), max_outputs=4, step=epoch)
            # tf.summary.image('detection_gt', img, max_outputs=4, step=epoch)
            tf.summary.scalar('learning rate', data = optimizer.lr, step = epoch)
            tf.summary.scalar('mIoU', mIoU, step=epoch)
            tf.summary.scalar('accuracy', mean_accuracy, step=epoch)
            tf.summary.scalar('total_loss', mean_loss, step=epoch)
            tf.summary.scalar('det_loss', mean_det_loss, step=epoch)
            tf.summary.scalar('loc_loss', mean_loc_loss, step=epoch)
            tf.summary.scalar('conf_loss', mean_conf_loss, step=epoch)
            tf.summary.scalar('seg_loss', mean_seg_loss, step=epoch)              
            tf.summary.image('prediction', 18*seg_map, max_outputs=4, step=epoch)
            # tf.summary.image('ground_truth', 18*y_seg_map, max_outputs=4, step=epoch)
            # tf.summary.image('confusion_matrix', cm_image, step=epoch)
            # tf.summary.image('iou', iou_image, step=epoch)

        print("[INFO]: Time taken for training set: %.2fs" % (time.time() - start_time))

        if CONFIG['use_val']:
            # Initialize some variables for validation set
            print("\n[INFO]: Computing validation set")
            start_time = time.time()
            acc_loss = 0.0
            acc_seg_loss = 0.0
            acc_det_loss = 0.0
            acc_loc_loss = 0.0
            acc_conf_loss = 0.0
            acc_accuracy = 0.0
            acc_cm = np.zeros(shape = (CONFIG['num_classes'], CONFIG['num_classes']))

            for step, (x_batch, y_batch) in enumerate(val_dataset):
                # Get the image and the ground truth for the mini batch    
                x_image = x_batch
                y_label, y_seg_mask = y_batch
                # x_gt_bbox = y_label[..., :-1] # Used just for Tensorboard
                
                # Substract mean color before propagating the image through the CNN
                red, green, blue = tf.split(x_image, 3, axis = -1)
                image = 255.0 * (tf.concat([blue, green, red], axis = -1) - CONFIG['MEAN_COLOR'])

                # Calculate losses and metrics for the mini batch
                loss_value, det_loss, loc_loss, conf_loss, seg_loss, acc, logits, predictions = train_step(x_image, y_label, y_seg_mask)
                cm = calculate_confusion_matrix(logits, y_seg_mask)
                acc_cm += cm
                acc_loss += loss_value
                acc_det_loss += det_loss
                acc_loc_loss += loc_loss
                acc_conf_loss += conf_loss
                acc_seg_loss += seg_loss
                acc_accuracy += acc

            # Calculate accumulate losses and metrics for the epoch
            mean_loss = acc_loss / (step + 1)
            mean_det_loss = acc_det_loss / (step + 1)
            mean_loc_loss = acc_loc_loss / (step + 1)
            mean_conf_loss = acc_conf_loss / (step + 1)
            mean_seg_loss = acc_seg_loss / (step + 1)
            mean_accuracy = acc_accuracy / (step + 1)
    
            # Print the results after the epoch
            mIoU = segmentation_metric.result()
            template = '[INFO]: Epoch {}, Loss: {}, Det Loss: {}, Loc Loss: {}, Conf Loss: {}, Accuracy: {}, Seg Loss: {}, mIoU: {}'
            print (template.format(epoch,
                                    mean_loss, 
                                    mean_det_loss,
                                    mean_loc_loss, 
                                    mean_conf_loss,
                                    mean_accuracy,
                                    mean_seg_loss,
                                    mIoU))    
            segmentation_metric.reset_states()

            # Get some stuff to represent in Tensorboard. TODO Uncomment the data you want to represent

            # figure = plot_confusion_matrix(acc_cm, class_names = CONFIG['class_names'])
            # cm_image = plot_to_image(figure)
            # iou = calculate_IoU(acc_cm)
            # figure = plot_iou(iou, class_names = CONFIG['class_names'])
            # iou_image = plot_to_image(figure)            
            seg_map = tf.cast(tf.reshape(tf.math.argmax(logits, axis = -1), [CONFIG['batch_size'], CONFIG['img_height']//4, CONFIG['img_width']//4, 1]), tf.uint8)
            # y_seg_map = tf.cast(tf.reshape(tf.math.argmax(y_seg_mask, axis = -1), [CONFIG['batch_size'], CONFIG['img_height']//4, CONFIG['img_width']//4, 1]), tf.uint8)

            # nms = NMS_fc(predictions)
            # image = tf.cast(x_image*255.0, tf.uint8)
            # det_img = draw_boxes_batch(image.numpy(), nms)
            # colors = np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]) # 2 colors needed for this function to represent the bboxes
            # img = tf.image.draw_bounding_boxes(x_image, tf.reshape(x_gt_bbox, [CONFIG['batch_size'], -1, 4]), colors)              
            with val_summary_writer.as_default():
                # tf.summary.image('detection', tf.reshape(det_img, [CONFIG['batch_size'], CONFIG['img_height'], CONFIG['img_width'], 3]), max_outputs=4, step=epoch)
                # tf.summary.image('detection_gt', img, max_outputs=4, step=epoch)
                tf.summary.scalar('learning rate', data = optimizer.lr, step = epoch)
                tf.summary.scalar('mIoU', mIoU, step=epoch)
                tf.summary.scalar('accuracy', mean_accuracy, step=epoch)
                tf.summary.scalar('total_loss', mean_loss, step=epoch)
                tf.summary.scalar('det_loss', mean_det_loss, step=epoch)
                tf.summary.scalar('loc_loss', mean_loc_loss, step=epoch)
                tf.summary.scalar('conf_loss', mean_conf_loss, step=epoch)
                tf.summary.scalar('seg_loss', mean_seg_loss, step=epoch)              
                tf.summary.image('prediction', 18*seg_map, max_outputs=4, step=epoch)
                # tf.summary.image('ground_truth', 18*y_seg_map, max_outputs=4, step=epoch)
                # tf.summary.image('confusion_matrix', cm_image, step=epoch)
                # tf.summary.image('iou', iou_image, step=epoch)

            print("[INFO]: Time taken for validation set: %.2fs" % (time.time() - start_time))

        # Save the model weights
        final_loss = mean_loss
        if final_loss < best_loss:
            best_loss = final_loss
            if CONFIG['overwrite_ckpt'] == True:
                ckpt_dir = CONFIG['CKPT_ROOT'] + experiment_name + '/cp_model.ckpt'
            else:
                ckpt_dir = CONFIG['CKPT_ROOT'] + experiment_name + '/cp_model_%s.ckpt' % epoch
            blitznet.save_weights(ckpt_dir)
            print("\n[INFO]: Saving weights to: %s" % (ckpt_dir))

if __name__ == "__main__":
	args = docopt(__doc__)
	experiment_name = args['<experiment_name>']
	epochs = int(args['<epochs>'])
	restore_ckpt = args['--restore_ckpt']

	gpus = tf.config.experimental.list_physical_devices('GPU')
	for gpu in gpus:
		tf.config.experimental.set_memory_growth(gpu, True)
	
	training_loop(experiment_name, epochs, restore_ckpt)
	print('[INFO]: Training finished')