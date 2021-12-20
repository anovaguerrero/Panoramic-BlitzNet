"""sun360 dataset."""
import tensorflow_datasets as tfds
import tensorflow as tf
import glob
import json

# TODO: Make nicer this markdown description and add the correct citation to Zhang et al. (it will appear on the catalog page)
_DESCRIPTION = """
##EXTENDED SUN360
Dataset from PanoBliztNet paper that includes 666 indoor panoramas of bedrooms and living rooms
from the SUN360 dataset [Zhang et al.] extended with semantic segmentation labels.

It includes two splits for train and test, each of them containing:
    - RGB images
    - bounding boxes of all objects from 14 different classes
    - semantic segmentation masks

For more information visit the official webpage from paper 'What's in my Room? Object Recognition on Indoor Panoramic Images'
"""

# BibTeX citation
_CITATION = """
@INPROCEEDINGS{9197335,  
    author={J. {Guerrero-Viu} and C. {Fernandez-Labrador} and C. {Demonceaux} and J. J. {Guerrero}},  
    booktitle={2020 IEEE International Conference on Robotics and Automation (ICRA)},   
    title={What’s in my Room? Object Recognition on Indoor Panoramic Images},   
    year={2020},  
    volume={},  
    number={},  
    pages={567-573},  
    doi={10.1109/ICRA40945.2020.9197335}
}
"""

_HOMEPAGE = 'https://webdiis.unizar.es/~jguerrer/room_OR/'

_CLASS_NAMES = ['__background__', 'painting', 'bed', 'table', 'mirror', 'window', 'curtain', 'chair', 'light', 'sofa', 'door', 'cabinet', 'bedside', 'tv', 'shelf']

LABELS_ID = {'bed': "1", 'painting': "2", 'table': "3", 'mirror': "4", 'window': "5", 'curtain': "6", 'chair': "7", 'light': "8", 'sofa': "9", 'door': "10", 'cabinet': "11", 'bedside': "12", 
				'tv': "13" , 'shelf': "17"} # Labels ids used to create binary mask images. It is just for loading the images and store them into the record but they are not the final labels ids.

def build_bbox(x, y, w, h):
    image_height = 512.0
    image_width = 1024.0
    return tfds.features.BBox(
            ymin = y / image_height,
            xmin = x / image_width,
            ymax = (y + h) / image_height,
            xmax = (x + w) / image_width,
        )

class Sun360(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for sun360 dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # Images can have variable shape
            'image': tfds.features.Image(shape=(512, 1024, 3), encoding_format='png'),
            'image_filename': tfds.features.Text(),
            'image_id': tf.int64,
            'objects': tfds.features.Sequence({
              'id': tf.int64, # Unique id for each object within an image
              'area': tf.int64,
              'bboxes_list': tfds.features.Sequence(tfds.features.BBoxFeature()), # Each object can have two different bboxes whether it is cropped within the image limits or not
              'label': tfds.features.ClassLabel(names=_CLASS_NAMES), # 14 different classes + background
              'binary_mask': tfds.features.Image(shape=(512, 1024, 1), encoding_format='png'), 
            }),
            'segmentation_mask': tfds.features.Image(shape=(512, 1024, 1), encoding_format='png'),
        }),
        homepage=_HOMEPAGE,
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    path = '/home/alejandro/PanoBlitznet/dataset/' # CHANGE THIS PATH
    
    return {
        'train': self._generate_examples(path + 'train'),
        'test': self._generate_examples(path + 'test')
    }

  def _generate_examples(self, path):
    """Yields examples."""
    json_path = path + '/instances_SUN360_bboxes.json'
    rgb_folder = path + '/rgb/'
    binary_mask_folder = path + '/masks/'
    segmentation_folder = path + '/segmentation/'

    json_file = json.load(open(json_path,'r'))

    for i in json_file['images']:
        list_instances = [a for a in json_file['annotations'] if a['image_id'] == i['id']]

        # Get objects reversed list in order to achieve the second bbox of the objects that have it
        rev_list_instances = [list_instances[ins] for ins in reversed(range(len(list_instances)))] 
        instances_len = len(list_instances) # Number of objects of each image to use below

	# For figuring out if an object has two bboxes, the object area and the object category of sucesive items in the list are compared. 
    	# If both are equal, then the object has two different bboxes to be stored in 'bboxes_list'. If not, then the unique bbox is stored adding 
    	# a (0.0, 0.0, 0.0, 0.0) vector to define that the object does not have a second bbox since it is necessary to pass a float values vector to the 
    	# tfds.feature.BBoxFeature(). To do this checking with every object in every image, rev_list_instances array is used for avoiding an error when 
    	# comparing the last list_instances array value with the next one that does not exist. Using this reversed list to get the next list_instances value 
    	# but in rev_list_instances, when comparing the list_instances['last_item'] it will be compared with rev_list_instances[0] so the error does not appear.
		
        record = {
            'image': rgb_folder + i['file_name'] + '.png',
            'image_filename': i['file_name'],
            'image_id': i['id'], 
            'objects': [{
              'id': list_instances[ins]['id'],
              'area': list_instances[ins]['area'],
              'bboxes_list': [build_bbox(*list_instances[ins]['bbox']), build_bbox(*rev_list_instances[instances_len-(ins+2)]['bbox']) if \
                              (list_instances[ins]['category_id'] == rev_list_instances[instances_len-(ins+2)]['category_id'] and \
                              list_instances[ins]['area'] == rev_list_instances[instances_len-(ins+2)]['area']) else build_bbox(0.0, 0.0, 0.0, 0.0)],
              'label': list_instances[ins]['category_id'],
              'binary_mask': binary_mask_folder + i['file_name'] +'_'+ list_instances[ins]['id'] +'_'+ LABELS_ID[list_instances[ins]['category_id']] +'_'+ \
                              list_instances[ins]['category_id'] + '.png',
            } for ins in range(instances_len) if (list_instances[ins]['category_id'] != list_instances[ins-1]['category_id'] or \
              list_instances[ins]['area'] != list_instances[ins-1]['area'])],
            'segmentation_mask': segmentation_folder + i['file_name'] + '_seg.png',
        }

        yield i['id'], record
