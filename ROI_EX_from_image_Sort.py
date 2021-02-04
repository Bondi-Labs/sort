"""ROI extraction from an the current frame

---input = RGB image (frame)
---output = detections


Params:
  detections - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
Returns the a similar array, where the last column is the object ID.

NOTE: The number of objects returned may differ from the number of detections provided.



"""

import numpy as np
import tensorflow as tf
import matplotlib

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util

# Set a threshold for picking up detected objects
Score_Th = 0.5

PATH_TO_FROZEN_GRAPH = '/media/pk/Data/Project/Project/Venice/Maet_stage1/efficientdet_d1_coco17_tpu-32/6class/' \
                       'TRT_model/With_calibration_input'
PATH_TO_LABELS = '/media/pk/Data/Project/dataset/Meat_image/RMIT/Original_dataset/' \
                 'Annotation_Pascal_Voc/TFrecord_format/6_class/Meat_label_map_6_classes.pbtxt'
target_name = ['ChunkRoll', 'CubeRoll', 'Rump', 'ShinBeef', 'Striploin', 'Tenderloin']

# Load model
# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile

# Loading label map
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# Load Model
detection_model = tf.saved_model.load(PATH_TO_FROZEN_GRAPH)

# check the model's input
print(detection_model.signatures['serving_default'].inputs)
detection_model.signatures['serving_default'].output_dtypes
detection_model.signatures['serving_default'].output_shapes


# generate boundary box in Deep sort format
# Standard Detection Box format is ['xmin', 'ymin', 'xmax', 'ymax'] which is converted/
# to  (x, y, w, h) where (x, y) is the top-left corner and (w, h) is the extent
def DS_Bb_format(Detection_Bb, Original_image):
    width, height, depth = Original_image.shape
    # Coordinates of detected objects
    xmin = int(Detection_Bb[0] * width)
    ymin = int(Detection_Bb[1] * height)
    xmax = int(Detection_Bb[2] * width)
    ymax = int(Detection_Bb[3] * height)
    New_coor = (ymin, xmin, ymax, xmax)
    return New_coor


# run_inference_for_single_image
def ROI_Extraction(model, image):
    detections = []
    Class_name = []
    # plt.imshow(image)
    # plt.show()
    # Image_name = str(image_path).split('/')[-1]

    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    for index, score in enumerate(output_dict['detection_scores']):
        if score > Score_Th:
            if output_dict['detection_classes'][index] == 1:
                Class = target_name[0]
            elif output_dict['detection_classes'][index] == 2:
                Class = target_name[1]
            elif output_dict['detection_classes'][index] == 3:
                Class = target_name[2]
            elif output_dict['detection_classes'][index] == 4:
                Class = target_name[3]
            elif output_dict['detection_classes'][index] == 5:
                Class = target_name[4]
            elif output_dict['detection_classes'][index] == 6:
                Class = target_name[5]
            new_cor = DS_Bb_format(output_dict['detection_boxes'][index], image)
            # Out = [new_cor[0], new_cor[1], new_cor[2], new_cor[3], round(output_dict['detection_scores'][index], 2)]
            Out = [new_cor[0], new_cor[1], new_cor[2], new_cor[3], int(1)] #use hard detection score ( 1) for the detected object
            detections.append(Out)
            Class_name.append(Class)
    if not detections:
        detections = np.empty((0, 5))
    return np.array(detections), Class_name


def ROX_EX_image(image):
    detections, class_name = ROI_Extraction(detection_model, image)
    return detections, class_name
