import cv2
import os
import numpy as np
from keras.models import Model
from keras.models import load_model
from keras.layers import Input
from yolov3.decoder_layer import make_decoder_layer
from yolov3.yolo_util import compose
from util.bounding_box import get_centroid
from shapely.geometry import Point
from util.logger import get_logger
logger = get_logger()


class YoloDetector(object):
    """
    This is the class of the detection by yolo model.
    """
    _defaults = {
        'model_path': 'model_data/yolov3.h5',
        'anchors_path': 'model_data/coco_anchors.txt',
        'classes_path': 'model_data/coco_classes.txt',
        'height': 416,  # height
        'width': 416,  # width
        'score_threshold': 0.1,  # a box is considered for class c iff confidence times class_prob for c is >= 0.5
        'iou_threshold': 0.4,  # boxes with iou 0.4 or greater are suppressed in nms
        'max_num_boxes': 10  # max number of boxes for a class
    }
    from dotenv import load_dotenv
    load_dotenv(dotenv_path="./env.env")

    def __init__(self, roi_polygon):
        """
        init the detection model by user configuration.
        :param roi_polygon: the roi polygon for the detection.
        """
        # the path of the model location.
        self.model_path = os.getenv("YOLO_MODEL_PATH", default=self._defaults['model_path'])
        self.anchors_path = self._defaults['anchors_path']
        self.classes_path = self._defaults['classes_path']
        self.height = self._defaults['height']
        self.width = self._defaults['width']

        # the score threshold, we will detect only object that have score higher than the threshold
        self.score_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", default=self._defaults['score_threshold']))

        # The threshold of frame size (measured as a percentage of the frame's size) above which a detected box is removed.
        self.percentage_of_frame = float(os.getenv("PERCENTAGE_OF_FRAME_THRESHOLD"))

        # The maximum ratio allowed between width and height of a box , ensuring only proportionate bounding boxes.
        self.width_to_height_ratio = float(os.getenv("MAXIMUM_WIDTH_TO_HEIGHT_RATIO_ALLOWED"))

        # The threshold above which two boxes detected will be considered as the same object, and the one with the lower confidence be removed
        self.iou_threshold = float(os.getenv("IOU_THRESHOLD", default=self._defaults['iou_threshold']))
        self.max_num_boxes = self._defaults['max_num_boxes']
        #
        self.class_names = self._get_class_names()
        self.num_classes = len(self.class_names)
        self.anchors = self._get_anchors()
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = 3
        self.num_scales = 3
        self.roi_polygon = roi_polygon
        # image: run detection on this image/frame
        # other params: describe how to transform image to model input
        self.image = None
        self.image_height = None
        self.image_width = None
        self.image_area = None
        self.scale = None
        self.offset_height = None
        self.offset_width = None
        self.input = None
        # decoded YOLOv3 output
        self.boxes = None
        self.confidence = None
        self.class_probs = None
        self.scores = None

        assert self.num_anchors == self.num_scales * self.num_anchors_per_scale, 'Mismatch of number of anchors'
        self.model = self._get_detection_model()

    def _get_class_names(self):
        """
        return the classes names from the classes file.
        :return: list of the classes names.
        """
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        """
        return the anchors from the anchors file.
        :return: list of the anchors.
        """
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def _get_detection_model(self):
        """
        create the detection model and return it.
        :return: the detection model
        """

        input = Input(shape=(self.height, self.width, 3))
        yolo_model = load_model(self.model_path, compile=False)

        assert len(yolo_model.output) == self.num_scales, \
            'Mismatch between model number of scales and given number of scales.'

        for idx in np.arange(self.num_scales):
            assert yolo_model.output[idx].shape[-1] == self.num_anchors_per_scale * (5 + self.num_classes), \
                'Mismatch between model output length and number of anchors and and number of classes'

        decoder_layer = make_decoder_layer(self.anchors,
                                           self.num_classes,
                                           (self.height, self.width))
        return Model(input, compose(yolo_model, decoder_layer)(input))

    def make_model_input(self, image):
        """
        scale the image to detect before detection.
        :param image: the image to detect
        :return: the scale image
        """
        self.image = image
        self.image_height = image.shape[0]
        self.image_width = image.shape[1]
        self.image_area = self.image_height * self.image_width
        self.scale = min(self.height / self.image_height, self.width / self.image_width)
        tmp_height = int(self.scale * self.image_height)
        tmp_width = int(self.scale * self.image_width)
        self.offset_height = (self.height - tmp_height) // 2
        self.offset_width = (self.width - tmp_width) // 2
        input = np.full((self.height, self.width, 3), 128, np.uint8)
        input[self.offset_height:self.offset_height + tmp_height, self.offset_width:self.offset_width + tmp_width] = \
            cv2.resize(image, (tmp_width, tmp_height))

        input = np.array(input, dtype='float32')
        input /= 255.
        self.input = np.expand_dims(input, 0)  # add batch dimension.

    def run_yolov3(self):
        """
        perform the detection on the image by the model.
        """
        outputs = self.model.predict(self.input, batch_size=1)
        # the second value is 0 because batch size = 1 here for prediction

        self.confidence = outputs[1][0]
        idxs = self.confidence >= self.score_threshold

        self.boxes = (outputs[0][0])[idxs, :]
        self.confidence = np.reshape((outputs[1][0])[idxs], [-1, 1])
        self.class_probs = (outputs[2][0])[idxs, :]
        self.scores = self.confidence * self.class_probs

    def translate_coord(self, box):
        """
        the YOLOv3 model returns y and x coords in the range [0, 1] with respect to the the model height and width
        those corrds need to be translated to the coords of the original image
        :param box: the detected box
        :return: the scaled box
        """

        return \
            int((box[0] * self.width - self.offset_width) / self.scale),\
            int((box[1] * self.height - self.offset_height) / self.scale),\
            int((box[2] * self.width - self.offset_width) / self.scale),\
            int((box[3] * self.height- self.offset_height) / self.scale)

    def get_detections(self, image):
        """
        Returns a list of bounding boxes of peoples detected,
        their classes and the confidences of the detections made.
        :param image: the image to detect
        :return: list of bounding boxes of peoples detected, their classes and the confidences of the detections made.
        """

        self.make_model_input(image)
        # run the detection of the model
        self.run_yolov3()

        # handle bounding boxes
        pick_for_class = self.handle_bounding_boxes()

        bounding_boxes = []
        classes = []
        confidence = []
        for box_idx in pick_for_class:
            x_min, y_min, x_max, y_max = self.translate_coord(self.boxes[box_idx])
            bounding_boxes.append([x_min, y_min, x_max - x_min, y_max - y_min])
            classes.append(self.class_names[0])
            confidence.append(self.class_probs[box_idx, 0])
        return bounding_boxes, classes, confidence

    def handle_bounding_boxes(self):
        """
        for each bounding boxes :
        1. remove bounding box with height / width that are 0.
        2. Remove boxes that are too large relatively to the frame size, or have disproportional width to height ratios
        3. Remove boxes that their centroids are out of the detection roi.
        4. Remove boxes that have iou greater than the provided overlap threshold.
        :return: the picked bounding boxes
        """

        scores = self.scores[:, 0]
        # initialize the list of picked indexes
        pick = []

        # grab the coordinates of the bounding boxes
        x_min = self.boxes[:, 0]
        y_min = self.boxes[:, 1]
        x_max = self.boxes[:, 2]
        y_max = self.boxes[:, 3]

        # compute the area of the bounding boxes and grab the indexes to sort
        # (in the case that no probabilities are provided, simply sort on the
        # bottom-left y-coordinate)
        area = (x_max - x_min) * (y_max - y_min)

        # sort the indexes; note: one could use a priority queue of size max_num_boxes, but that's probably overkill
        idxs = np.argsort(scores)

        # keep looping while some indexes still remain in the indexes list
        while len(idxs) > 0:
            # grab the last index in the indexes list and add the index value
            # to the list of picked indexes
            last = len(idxs) - 1
            i = idxs[last]
            if scores[i] < self.score_threshold:
                return pick

            # Remove boxes that are too large relatively to the frame size, or have disproportional width to height ratios
            x_left, y_bottom, x_right, y_top = self.translate_coord(self.boxes[i])
            width = x_right - x_left
            height = y_top - y_bottom
            if height is 0 or width is 0:
                idxs = np.delete(idxs, last)
                logger.info('Deleted detection with 0 width \ height', extra={'meta': {'cat': 'DETECTION_PROCESS', 'height': height, 'width': width}})
                continue

            box_area = width * height
            percentage_of_frame = box_area / self.image_area
            width_to_height_ratio = width / height

            if percentage_of_frame > self.percentage_of_frame:
                idxs = np.delete(idxs, last)
                logger.info('Deleted detection exceeding percentage of frame threshold', extra={'meta': {'cat': 'DETECTION_PROCESS', 'percentage_of_frame': percentage_of_frame}})
                continue

            if width_to_height_ratio > self.width_to_height_ratio or width_to_height_ratio < (1 / self.width_to_height_ratio):
                idxs = np.delete(idxs, last)
                logger.info('Deleted detection exceeding width to height ratio', extra={'meta': {'cat': 'DETECTION_PROCESS', 'width_to_height_ratio': width_to_height_ratio}})
                continue

            # Remove boxes that their centroids are out of the detection roi.
            centroid_box = get_centroid([x_left, y_bottom, width, height])
            centroid_box_point = Point(centroid_box[0], centroid_box[1])
            if not self.roi_polygon.contains(centroid_box_point):
                idxs = np.delete(idxs, last)
                logger.info('Deleted detection out of DROI',
                            extra={'meta': {'cat': 'DETECTION_PROCESS', 'centroid_box': centroid_box}})
                continue

            pick.append(i)

            if len(pick) == self.max_num_boxes:
                return pick

            # compute the width and height of the intersection of
            # the picked bounding box with all other bounding boxes
            yy_min = np.maximum(y_min[i], y_min[idxs[:last]])
            xx_min = np.maximum(x_min[i], x_min[idxs[:last]])
            yy_max = np.minimum(y_max[i], y_max[idxs[:last]])
            xx_max = np.minimum(x_max[i], x_max[idxs[:last]])
            w = np.maximum(0, xx_max - xx_min)
            h = np.maximum(0, yy_max - yy_min)

            # compute intersection over union
            iou = (w * h) / (area[i] + area[idxs[:last]] - w * h + 1e-5)

            # delete all indexes from the index list that have iou greater
            # than the provided overlap threshold
            idxs = np.delete(idxs, np.concatenate(([last], np.where(iou > self.iou_threshold)[0])))

        return pick
