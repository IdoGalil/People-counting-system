from .bounding_box import get_centroid, get_area
from shapely.geometry import Point
from tracker import get_tracker
from util.bounding_box import get_overlap
from util.logger import get_logger
import uuid
logger = get_logger()


class Object:
    def __init__(self, _bounding_box, _type, _confidence, _tracker):
        """
        init a new object that was detected and saves all the object parameters.
        :param _bounding_box: the corrd of the detected object
        :param _type: the class of the object
        :param _confidence: the confidence of the class of the object
        :param _tracker: the tracker of the object
        """
        self.bounding_box = _bounding_box
        self.type = _type
        self.type_confidence = _confidence
        self.centroid = get_centroid(_bounding_box)
        self.centroid_point = Point(self.centroid[0], self.centroid[1])
        self.area = get_area(_bounding_box)
        self.tracker = _tracker
        self.num_consecutive_tracking_failures = 0
        self.num_consecutive_detection_failures = 0
        self.counted = False
        self.just_counted = False
        self.object_color_num_frame = 0
        self.position_first_detected = tuple(self.centroid)
        self.point_first_detected = Point(self.position_first_detected[0], self.position_first_detected[1])

    def update(self, _bounding_box, _type=None, _confidence=None, _tracker=None):
        """
        update the object with the new location, type, confidence and tracker.
        :param _bounding_box: the new corrdinates of the object
        :param _type: the new class of the object
        :param _confidence: the new confidence of the object
        :param _tracker: the new tracker of the object
        """
        self.bounding_box = _bounding_box
        self.type = _type if _type != None else self.type
        self.type_confidence = _confidence if _confidence != None else self.type_confidence
        self.centroid = get_centroid(_bounding_box)
        self.centroid_point = Point(self.centroid[0], self.centroid[1])
        self.area = get_area(_bounding_box)
        if _tracker:
            self.tracker = _tracker


def add_new_objects(boxes, classes, confidences, objects, frame, tracker, mcdf, overlap_threshold):
    """
    This function create new objects by the detected boxes or update existing one.
    decide to update existing one if the overlap between them is grater than the overlap_threshold
    :param boxes: the bounding boxes of the detected objects (the new ones)
    :param classes: the classes of the objects
    :param confidences: the confidence of the objects
    :param objects: list of the objects that exists (that was updated by the tracker)
    :param frame: the frame that we detect.
    :param tracker: the tracker.
    :param mcdf: the maximum detection failures, remove objects that have mcdf greater than this param.
    :param overlap_threshold: the maximum overlap between the detected object (boxes) to the tracked object (from objects) in order to update the object and not create new one.
    :return: the objects.
    """
    matched_object_ids = []
    for i, box in enumerate(boxes):
        _type = classes[i] if classes is not None else None
        _confidence = confidences[i] if confidences is not None else None
        _tracker = get_tracker(tracker, box, frame)

        match_found = False
        for _id, object in objects.items():
            if get_overlap(box, object.bounding_box) >= overlap_threshold:
                match_found = True
                if _id not in matched_object_ids:
                    object.num_consecutive_detection_failures = 0
                    matched_object_ids.append(_id)
                object.update(box, _type, _confidence, _tracker)

                logger.info('Object updated.', extra={
                    'meta': {
                        'cat': 'Object updated',
                        'people_id': _id,
                        'bounding_box': object.bounding_box,
                        'type': object.type,
                        'type_confidence': object.type_confidence,
                    },
                })
                break
        if not match_found:
            # create new object
            _object = Object(box, _type, _confidence, _tracker)
            object_id = generate_people_id()
            objects[object_id] = _object

            logger.info('Object created.', extra={
                'meta': {
                    'cat': 'Object created',
                    'people_id': object_id,
                    'bounding_box': _object.bounding_box,
                    'type': _object.type,
                    'type_confidence': _object.type_confidence,
                },
            })

    objects = remove_stray_objects(objects, matched_object_ids, mcdf)
    return remove_duplicates(objects, overlap_threshold)


def remove_stray_objects(objects, matched_object_ids, mcdf):
    """
    Removes objects that "hang" after a tracked object has left the frame.
    :param objects: list of the objects.
    :param matched_object_ids: list of the is of the objects that was updated
    :param mcdf: the maximum detection failures, remove objects that have mcdf greater than this param.
    :return: the objects.
    """
    for _id, object in list(objects.items()):
        if _id not in matched_object_ids:
            object.num_consecutive_detection_failures += 1
        if object.num_consecutive_detection_failures > mcdf:
            logger.info('People deleted due to detection failures', extra={
                'meta': {
                    'cat': 'PEOPLE_DELETED',
                    'id': _id,
                    'type': object.type,
                    'bounding_box': object.bounding_box,
                    'centroid': object.centroid,
                    'first_detected': object.position_first_detected,
                    'tracker_fail': object.num_consecutive_tracking_failures,
                    'detection_fail': object.num_consecutive_detection_failures,
                    'position_counted': object.counted,
                },
            })
            del objects[_id]
    return objects


def remove_duplicates(objects, overlap_threshold):
    """
    Removes duplicate objects i.e objects that point to an already detected and tracked people.
    :param objects: the detected objects.
    :param overlap_threshold: the maximum overlap between the objects in order to consider one of them as duplicate.
    :return: the objects
    """
    for _id, object_a in list(objects.items()):
        for _, object_b in list(objects.items()):
            if object_a == object_b:
                break
            overlap = get_overlap(object_a.bounding_box, object_b.bounding_box)
            logger.info('overlap info:', extra={
                'meta': {
                    'cat': 'OBJECT_UPDATE',
                    'people_id_a': _id,
                    'bounding_box_a': object_a.bounding_box,
                    'people_id_b': _,
                    'bounding_box_b': object_b.bounding_box,
                    'overlap': overlap
                },
            })
            if overlap >= overlap_threshold and _id in objects:
                logger.info('People tracker delete', extra={
                    'meta': {
                        'cat': 'TRACKER_UPDATE',
                        'people_id': _id,
                        'bounding_box': object_a.bounding_box,
                        'overlap': overlap
                    },
                })
                del objects[_id]
    return objects


def generate_people_id():
    """
    generate id for the object
    """
    return 'id_' + uuid.uuid4().hex
