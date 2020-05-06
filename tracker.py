'''
Functions for keeping track of detected peoples in a video.
'''

import cv2
from util.logger import get_logger


logger = get_logger()


def csrt_create(bounding_box, frame):
    """
    Creates an OpenCV CSRT Tracker object.
    :param bounding_box: the location of the object
    :param frame: frame that the object was detected on
    :return: the tracker
    """
    tracker = cv2.TrackerCSRT_create()
    tracker.init(frame, tuple(bounding_box))
    return tracker


def kcf_create(bounding_box, frame):
    """
    Creates an OpenCV KCF Tracker object.
    :param bounding_box: the location of the object
    :param frame: frame that the object was detected on
    :return: the tracker
    """
    tracker = cv2.TrackerKCF_create()
    tracker.init(frame, tuple(bounding_box))
    return tracker

# ###### untested trackers : ############


def MEDIANFLOW_create(bounding_box, frame):
    """
    Creates an OpenCV MEDIANFLOW Tracker object.
    :param bounding_box: the location of the object
    :param frame: frame that the object was detected on
    :return: the tracker
    """
    tracker = cv2.TrackerMedianFlow_create()
    tracker.init(frame, tuple(bounding_box))
    return tracker


def GOTURN_create(bounding_box, frame):
    """
    Creates an OpenCV GOTURN Tracker object.
    :param bounding_box: the location of the object
    :param frame: frame that the object was detected on
    :return: the tracker
    """
    tracker = cv2.TrackerGOTURN_create()
    tracker.init(frame, tuple(bounding_box))
    return tracker


def MOSSE_create(bounding_box, frame):
    """
    Creates an OpenCV MOSSE Tracker object
    :param bounding_box: the location of the object
    :param frame: frame that the object was detected on
    :return: the tracker
    """
    tracker = cv2.TrackerMOSSE_create()
    tracker.init(frame, tuple(bounding_box))
    return tracker


def TLD_create(bounding_box, frame):
    """
    Creates an OpenCV TLD Tracker object.
    :param bounding_box: the location of the object
    :param frame: frame that the object was detected on
    :return: the tracker
    """
    tracker = cv2.TrackerTLD_create()
    tracker.init(frame, tuple(bounding_box))
    return tracker


def get_tracker(algorithm, bounding_box, frame):
    """
    Fetches a tracker object based on the algorithm specified.
    :param algorithm: the algorithm for the tracker.
    :param bounding_box: the location of the object.
    :param frame: frame that the object was detected on
    :return: the tracker.
    """


    if algorithm == 'csrt':
        return csrt_create(bounding_box, frame)
    elif algorithm == 'kcf':
        return kcf_create(bounding_box, frame)
    elif algorithm == 'medianflow':
        return MEDIANFLOW_create(bounding_box, frame)
    elif algorithm == 'goturn':
        return GOTURN_create(bounding_box, frame)
    elif algorithm == 'mosse':
        return MOSSE_create(bounding_box, frame)
    elif algorithm == 'tld':
        return TLD_create(bounding_box, frame)
    logger.error('Invalid tracking algorithm specified (options: csrt, kcf)', extra={
        'meta': {'cat': 'TRACKER_CREATE'},
    })


