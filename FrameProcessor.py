import cv2
from util.objects import add_new_objects
from collections import OrderedDict
import time
from util.detection_roi import get_roi_frame, draw_roi
from util.logger import get_logger
from counter import get_counting_line, is_passed_counting_line, is_passed_counting_roi
from yolov3.yolo_detection_model import YoloDetector
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point
from util.bounding_box import get_centroid
import requests
logger = get_logger()


class FrameProcessor:
    '''
    @summery: class that is going to count people every frame and show the results on the video.
    this is the main class that holds all the relevant objects for counting.
    '''
    def __init__(self, initial_frame, tracker, droi, show_droi, mcdf, mctf, di, cl_orientation, cl_position,
                 show_counting_roi, counting_roi, counting_roi_outside, frame_number_counting_color, detection_slowdown, roi_object_liveness,
                 show_object_liveness, confidence_threshold, sensitive_confidence_threshold, duplicate_object_threshold, event_api_url):
        '''
        @summery: constractor for counting people. in order to start to count people you should create PeopleCounter object.
        :param initial_frame: the current frame of the video. this value is changing every time we are calling to the "count" method.
        :param tracker: string of the tracker we want to use in order to track people every frame. the valid values are- kcf, csrt, MEDIANFLOW ,MOSSE ,TLD
        :param droi: set of vertices that represent the area (polygon) where you want detections to be made
        :param show_droi: Enable/Disable detection by region
        :param mcdf: Maximum consecutive detection failures i.e number of detection failures before it's concluded that an object is no longer in the frame
        :param mctf: Maximum consecutive tracking failures i.e number of tracking failures before the tracker concludes the tracked object has left the frame
        :param di: Detection interval i.e number of frames before detection is carried out again
        :param cl_orientation: Orientation of counting line (options: top, bottom, left, right)
        :param cl_position: Position of counting line - need to be float number between 0 to 1
        :param show_counting_roi: Display/overlay the counting ROI on the video
        :param counting_roi: set of vertices that represent the area (polygon) where you want counting to be made.if None - will count by line.
        :param counting_roi_outside: orientation of counting region (options: True - the region is the out area , False - the region is inside area)
        :param frame_number_counting_color: number of frames that the counting color (green) will stay on the object that just counted.
        :param detection_slowdown: enable the feature that will do detection every frame when there is object in the frame.
        :param roi_object_liveness: set of vertices that represent the area of the counted objects, if person is out of this area and was counted, we will delete him.
        :param show_object_liveness: Enable/Disable object liveness by region
        :param sensitive_confidence_threshold: A more sensitive confidence threshold meant to be lower until the first detection occurs
        :param duplicate_object_threshold: minimum overlap objects for remove object.
        :param event_api_url: the url to send the event of in/out.
        '''
        self.frame = initial_frame  # current frame of video
        self.tracker = tracker
        self.droi = droi  # detection region of interest
        self.show_droi = show_droi
        self.mcdf = mcdf  # maximum consecutive detection failures
        self.mctf = mctf  # maximum consecutive tracking failures
        self.di = di  # detection interval
        self.cl_orientation = cl_orientation # counting line position
        self.cl_position = cl_position
        self.objects = OrderedDict()
        self.f_height, self.f_width, _ = self.frame.shape
        self.frame_count = 0 # number of frames since last detection
        self.processing_frame_rate = 0 # number of frames processed per second
        self.person_count_in = 0 # number of peoples counted
        self.person_count_out = 0  # number of peoples counted
        self.types_counts = OrderedDict() # counts by people type
        self.counting_line = None if cl_orientation == None else get_counting_line(self.cl_orientation, self.f_width, self.f_height, self.cl_position)
        self.show_counting_roi = show_counting_roi
        self.counting_roi = counting_roi
        self.roi_poligon = Polygon(counting_roi) if counting_roi else None
        self.counting_roi_outside = counting_roi_outside
        self.detection_slowdown = detection_slowdown
        self.confidence_threshold = confidence_threshold
        self.sensitive_confidence_threshold = sensitive_confidence_threshold
        self.detection_roi_polygon = Polygon(droi)
        self.roi_object_liveness = roi_object_liveness if roi_object_liveness else None
        self.roi_object_liveness_polygon = Polygon(roi_object_liveness) if roi_object_liveness else None
        self.show_object_liveness = show_object_liveness
        self.event_api_url = event_api_url

        #create net object for detection
        self.net = YoloDetector(self.detection_roi_polygon)

        # threshold for remove duplicate objects
        self.duplicate_object_threshold = duplicate_object_threshold

        # number of the frames left to show the counted person in different color.
        self.roi_color_num_frame = 0

        # number of frame to show the counted person in different color.
        self.frame_number_counting_color = frame_number_counting_color

        # save the order of the counting
        self.count_order = []

        self.detect()
    # ------------------------------------------------------------------------------------

    def track_and_detect(self, frame):
        """
         this function do the following steps-
            for each object:
                update location and count.
            preform detection on the frame if we pass amount of frame == DI .
            preform detection on the frame and increase the confidence of detection if detection_slowdown feature is enable and there is object in the frame.
            decrease the confidence of detection if if detection_slowdown feature is enable and there is no object in the frame.
        :param frame: the current frame.
        """

        _timer = cv2.getTickCount() # set timer to calculate processing frame rate

        self.frame = frame

        for _id, object in list(self.objects.items()):
            self.handle_object(object, _id)

        # do detection if we pass the amount of frame till detection
        if self.frame_count >= self.di:
            # return detection
            self.detect()

        # increase the confidence of detection and decrease the di so we will do detection every frame if detection_slowdown feature is enable and there is object in the frame
        if len(self.objects) > 0 and self.detection_slowdown:
            self.frame_count = self.di
            self.net.score_threshold = self.confidence_threshold
        # decrease the confidence of detection if if detection_slowdown feature is enable and there is no object in the frame
        elif self.detection_slowdown:
            self.net.score_threshold = self.sensitive_confidence_threshold

        self.frame_count += 1

        # calculate frame rate
        self.processing_frame_rate = round(cv2.getTickFrequency() / (cv2.getTickCount() - _timer), 2)
        logger.debug('Processing frame rate updated.', extra={
            'meta': {'cat': 'PROCESSING_SPEED', 'frame_rate': self.processing_frame_rate},
        })

    # ------------------------------------------------------------------------------------

    def detect(self):
        """
        this function detect objects in the frame by the following-
            1. detect people only in the detection ROI.
            2. add the detected people (bounding boxes) to the objects list.
            3. remove all the duplication between the objects (old and new) by the overlap threshold.
            4. reset the frame counting
        """
        droi_frame = get_roi_frame(self.frame, self.droi)
        _bounding_boxes, _classes, _confidences = self.net.get_detections(droi_frame)
        self.objects = add_new_objects(_bounding_boxes, _classes, _confidences, self.objects, self.frame, self.tracker, self.mcdf, self.duplicate_object_threshold)
        self.frame_count = 0
    # ------------------------------------------------------------------------------------

    def handle_object(self, object, object_id):
        """
        this function purpose is to update the location of the object, count it and delete it if necessary.
        this function do the following steps-
            1. update the location of the object by the tracker.
            2. count object if the object pass the line/area.
            3. delete object if it pass the max tracking failures.
            4. delete object if the object was counted and not in the liveness area.
        :param object: object to handel
        :param object_id: int that represent the object id
        """
        # update trackers
        success, box = object.tracker.update(self.frame)

        # if we success track the object
        if success:
            centroid_box = get_centroid(box)
            centroid_point = Point(centroid_box[0], centroid_box[1])
            if self.detection_roi_polygon.contains(centroid_point):
                object.num_consecutive_tracking_failures = 0
                object.update(box)
                logger.info('People tracker updated.', extra={
                    'meta': {
                        'cat': 'TRACKER_UPDATE',
                        'people_id': object_id,
                        'bounding_box': object.bounding_box,
                        'centroid': object.centroid,
                        'first_detected': object.position_first_detected,
                        'tracker_fail': object.num_consecutive_tracking_failures,
                        'detection_fail': object.num_consecutive_detection_failures
                    },
                })
            else:
                object.num_consecutive_tracking_failures += 1
        else:
            object.num_consecutive_tracking_failures += 1

        # inform to the logger if we count this object
        if (self.counting_roi and self.count_by_roi(object)) or (not self.counting_roi and self.count_by_line(object)):

            # information to the logger
            logger.info('People counted.', extra={
                'meta': {
                    'cat': 'PEOPLE_COUNT',
                    'id': object_id,
                    'type': object.type,
                    'count_in': self.person_count_in,
                    'count_out': self.person_count_out,
                    'position_first_detected': object.position_first_detected,
                    'centroid': object.centroid,
                    'position_counted': object.counted,
                    'counted_at': time.time(),
                },
            })


        # delete object if it pass the max tracking failures
        if object.num_consecutive_tracking_failures >= self.mctf:
            logger.info('People deleted due to tracking failures', extra={
                'meta': {
                    'cat': 'PEOPLE_DELETED',
                    'id': object_id,
                    'type': object.type,
                    'bounding_box': object.bounding_box,
                    'centroid': object.centroid,
                    'first_detected': object.position_first_detected,
                    'tracker_fail': object.num_consecutive_tracking_failures,
                    'detection_fail': object.num_consecutive_detection_failures,
                    'position_counted': object.counted,
                },
            })
            del self.objects[object_id]

        # delete object if it was counted and pass the area of object_liveness
        if object.counted and self.roi_object_liveness and not (self.roi_object_liveness_polygon.contains(object.centroid_point)):
            del self.objects[object_id]

    # ------------------------------------------------------------------------------------

    def count_by_roi(self, object):
        """
        this function count the object if the object exit/enter the area of the roi.
        :param object: object to count
        :return: True if the object count or False if not.
        """
        counted = False
        # count the object if the object didn't count and the object exit the area that it was first detected
        if not object.counted and (self.roi_poligon.contains(object.centroid_point) ^ self.roi_poligon.contains(object.point_first_detected)):
            object.counted = True
            counted = True
            # count the object as person_count_in if the object enter to the counting roi.
            if is_passed_counting_roi(object.centroid_point, object.point_first_detected, self.roi_poligon, self.counting_roi_outside, is_enter=True):
                self.person_count_in += 1
                self.count_order.append("_In")
                data = {'in': True}
            else:
                self.person_count_out += 1
                self.count_order.append("_Out")
                data = {'out': True}

            requests.post(url=self.event_api_url, data=data)
        return counted

    # ------------------------------------------------------------------------------------

    def count_by_line(self, object):
        """
        this function count the object if the object passed the line by the line orientation.
        count the object if the current object location pass the line by the line orientation and didn't was in this side of line in the first detection.
        :param object: object to count
        :return: True if the object count or False if not.
        """
        counted = False
        # if counting line exists count the peoples that have passed the counting line and didn't count yet.
        if (self.counting_line and not object.counted and (is_passed_counting_line(object.centroid, self.counting_line, self.cl_orientation) ^
                                                           is_passed_counting_line(object.position_first_detected, self.counting_line,
                                                                                   self.cl_orientation))):
            object.counted = True
            counted = True
            # count the object as person_count_in if the person passed the line by the line orientation.
            if not is_passed_counting_line(object.position_first_detected, self.counting_line, self.cl_orientation):
                self.person_count_in += 1
                data = {'in': True}
            else:
                self.person_count_out += 1
                data = {'out': True}
            requests.post(url=self.event_api_url , data=data)
        return counted

    # ------------------------------------------------------------------------------------

    def visualize(self):
        """
        visualize the objects and the counting information on the frame.
        colors-
            counting roi: yellow.
            box:blue.
            detection roi: orange.
        if we just count person , the box and the counting line/roi will be in green for self.frame_number_counting_color frames.
        :return: frame
        """
        blue = (255, 0, 0)
        yellow = (0, 255, 255)
        green = (0, 255, 0)
        orange = (0, 50, 252)
        frame = self.frame
        # blue color of droi
        color = blue

        # yellow color for the counting roi
        counting_roi_color = yellow

        # draw and label object bounding boxes
        for _id, object in self.objects.items():
            (x, y, w, h) = [int(v) for v in object.bounding_box]
            # blue color to the box
            color = blue
            if object.counted and not object.just_counted:
                color = green
                counting_roi_color = green
                object.object_color_num_frame = self.frame_number_counting_color
                self.roi_color_num_frame = self.frame_number_counting_color
                object.just_counted = True
            elif object.object_color_num_frame > 0:
                color = green
                object.object_color_num_frame -= 1

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            people_label = 'ID: ' + _id[:8] \
                            if object.type == None \
                            else 'ID: {0}, {1} ({2}%)'.format(_id[:8], object.type, str(object.type_confidence*100)[:4])
            cv2.putText(frame, people_label, (x, y - 5), cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 2, cv2.LINE_AA)

        # draw counting line
        if self.counting_line is not None and self.counting_roi is None:
            cv2.line(frame, self.counting_line[0], self.counting_line[1], color, 3)

        # display people count
        types_counts_str = ', '.join([': '.join(map(str, i)) for i in self.types_counts.items()])
        types_counts_str = ' (' + types_counts_str + ')' if types_counts_str != '' else types_counts_str
        cv2.putText(frame, 'Count in: ' + str(self.person_count_in) + types_counts_str, (20, 60), cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, 'Count out: ' + str(self.person_count_out) + types_counts_str, (20, 120), cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, 'Processing speed: ' + str(self.processing_frame_rate) + ' FPS', (20, 180), cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 0, 0), 2, cv2.LINE_AA)

        # show detection roi
        if self.show_droi:
            frame = draw_roi(frame, self.droi, orange)

        # show counting roi
        if self.show_counting_roi:
            if self.roi_color_num_frame > 0:
                counting_roi_color = green
                self.roi_color_num_frame -= 1
            frame = draw_roi(frame, self.counting_roi, counting_roi_color)

        # show liveness roi
        if self.show_object_liveness and self.roi_object_liveness:
            frame = draw_roi(frame, self.roi_object_liveness, blue)

        return frame
