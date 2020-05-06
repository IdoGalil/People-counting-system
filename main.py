'''
detecting people that enter and exiting from librarys in the Technion.
'''
import os
import ast
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)


def run():
    '''
    Initialize counter class and run counting loop.
    '''
    import sys
    import time
    import cv2

    from util.logger import get_logger
    from FrameProcessor import FrameProcessor
    from util.debugger import mouse_callback, take_screenshot
    from keras import backend as K
    logger = get_logger()

    # IGNORE WARNINGS:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


    # capture live camera
    is_cam = ast.literal_eval(os.getenv('IS_CAM'))
    if is_cam:
        video = os.getenv('CONNECTION')
        cap = cv2.VideoCapture(video)
    else:
        video = os.getenv('VIDEO')
        cap = cv2.VideoCapture(video)

    # if didn't successed open the video/live camera
    if not cap.isOpened():
        if is_cam:
            error = 'Error in connecting to the camera . Invalid source.'
        else:
            error = 'Error capturing video. Invalid source.'
        logger.error(error, extra={
            'meta': {
                'cat': 'VIDEO_CAPTURE',
                'source': video,
            },
        })
        sys.exit(0)

    # start reading the frames
    ret, frame = cap.read()
    f_height, f_width, _ = frame.shape

    # ####load configuration from the env file#####
    # ##detection configuratio###
    detection_slowdown = ast.literal_eval(os.getenv('DETECTION_SLOWDOWN'))
    detection_interval = int(os.getenv('DI'))
    mcdf = int(os.getenv('MCDF'))
    detector = os.getenv('DETECTOR')

    # create detection region of interest polygon#
    use_droi = ast.literal_eval(os.getenv('USE_DROI'))
    droi = ast.literal_eval(os.getenv('DROI')) \
            if use_droi \
            else [(0, 0), (f_width, 0), (f_width, f_height), (0, f_height)]
    show_droi = ast.literal_eval(os.getenv('SHOW_DROI'))

    # confidence threshold of detection#
    confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD"))
    if not (0 < confidence_threshold < 1):
        logger.error('Error confidence threshold should be between 0 to 1', extra={
            'meta': {
                'cat': 'CONF_THRESHOLD',
                'confidence_threshold': confidence_threshold,
            },
        })
        sys.exit(0)

    # A more sensitive confidence threshold meant to be lower until the first detection occurs
    sensitive_confidence_threshold = float(os.getenv("SENSITIVE_CONFIDENCE_THRESHOLD"))
    if not (0 < sensitive_confidence_threshold < 1):
        logger.error('Error sensitive confidence threshold should be between 0 to 1', extra={
            'meta': {
                'cat': 'CONF_THRESHOLD',
                'sensitive_confidence_threshold': sensitive_confidence_threshold,
            },
        })
        sys.exit(0)

    ###tracking configuration###
    mctf = int(os.getenv('MCTF'))
    tracker = os.getenv('TRACKER')

    # threshold for remove duplicate objects
    duplicate_object_threshold = float(os.getenv('OVERLAP_THRESHOLD'))
    if not (0 < duplicate_object_threshold < 1):
        logger.error('Error duplicate_object_threshold should be between 0 to 1', extra={
            'meta': {
                'cat': 'DUP_OBJECT_THRESHOLD',
                'duplicate_object_threshold': duplicate_object_threshold,
            },
        })
        sys.exit(0)

    ###counting configuration###
    #create counting region of interest polygon#
    use_counting_roi = ast.literal_eval(os.getenv('USE_COUNT_ROI'))
    counting_roi = ast.literal_eval(os.getenv('COUNTING_ROI')) if use_counting_roi else None
    show_roi_counting = ast.literal_eval(os.getenv('SHOW_COUNT_ROI'))
    counting_roi_outside = ast.literal_eval(os.getenv('COUNTING_ROI_OUTSIDE'))
    #counting by line#
    counting_line_orientation = os.getenv('COUNTING_LINE_ORIENTATION')
    counting_line_position = float(os.getenv('COUNTING_LINE_POSITION'))
    #create liveness of counted objects#
    use_object_liveness = ast.literal_eval(os.getenv('ENABLE_OBJECT_LIVENESS'))
    roi_object_liveness = ast.literal_eval(os.getenv('OBJECT_LIVENESS_ROI')) if use_object_liveness else None
    show_object_liveness = ast.literal_eval(os.getenv('SHOW_OBJECT_LIVENESS'))

    #if the counting line position is not valid#
    if not (0 < counting_line_position < 1):
        logger.error('Error counting line position. need to be between 0 to 1', extra={
            'meta': {
                'cat': 'VIDEO_CAPTURE',
                 'counting line position': counting_line_position,
            },
        })
        sys.exit(0)

    # number of frame to be with different color from the moment the object was counted#
    frame_number_counting_color = int(os.getenv('COLOR_CHANGE_INTERVAL_FOR_COUNTING_LINE'))

    event_api_url = os.getenv('EVENT_API_URL')

    # ##output configuration###
    record = ast.literal_eval(os.getenv('RECORD'))
    UI = ast.literal_eval(os.getenv('UI'))
    debug = ast.literal_eval(os.getenv('DEBUG'))
    # ####### create people counter obejct ########
    people_counter = FrameProcessor(frame, tracker, droi, show_droi, mcdf,
                                     mctf, detection_interval, counting_line_orientation, counting_line_position,
                                   show_roi_counting, counting_roi, counting_roi_outside, frame_number_counting_color,
                                   detection_slowdown, roi_object_liveness, show_object_liveness, confidence_threshold, sensitive_confidence_threshold,
                                   duplicate_object_threshold, event_api_url)

    # if need to record the video
    if record:
        output_name ="output.avi"
        output_video = cv2.VideoWriter(os.getenv('OUTPUT_VIDEO_PATH') + output_name,
                                       cv2.VideoWriter_fourcc(*'MJPG'),
                                        30, (f_width, f_height))

    logger.info('Processing started.', extra={
        'meta': {
            'cat': 'COUNT_PROCESS',
            'counter_config': {
                'di': detection_interval,
                'mcdf': mcdf,
                'mctf': mctf,
                'detector': detector,
                'tracker': tracker,
                'use_droi': use_droi,
                'droi': droi,
                'show_droi': show_droi,
                'counting_line_orientation': counting_line_orientation,
                'counting_line_position': counting_line_position
            },
        },
    })
    # capture mouse events in the debug window
    cv2.namedWindow('Debug')
    cv2.setMouseCallback('Debug', mouse_callback, {'frame_width': f_width, 'frame_height': f_height})

    is_paused = False
    output_frame = None
    start_time = time.time()

    # ##########main loop ###############
    while is_cam or cap.get(cv2.CAP_PROP_POS_FRAMES) + 1 < cap.get(cv2.CAP_PROP_FRAME_COUNT):
        if debug:
            k = cv2.waitKey(1) & 0xFF
            # pause/play loop if 'p' key is pressed
            if k == ord('p'):
                is_paused = False if is_paused else True
                logger.info('Loop paused/played.', extra={'meta': {'cat': 'COUNT_PROCESS', 'is_paused': is_paused}})
            # save frame if 's' key is pressed
            if k == ord('s') and output_frame is not None:
                take_screenshot(output_frame)
            # end video loop if 'q' key is pressed
            if k == ord('q'):
                logger.info('Loop stopped.', extra={'meta': {'cat': 'COUNT_PROCESS'}})
                break

        if is_paused:
            time.sleep(0.5)
            continue

        # count people and show it in the video
        if ret:
            people_counter.track_and_detect(frame)
            output_frame = people_counter.visualize()

        if record:
             output_video.write(output_frame)

        # if we run it with UI display
        if UI:
             debug_window_size = ast.literal_eval(os.getenv('DEBUG_WINDOW_SIZE'))
             resized_frame = cv2.resize(output_frame, debug_window_size)
             cv2.imshow('Debug', resized_frame)

        ret, frame = cap.read()

    end_time = time.time()
    total_time = str(end_time-start_time)
    print("total time = " + total_time)
    print("total in : {0} \n total out {1}\n".format(str(people_counter.person_count_in), str(people_counter.person_count_out)))
    # end capture, close window, close log file and video object if any
    cap.release()
    if UI:
        cv2.destroyAllWindows()
    if record:
        output_video.release()
    #del people_counter.net.model
    # clear all old sessions
    K.clear_session()
    return people_counter.person_count_in,people_counter.person_count_out, total_time, people_counter.count_order


def run_test():

    from xlwt import Workbook

    test_dict = {"Architecture": {
                    "USE_COUNT_ROI": 'False',
                    "USE_DROI": 'False',
                    'COUNTING_LINE_ORIENTATION': "bottom",
                    'COUNTING_LINE_POSITION': "0.5",
                    "DI_EVERY_FRAME": ['True', 'False'],
                    "MCDF_list": [['10'], ['4', '6']]
                },
                "medical": {
                        "USE_COUNT_ROI": 'True',
                        "COUNTING_ROI": '[(1,3), (173, 87), (172, 214), (965, 214), (1279,0)]',
                        "USE_DROI": 'True',
                        "DROI": "[(188, 4), (158,202), (0, 208), (0, 718), (1162, 718), (1208, 284), (948, 256), (940, 0)]",
                        "DI_EVERY_FRAME": ['True', 'False'],
                        "MCDF_list": [['6', '7'], ['4', '5']]
                },
                "Electrical": {
                    "USE_COUNT_ROI": 'False',
                    "USE_DROI": 'False',
                    'COUNTING_LINE_ORIENTATION': "right",
                    'COUNTING_LINE_POSITION': "0.5",
                    "DI_EVERY_FRAME": ['True', 'False'],
                    "MCDF_list": [['6', '4'], ['3']]
                },
                "Central": {
                    "USE_COUNT_ROI": 'False',
                    "USE_DROI": 'False',
                    'COUNTING_LINE_ORIENTATION': "right",
                    'COUNTING_LINE_POSITION': "0.61",
                    "DI_EVERY_FRAME": ['True', 'False'],
                    "MCDF_list": [['7', '12', '18'], ['6']]
                },
                "Mechanical": {
                    "USE_COUNT_ROI": 'True',
                    "COUNTING_ROI": '[(418, 7), (730, 567), (1112, 542), (1279, 1), (1060, 4)]',
                    "USE_DROI": 'True',
                    "DI_EVERY_FRAME": ['True', 'False'],
                    "MCDF_list": [['20'], ['9']]
                }
                }

    for video in test_dict.keys():
        os.environ['VIDEO'] = "./data/videos/{0}_cut.mp4".format(video)
        os.environ['USE_COUNT_ROI'] = test_dict[video]["USE_COUNT_ROI"]
        os.environ['USE_DROI'] = test_dict[video]["USE_DROI"]
        if ast.literal_eval(os.getenv('USE_DROI')):
            os.environ['DROI'] = test_dict[video]["DROI"]
        if ast.literal_eval(os.getenv('USE_COUNT_ROI')):
            os.environ['COUNTING_ROI'] = test_dict[video]["COUNTING_ROI"]
        else:
            os.environ['COUNTING_LINE_ORIENTATION'] =  test_dict[video]["COUNTING_LINE_ORIENTATION"]
            os.environ['COUNTING_LINE_POSITION'] = test_dict[video]["COUNTING_LINE_POSITION"]

        for detector in ["yolov3-416", "yolov3-608"]:
            os.environ['YOLO_KERAS_MODEL_PATH'] = "./model_data/{0}.h5".format(detector)
            for tracker in ["csrt", "MEDIANFLOW", "kcf"]:
                os.environ['TRACKER'] = tracker
                # Workbook is created
                wb = Workbook()
                try:
                    # add_sheet is used to create sheet.
                    sheet1 = wb.add_sheet('Sheet 1')
                    sheet1.write(0, 0, str(test_dict[video]))
                    sheet1.write(1, 0, 'Tracker')
                    sheet1.write(1, 1, 'DI')
                    sheet1.write(1, 2, 'Remove_overlap')
                    sheet1.write(1, 3, 'confidence')
                    sheet1.write(1, 4, 'di_every_frame')
                    sheet1.write(1, 5, 'MCDF')
                    sheet1.write(1, 6, 'MCTF')
                    sheet1.write(1, 7, 'Total_in')
                    sheet1.write(1, 8, 'Total_out')
                    sheet1.write(1, 9, 'count_order')
                    sheet1.write(1, 10, 'Total_time')
                    wb.save('./{0}_{1}_{2}.xls'.format(video, detector, tracker))
                    line = 2
                    for DI in ['3']:
                        os.environ['DI'] = DI
                        for remove_overlap in ['0.05', '0.1', '0.3', '0.5']:
                            os.environ['OVERLAP_THRESHOLD'] = remove_overlap
                            for confidence in ['0.1']:
                                os.environ['YOLO_KERAS_CONFIDENCE_THRESHOLD'] = confidence
                                for di_every_frame, MCDF_list in zip(test_dict[video]['DI_EVERY_FRAME'], test_dict[video]['MCDF_list']):
                                    os.environ['DI_EVERY_FRAME'] = di_every_frame

                                    for MCDF in MCDF_list:
                                        os.environ['MCDF'] = MCDF

                                        for MCTF in ['5', '2']:
                                            os.environ['MCTF'] = MCTF

                                            print(video, tracker, DI, remove_overlap, confidence,
                                                  di_every_frame, MCDF, MCTF)
                                            person_count_in, person_count_out, total_time, count_order = run()
                                            for coulmn, value in zip(range(0, 11),
                                                                 [tracker, DI, remove_overlap, confidence,
                                                                  di_every_frame, MCDF, MCTF, person_count_in, person_count_out,
                                                                  count_order, total_time]):

                                                sheet1.write(line, coulmn, value)
                                            line += 1
                                            wb.save('./{0}_{1}_{2}.xls'.format(video, detector, tracker))
                finally:
                    wb.save('./{0}_{1}_{2}.xls'.format(video, detector, tracker))


if __name__ == '__main__':
    from dotenv import load_dotenv
    #Or change
    load_dotenv(dotenv_path="./env.env")

    from util.logger import init_logger
    init_logger()
    #run_test()
    run()
