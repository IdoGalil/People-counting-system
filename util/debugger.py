'''
Utilities for configuring and debugging the people count process.
'''

import os
import ast
import cv2
import pathlib
import uuid
from .logger import get_logger


logger = get_logger()


def mouse_callback(event, x, y, flags, param):
    """
    Handler for mouse events in the debug window.
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        capture_pixel_position(x, y, param['frame_width'], param['frame_height'])


def capture_pixel_position(window_x, window_y, frame_w, frame_h):
    """
    Capture the position of a pixel in a video frame.
    """
    debug_window_size = ast.literal_eval(os.getenv('DEBUG_WINDOW_SIZE'))
    x = round((frame_w / debug_window_size[0]) * window_x)
    y = round((frame_h / debug_window_size[1]) * window_y)
    logger.info('Pixel position captured.', extra={'meta': {'cat': 'DEBUG', 'position': (x, y)}})


def take_screenshot(frame):
    """
    Save frame to file as screenshot.
    """
    screenshots_directory = 'data/screenshots'
    pathlib.Path(screenshots_directory).mkdir(parents=True, exist_ok=True)
    screenshot_path = os.path.join(screenshots_directory, 'img_' + uuid.uuid4().hex + '.jpg')
    cv2.imwrite(screenshot_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

    logger.info('Screenshot captured.', extra={
        'meta': {
            'cat': 'SCREENSHOT_CAPTURE',
            'path': screenshot_path,
        },
    })
