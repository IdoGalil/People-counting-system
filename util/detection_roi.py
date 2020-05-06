import numpy as np
import cv2


def get_roi_frame(current_frame, polygon):
    """
    return masked frame by roi (polygon)
    :param current_frame: the frame to mask
    :param polygon: the roi to ignore
    :return: masked frame
    """
    mask = np.zeros(current_frame.shape, dtype=np.uint8)
    polygon = np.array([polygon], dtype=np.int32)
    num_frame_channels = current_frame.shape[2]
    mask_ignore_color = (255,) * num_frame_channels
    cv2.fillPoly(mask, polygon, mask_ignore_color)
    masked_frame = cv2.bitwise_and(current_frame, mask)
    return masked_frame


def draw_roi(frame, polygon, color = (0, 255, 255)):
    """
    draw the roi on a frame by polygon.
    :param frame: frame to draw on.
    :param polygon: the polygon to draw.
    :param color: the color of the drawing.
    :return: the frame with the roi.
    """
    frame_overlay = frame.copy()
    polygon = np.array([polygon], dtype=np.int32)
    cv2.fillPoly(frame_overlay, polygon, color)
    alpha = 0.3
    output_frame = cv2.addWeighted(frame_overlay, alpha, frame, 1 - alpha, 0)
    return output_frame
