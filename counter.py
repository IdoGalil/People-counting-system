
def get_counting_line(line_orientation, frame_width, frame_height, line_position):
    """
    This function return the cordd of the counting line by the line position and the frame width and height.
    :param line_orientation: string of the orientation of the line.need to be top, bottom, left, right. example- if right - the right side it the outside.
    :param frame_width: the width of the frame
    :param frame_height: the height of the frame
    :param line_position: the position of the line in the frame (flout number)
    :return: corrdinates list of the area.
    """
    line_orientations_list = ['top', 'bottom', 'left', 'right']
    if line_orientation not in line_orientations_list:
        raise Exception('Invalid line position specified (options: top, bottom, left, right)')

    if line_orientation == 'top':
        counting_line_y = round(line_position * frame_height)
        return [(0, counting_line_y), (frame_width, counting_line_y)]
    elif line_orientation == 'bottom':
        counting_line_y = round(line_position * frame_height)
        return [(0, counting_line_y), (frame_width, counting_line_y)]
    elif line_orientation == 'left':
        counting_line_x = round(line_position * frame_width)
        return [(counting_line_x, 0), (counting_line_x, frame_height)]
    elif line_orientation == 'right':
        counting_line_x = round(line_position * frame_width)
        return [(counting_line_x, 0), (counting_line_x, frame_height)]


def is_passed_counting_line(point, counting_line, line_orientation):
    """
    will check if the point passed the counting line by the x corrd if it left/right or y corrd if it bottom/top.
    :param point: the object location.
    :param counting_line: corrdinates list of the area.
    :param line_orientation: string of the orientation of the line.need to be top, bottom, left, right.
    :return: true if the point passed the line , False if the point didnt pass the line.
    """
    if line_orientation == 'top':
        return point[1] < counting_line[0][1]
    elif line_orientation == 'bottom':
        return point[1] > counting_line[0][1]
    elif line_orientation == 'left':
        return point[0] < counting_line[0][0]
    elif line_orientation == 'right':
        return point[0] > counting_line[0][0]


def is_passed_counting_roi(current_point, first_point, roi_polygon, roi_outside, is_enter):
    """
    will check if the point passed the counting roi. if the current location in the roi and the first location not in the roi - it passed
    if the current location not in the roi and the first location in the roi - it passed.
    :param current_point: Point object for the current location
    :param first_point: Point object for the first detected location.
    :param roi_polygon: Polygon object for the counting roi.
    :param roi_outside: True if the roi if the outside.
    :param is_enter: True if we want to check if the person enter to the roi and false if we want to check if the person left the roi.
    :return: True if the point passed the roi , False if the point not passed the roi.
    """
    if roi_outside ^ is_enter:
        return roi_polygon.contains(current_point) and not roi_polygon.contains(first_point)
    else:
        return roi_polygon.contains(first_point) and not roi_polygon.contains(current_point)
