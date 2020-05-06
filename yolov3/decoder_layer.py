import numpy as np
from keras import backend as K
from keras.layers import Lambda


def convert_box_params(b_xy, b_wh):
    b_min = b_xy - (b_wh / 2.0)
    b_max = b_xy + (b_wh / 2.0)
    b_min_max = K.concatenate([b_min, b_max])
    return b_min_max


def make_decoder_layer(all_anchors, num_classes, input_shape):
    # Lambda layer for postprocessing YOLOv3 output
    def decode(yolo_outputs):
        num_scales = len(yolo_outputs)
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_scales == 3 else [[3, 4, 5], [1, 2, 3]]

        b_min_max_list = []
        box_confidence_list = []
        class_probs_list = []

        for scale_idx in np.arange(3):
            anchors = all_anchors[anchor_mask[scale_idx]]
            output = yolo_outputs[scale_idx]
            num_anchors = len(anchors)

            batch_size = K.shape(output)[0]
            grid_shape = K.shape(output)[1:3]
            grid_height = grid_shape[0]  # height
            grid_width = grid_shape[1]  # width

            # reshape to tensor of dimensions batch_size, grid_height, grid_width, num_anchors, 5 + num_classes
            # the five box parameters are:
            #   t_x, t_y determine the center point of the box
            #   t_w, t_h determine the width and height of the box
            #   the box confidence indicates the confidence that box contains an object and box is accurate
            output = K.reshape(output, [-1, grid_height, grid_width, num_anchors, 5 + num_classes])

            # compute b_x, b_y for each cell and each anchor
            c_x = K.tile(K.reshape(K.arange(grid_width),  [1, -1, 1, 1]), [grid_height, 1,          num_anchors, 1])
            c_y = K.tile(K.reshape(K.arange(grid_height), [-1, 1, 1, 1]), [1,           grid_width, num_anchors, 1])
            c_xy = K.concatenate([c_x, c_y])
            c_xy = K.cast(c_xy, K.dtype(output))
            b_xy = (K.sigmoid(output[..., :2]) + c_xy) / K.cast(grid_shape[::-1], K.dtype(output))

            # compute b_w and b_h for each cell and each anchor
            p_wh = K.tile(K.reshape(K.constant(anchors), [1, 1, num_anchors, 2]), [grid_height, grid_width, 1, 1])
            b_wh = p_wh * K.exp(output[..., 2:4]) / K.cast(input_shape[::-1], K.dtype(output))

            b_min_max = K.reshape(convert_box_params(b_xy, b_wh), [batch_size, -1, 4])  # y_min, x_min, y_max, x_max

            # compute box confidence for each cell and each anchor
            box_confidence = K.reshape(K.sigmoid(output[..., 4]), [batch_size, -1])

            # compute class probabilities for each cell and each anchor
            class_probs = K.reshape(K.sigmoid(output[..., 5:]), [batch_size, -1, num_classes])

            b_min_max_list.append(b_min_max)
            box_confidence_list.append(box_confidence)
            class_probs_list.append(class_probs)

        return [
            K.concatenate(b_min_max_list, axis=1),
            K.concatenate(box_confidence_list, axis=1),
            K.concatenate(class_probs_list, axis=1)
        ]

    return Lambda(decode)
