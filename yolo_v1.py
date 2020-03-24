import tensorflow as tf
import numpy as np


S = 7  # cell size
B = 2  # boxes_per_cell
C = 20  # number of classes


def leak_relu(inputs, alpha=0.1):
    return tf.maximum(alpha * inputs, inputs)


def conv_op(inputs, id_, num_filters, filter_size, stride):
    inputs_pad = tf.pad(inputs, np.array([[0, 0], [filter_size // 2, filter_size // 2], [filter_size // 2, filter_size // 2], [0, 0]]))

    in_channels = inputs.get_shape().as_list()[-1]
    weight = tf.Variable(tf.truncated_normal(shape=[filter_size, filter_size, in_channels, num_filters], stddev=0.1, dtype=tf.float32))
    bias = tf.Variable(tf.constant(shape=[num_filters], value=0.0, dtype=tf.float32))

    conv = tf.nn.conv2d(inputs_pad, weight, strides=[1, stride, stride, 1], padding="VALID")
    outputs = leak_relu(tf.nn.bias_add(conv, bias))
    print("    Layer %d: type=Conv, num_filter=%d, filter_size=%d, stride=%d, output_shape=%s"
          % (id_, num_filters, filter_size, stride, str(outputs.get_shape())))
    return outputs


def maxpool_op(inputs, id_, pool_size, stride):
    outputs = tf.nn.max_pool(inputs, [1, pool_size, pool_size, 1], strides=[1, stride, stride, 1], padding="SAME")
    print("    Layer %d: type=MaxPool, pool_size=%d, stride=%d, output_shape=%s"
          % (id_, pool_size, stride, str(outputs.get_shape())))
    return outputs


def flatten_op(inputs, id_):
    length = np.product(inputs.get_shape().as_list()[1:])
    channel_first = tf.transpose(inputs, [0, 3, 1, 2])  # channel first mode
    outputs = tf.reshape(channel_first, [-1, length])
    print("    Layer %d: type=Flatten, output_shape=%s"
          % (id_, str(outputs.get_shape())))
    return outputs


def fc_op(inputs, id_, num_out, activation=None):
    num_in = inputs.get_shape().as_list()[-1]
    weight = tf.Variable(tf.truncated_normal(shape=[num_in, num_out], stddev=0.1, dtype=tf.float32))
    bias = tf.Variable(tf.constant(shape=[num_out], value=0.0, dtype=tf.float32))
    outputs = tf.nn.xw_plus_b(inputs, weight, bias)
    if activation:
        outputs = activation(outputs)
    print("    Layer %d: type=Fc, num_out=%d, output_shape=%s"
          % (id_, num_out, str(outputs.get_shape())))
    return outputs


def convert_result(prediction, threshold, iou_threshold, max_output_size):

    # class probability
    class_prob = tf.reshape(prediction[0, :S * S * C], [S, S, C])
    # confidence
    confidence = tf.reshape(prediction[0, S * S * C: S * S * (C + B)], [S, S, B])
    # boxes -> (x, y, w, h)
    boxes = tf.reshape(prediction[0, S * S * (C + B):], [S, S, B, 4])

    # convert the x, y to the coordinates relative to the top left point of the image
    # the predictions of w, h are the square root
    # multiply the width and height of image
    x_offset = np.transpose(np.reshape(np.array([np.arange(S)] * S * B), [B, S, S]), [1, 2, 0])
    y_offset = np.transpose(x_offset, [1, 0, 2])
    boxes = tf.stack([(boxes[:, :, :, 0] + tf.constant(x_offset, dtype=tf.float32)) / S,
                      (boxes[:, :, :, 1] + tf.constant(y_offset, dtype=tf.float32)) / S,
                      tf.square(boxes[:, :, :, 2]),
                      tf.square(boxes[:, :, :, 3])], axis=3)

    # class-specific confidence scores [S, S, B, C]
    scores = tf.expand_dims(confidence, 3) * tf.expand_dims(class_prob, 2)
    scores = tf.reshape(scores, [-1, C])  # [S*S*B, C]
    boxes = tf.reshape(boxes, [-1, 4])  # [S*S*B, 4]

    # find each box class, only select the max score
    box_classes = tf.argmax(scores, axis=1)
    box_class_scores = tf.reduce_max(scores, axis=1)

    # filter the boxes by the score threshold
    filter_mask = box_class_scores >= threshold
    scores = tf.boolean_mask(box_class_scores, filter_mask)

    boxes = tf.boolean_mask(boxes, filter_mask)
    box_classes = tf.boolean_mask(box_classes, filter_mask)

    # non max suppression (do not distinguish different classes)
    # box (x, y, w, h) -> _box (x1, y1, x2, y2)
    _boxes = tf.stack([boxes[:, 0] - boxes[:, 2] / 2, boxes[:, 1] - boxes[:, 3] / 2,
                       boxes[:, 0] + boxes[:, 2] / 2, boxes[:, 1] + boxes[:, 3] / 2], axis=1)
    nms_indices = tf.image.non_max_suppression(_boxes, scores, max_output_size, iou_threshold)
    scores = tf.gather(scores, nms_indices)
    boxes = tf.gather(boxes, nms_indices)
    box_classes = tf.gather(box_classes, nms_indices)
    return scores, boxes, box_classes


class YoloV1(object):
    def __init__(self):
        self.images = tf.placeholder(tf.float32, [None, 448, 448, 3])

        self.threshold = 0.25  # confidence scores threshold
        self.iou_threshold = 0.4  # iou threshold in non max suppression
        self.max_output_size = 10  # the maximum number of boxes to be selected by non max suppression

        self.prediction = self.inference_op()
        self.scores, self.boxes, self.box_classes = convert_result(threshold=self.threshold,
                                                                   iou_threshold=self.iou_threshold,
                                                                   max_output_size=self.max_output_size,
                                                                   prediction=self.prediction)

    def inference_op(self):
        net = conv_op(self.images, 1, 64, 7, 2)
        net = maxpool_op(net, 2, 2, 2)
        net = conv_op(net, 3, 192, 3, 1)
        net = maxpool_op(net, 4, 2, 2)
        net = conv_op(net, 5, 128, 1, 1)
        net = conv_op(net, 6, 256, 3, 1)
        net = conv_op(net, 7, 256, 1, 1)
        net = conv_op(net, 8, 512, 3, 1)
        net = maxpool_op(net, 9, 2, 2)
        net = conv_op(net, 10, 256, 1, 1)
        net = conv_op(net, 11, 512, 3, 1)
        net = conv_op(net, 12, 256, 1, 1)
        net = conv_op(net, 13, 512, 3, 1)
        net = conv_op(net, 14, 256, 1, 1)
        net = conv_op(net, 15, 512, 3, 1)
        net = conv_op(net, 16, 256, 1, 1)
        net = conv_op(net, 17, 512, 3, 1)
        net = conv_op(net, 18, 512, 1, 1)
        net = conv_op(net, 19, 1024, 3, 1)
        net = maxpool_op(net, 20, 2, 2)
        net = conv_op(net, 21, 512, 1, 1)
        net = conv_op(net, 22, 1024, 3, 1)
        net = conv_op(net, 23, 512, 1, 1)
        net = conv_op(net, 24, 1024, 3, 1)
        net = conv_op(net, 25, 1024, 3, 1)
        net = conv_op(net, 26, 1024, 3, 2)
        net = conv_op(net, 27, 1024, 3, 1)
        net = conv_op(net, 28, 1024, 3, 1)
        net = flatten_op(net, 29)
        net = fc_op(net, 30, 512, activation=leak_relu)
        net = fc_op(net, 31, 4096, activation=leak_relu)
        net = fc_op(net, 32, S * S * (C + 5 * B))
        return net


class YoloV1Tiny(object):
    def __init__(self):
        self.images = tf.placeholder(tf.float32, [None, 448, 448, 3])

        self.threshold = 0.15  # confidence scores threshold
        self.iou_threshold = 0.4  # iou threshold in non max suppression
        self.max_output_size = 10  # the maximum number of boxes to be selected by non max suppression

        self.prediction = self.inference_op()
        self.scores, self.boxes, self.box_classes = convert_result(threshold=self.threshold,
                                                                   iou_threshold=self.iou_threshold,
                                                                   max_output_size=self.max_output_size,
                                                                   prediction=self.prediction)

    def inference_op(self):
        net = conv_op(self.images, 1, 16, 3, 1)
        net = maxpool_op(net, 2, 2, 2)
        net = conv_op(net, 3, 32, 3, 1)
        net = maxpool_op(net, 4, 2, 2)
        net = conv_op(net, 5, 64, 3, 1)
        net = maxpool_op(net, 6, 2, 2)
        net = conv_op(net, 7, 128, 3, 1)
        net = maxpool_op(net, 8, 2, 2)
        net = conv_op(net, 9, 256, 3, 1)
        net = maxpool_op(net, 10, 2, 2)
        net = conv_op(net, 11, 512, 3, 1)
        net = maxpool_op(net, 12, 2, 2)
        net = conv_op(net, 13, 1024, 3, 1)
        net = conv_op(net, 14, 1024, 3, 1)
        net = conv_op(net, 15, 1024, 3, 1)
        net = flatten_op(net, 16)
        net = fc_op(net, 17, 256, activation=leak_relu)
        net = fc_op(net, 18, 4096, activation=leak_relu)
        net = fc_op(net, 19, S * S * (C + 5 * B))
        return net
