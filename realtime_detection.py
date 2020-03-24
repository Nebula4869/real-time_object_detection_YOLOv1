from yolo_v1 import *
import tensorflow as tf
import cv2


CLASSES = ["aeroplane", "bicycle", "bird", "boat", "bottle",
           "bus", "car", "cat", "chair", "cow",
           "diningtable", "dog", "horse", "motorbike", "person",
           "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


def detect_from_image(image_path, model, model_file):
    img = cv2.imread(image_path)
    img_h, img_w, _ = img.shape
    img_resized = cv2.resize(img, (448, 448))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_in = (img_rgb / 255.0 * 2.0 - 1.0).reshape((1, 448, 448, 3))

    tf.reset_default_graph()
    network = model()
    with tf.Session() as sess:
        tf.train.Saver().restore(sess, model_file)
        scores, boxes, box_classes = sess.run([network.scores, network.boxes, network.box_classes],
                                              feed_dict={network.images: img_in})

        for i in range(len(scores)):
            left = int((boxes[i, 0] - boxes[i, 2] / 2) * img_w)
            right = int((boxes[i, 0] + boxes[i, 2] / 2) * img_w)
            top = int((boxes[i, 1] - boxes[i, 3] / 2) * img_h)
            bottom = int((boxes[i, 1] + boxes[i, 3] / 2) * img_h)

            box_class = CLASSES[box_classes[i]]
            score = scores[i]

            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(img, (left, top - 20), (right, top), (125, 125, 125), -1)
            cv2.putText(img, box_class + ' : %.2f' % score, (left + 5, top - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        cv2.imshow('YOLOv1 result', img)
        cv2.waitKey()


def detect_from_video(video_path, model, model_file):
    tf.reset_default_graph()
    network = model()
    with tf.Session() as sess:
        tf.train.Saver().restore(sess, model_file)

        cap = cv2.VideoCapture(video_path)
        timer = 0
        scores, boxes, box_classes, img_h, img_w = None, None, None, None, None
        while True:
            timer += 1
            _, frame = cap.read()
            if timer % 5 == 0:
                img_h, img_w, _ = frame.shape
                img_resized = cv2.resize(frame, (448, 448))
                img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                img_in = (img_rgb / 255.0 * 2.0 - 1.0).reshape((1, 448, 448, 3))
                scores, boxes, box_classes = sess.run([network.scores, network.boxes, network.box_classes],
                                                      feed_dict={network.images: img_in})

            if scores is not None:
                for i in range(len(scores)):

                    box_class = CLASSES[box_classes[i]]

                    left = int((boxes[i, 0] - boxes[i, 2] / 2) * img_w)
                    right = int((boxes[i, 0] + boxes[i, 2] / 2) * img_w)
                    top = int((boxes[i, 1] - boxes[i, 3] / 2) * img_h)
                    bottom = int((boxes[i, 1] + boxes[i, 3] / 2) * img_h)

                    score = scores[i]

                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.rectangle(frame, (left, top - 20), (right, top), (125, 125, 125), -1)
                    cv2.putText(frame, box_class + ' : %.2f' % score, (left + 5, top - 7),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            cv2.imshow('YOLOv1 result', frame)
            cv2.waitKey(1)


if __name__ == "__main__":
    detect_from_image("test.jpg", model=YoloV1, model_file='./models/yolo_v1.ckpt')
    detect_from_video(0, model=YoloV1, model_file='./models/yolo_v1.ckpt')
