import time
import sys
import logging.config
import cv2
import tensorflow as tf
import numpy as np
from random import randint

import argparse

from models import yolo
from log_config import LOGGING
from utils.general import format_predictions, find_class_by_name, is_url, intersection_over_union, frame_size

logging.config.dictConfig(LOGGING)

logger = logging.getLogger('detector')
FLAGS = tf.flags.FLAGS


def image(_):
    absolute_start = time.time()
    img = cv2.imread(FLAGS.image, 1)

    source_h, source_w = img.shape[:2]

    yolo_start = time.time()
    model_cls = find_class_by_name(FLAGS.model_name, [yolo])
    model = model_cls(input_shape=(source_h, source_w, 3))
    model.init()
    yolo_end = time.time()

    pred_start = time.time()
    predictions = model.evaluate(img)
    pred_end = time.time()

    postprocess_start = time.time()
    p = []
    for i in predictions:
        if i['class_name'] == 'person':
            if i['score'] > float(FLAGS.threshold):
                p.append(i)

    predictions = p
    num_persons = len(predictions)

    draw_start = time.time()
    for o in predictions:
        x1 = o['box']['left']
        x2 = o['box']['right']

        y1 = o['box']['top']
        y2 = o['box']['bottom']

        color = o['color']
        class_name = o['class_name']

        # Draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Draw label
        (test_width, text_height), baseline = cv2.getTextSize(
            class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 1)
        cv2.rectangle(img, (x1, y1),
                      (x1+test_width, y1-text_height-baseline),
                      color, thickness=cv2.FILLED)
        cv2.putText(img, class_name, (x1, y1-baseline),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)

    draw_end = time.time()
    postprocess_end = time.time()
    absolute_end = time.time()

    absolute_execution_time = absolute_end - absolute_start
    print('Absolute execution time:', round(absolute_execution_time, 4))
    print('Model creation time:', round(yolo_end - yolo_start, 4))
    print('Prediction time:', round(pred_end - pred_start, 4))
    print('Postprocessing time:', round(postprocess_end - postprocess_start, 4))
    print('Boundary boxes drawing time:', round(draw_end - draw_start, 4))

    win_name = 'Image'
    cv2.imshow(win_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    sys.exit()


def video(_):
    win_name = 'Detector'
    cv2.namedWindow(win_name)

    video = FLAGS.video

    # Define the codec and create VideoWriter object
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter('videos/YOLO_output.avi',fourcc, 30.0, (640,480))
    # fourcc = cv2.VideoWriter_fourcc(*'0X00000021')
    # out = cv2.VideoWriter('videos/YOLO_output.mp4',fourcc, 30.0, (640,480))
    # out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (640,480))

    cam = cv2.VideoCapture(video)
    if not cam.isOpened():
        raise IOError('Can\'t open "{}"'.format(FLAGS.video))

    frame_width = int(cam.get(3))
    frame_height = int(cam.get(4))

    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    out = cv2.VideoWriter('videos/YOLO_output.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

    source_h = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
    source_w = cam.get(cv2.CAP_PROP_FRAME_WIDTH)

    model_cls = find_class_by_name(FLAGS.model_name, [yolo])
    model = model_cls(input_shape=(source_h, source_w, 3))
    model.init()

    frame_num = 0
    start_time = time.time()
    fps = 0

    try:
        while True:
            # ret, frame = cam.read()

            for i in range(int(FLAGS.skip_n_frames)):
                ret, frame = cam.read()

            if not ret:
                logger.info('Can\'t read video data. Potential end of stream')
                return

            predictions = model.evaluate(frame)

            p = []
            for i in predictions:
                if i['class_name'] == 'person':
                    box_1 = i['box']
                    r1 = (box_1['top'], box_1['left'], box_1['bottom'], box_1['right'])
                    if frame_size(r1) <= 100000:
                        if i['score'] > float(FLAGS.threshold):
                            p.append(i)

            predictions = p
            num_persons = len(predictions)

            # trajectory computation
            if frame_num != 0:
                cost_matrix = [[] for a in predictions]
                for i in range(len(predictions)):
                    box_1 = predictions[i]['box']
                    r1 = (box_1['top'], box_1['left'], box_1['bottom'], box_1['right'])
                    for j in range(len(previous_frame_pred)):
                        box_2 = previous_frame_pred[j]['box']
                        r2 = (box_2['top'], box_2['left'], box_2['bottom'], box_2['right'])

                        cost_matrix[i].append(intersection_over_union(r1, r2))

                cost = np.array(cost_matrix)
                # print(np.max(cost, axis=1))
                max_indices = np.argmax(cost, axis=1)


                for index in range(len(max_indices)):
                    if index <= num_previous_pred:
                        # if previous_frame_pred[max_indices[index]]['index'] and cost[index][max_indices[index]] >= 0.5:
                        #     predictions[index]['index'] = previous_frame_pred[max_indices[index]]['index']
                        if previous_frame_pred[max_indices[index]]['color'] and cost[index][max_indices[index]] >= 0.5:
                        # if previous_frame_pred[max_indices[index]]['color']:
                            predictions[index]['color'] = previous_frame_pred[max_indices[index]]['color']
                            predictions[index]['index'] = previous_frame_pred[max_indices[index]]['index']
                        else:
                            predictions[index]['color'] = (randint(0, 255), randint(0, 255), randint(0, 255))
                            predictions[index]['index'] = randint(12, 100)

            previous_frame_pred = predictions
            num_previous_pred = len(predictions)

            for o in predictions:
                x1 = o['box']['left']
                x2 = o['box']['right']

                y1 = o['box']['top']
                y2 = o['box']['bottom']

                color = o['color']
                class_name = o['class_name'] + str(o['index'])
                # index = str(o['index'])

                # Draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Draw label
                (test_width, text_height), baseline = cv2.getTextSize(
                    class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 1)
                cv2.rectangle(frame, (x1, y1),
                              (x1+test_width, y1-text_height-baseline),
                              color, thickness=cv2.FILLED)
                cv2.putText(frame, class_name, (x1, y1-baseline),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)

            end_time = time.time()
            # fps = fps * 0.9 + 1/(end_time - start_time) * 0.1
            fps = fps * 0.9 + 1/(end_time - start_time) * 0.1
            start_time = end_time

            if predictions:
                logger.info('Predictions: {}'.format(
                    format_predictions(predictions)))


            # Draw additional info
            frame_info = 'Frame: {0}, FPS: {1:.2f}, Persons: {2}'.format(frame_num, fps, num_persons)
            cv2.putText(frame, frame_info, (10, frame.shape[0]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            logger.info(frame_info)

            out.write(frame)
            cv2.imshow(win_name, frame)

            if FLAGS.save:
                image_name = 'videos/YOLOFrames/frame_' + str(frame_num) + '.png'
                cv2.imwrite(image_name, frame)

            # if predictions:
            #     logger.info('Predictions: {}'.format(
            #         format_predictions(predictions)))

            key = cv2.waitKey(1) & 0xFF

            # Exit
            if key == ord('q'):
                break

            # Take screenshot
            if key == ord('s'):
                cv2.imwrite('frame_{}.jpg'.format(time.time()), frame)

            frame_num += 1

    finally:
        if FLAGS.image:
            cv2.imshow('image', frame)
        else:
            cv2.destroyAllWindows()
            cam.release()
            out.release()
            model.close()


if __name__ == '__main__':

    tf.flags.DEFINE_boolean('save', False, 'Save frames')
    tf.flags.DEFINE_string('skip_n_frames', 5, 'Number of frames to skip for video to process faster.')
    tf.flags.DEFINE_string('threshold', 0.3, 'Threshold for recognized objects confidence.')

    tf.flags.DEFINE_string('image', None, 'Path to image file.')
    tf.flags.DEFINE_string('video', 0, 'Path to the video file.')
    tf.flags.DEFINE_string('model_name', 'Yolo2Model', 'Model name to use.')

    if FLAGS.image:
        tf.app.run(main=image)
    else:
        tf.app.run(main=video)
