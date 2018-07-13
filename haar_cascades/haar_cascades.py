import argparse
import datetime
import imutils
import cv2
import sys
from random import randint
import numpy as np

from utils.general import intersection_over_union, frame_size

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
	help="path to the input video")
ap.add_argument("-w", "--win-stride", type=str, default="(8, 8)",
	help="window stride")
ap.add_argument("-p", "--padding", type=str, default="(16, 16)",
	help="object padding")
ap.add_argument("-s", "--scale", type=float, default=1.05,
	help="image pyramid scale")
ap.add_argument("-m", "--mean-shift", type=int, default=-1,
	help="whether or not mean shift grouping should be used")
args = vars(ap.parse_args())

# evaluate the command line arguments (using the eval function like
# this is not good form, but let's tolerate it for the example)
winStride = eval(args["win_stride"])
padding = eval(args["padding"])
meanShift = True if args["mean_shift"] > 0 else False

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cap = cv2.VideoCapture(args['video'])

frame_counter = 0
while True:
    ret, frame = cap.read()

    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame = imutils.resize(frame, width=min(1000, frame.shape[1]))

    # detect people in the image
    (rects, weights) = hog.detectMultiScale(frame, winStride=winStride,
    	padding=padding, scale=args["scale"], useMeanshiftGrouping=meanShift)

    detections = []
    for i in range(len(rects)):
        # coordinates, weights, color
        detections.append([rects[i], (randint(0, 256), randint(0, 256), randint(0, 256))])

    num_detections = len(detections)

    if frame_counter != 0:
        cost_matrix = [[] for a in detections]
        for i in range(len(detections)):
            box_1 = detections[i][0]
            r1 = (box_1[0], box_1[1], box_1[0] + box_1[2], box_1[1] + box_1[3])
            for j in range(len(previous_detections)):
                box_2 = previous_detections[j][0]
                r2 = (box_2[0], box_2[1], box_2[0] + box_1[2], box_2[1] + box_1[3])

                cost_matrix[i].append(intersection_over_union(r1, r2))

        cost = np.array(cost_matrix)

        if detections and previous_detections:

            max_indices = np.argmax(cost, axis=1)

            for index in range(len(max_indices)):
                if index <= num_previous_detections:

                    if previous_detections[max_indices[index]][1] and cost[index][max_indices[index]] >= 0.5:
                        detections[index][1] = previous_detections[max_indices[index]][1]
                    # else:
                    #     predictions[index]['color'] = (randint(0, 255), randint(0, 255), randint(0, 255))
                    #     predictions[index]['index'] = randint(12, 100)

    # draw the original bounding boxes
    for o in detections:

        box = o[0]

        x1 = box[0]
        y1 = box[1]

        x2 = box[0] + box[2]
        y2 = box[1] + box[3]

        color = o[1]
        class_name = 'person'

        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        (test_width, text_height), baseline = cv2.getTextSize(
            class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 1)

        cv2.rectangle(frame, (x1, y1),
                      (x1+test_width, y1-text_height-baseline),
                      color, thickness=cv2.FILLED)

        cv2.putText(frame, class_name, (x1, y1-baseline),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)


    cv2.imshow('frame', frame)
    previous_detections = detections
    num_previous_detections = len(detections)
    frame_counter += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
