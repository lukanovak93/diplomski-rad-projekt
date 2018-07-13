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
ap.add_argument("--save", type=bool, default=False,
	help="whether or not to save every frame")
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

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv2.VideoWriter('videos/ViolaJones_output.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

frame_counter = 0
try:
	while True:
		ret, frame = cap.read()

		# frame = imutils.resize(frame, width=min(1000, frame.shape[1]))

		# detect people in the image
		(rects, weights) = hog.detectMultiScale(frame, winStride=winStride, padding=padding, scale=args["scale"], useMeanshiftGrouping=meanShift)

		detections = []
		for i in range(len(rects)):
			# coordinates, weights, color
			detections.append({
			'box': {
				'top': rects[i][0],
				'left': rects[i][1],
				'bottom': rects[i][0] + rects[i][2],
				'right': rects[i][1] + rects[i][3]
	        },
			'index': randint(0, 100),
			'color': (randint(0, 256), randint(0, 256), randint(0, 256))
			})

		num_detections = len(detections)

		if frame_counter != 0:
			cost_matrix = [[] for a in detections]
			for i in range(len(detections)):
				box_1 = detections[i]['box']
				r1 = (box_1['top'], box_1['left'], box_1['bottom'], box_1['right'])
				for j in range(len(previous_detections)):
					box_2 = previous_detections[j]['box']
					r2 = (box_2['top'], box_2['left'], box_2['bottom'], box_2['right'])

					cost_matrix[i].append(intersection_over_union(r1, r2))

			cost = np.array(cost_matrix)

			if detections and previous_detections:

				max_indices = np.argmax(cost, axis=1)

				for index in range(len(max_indices)):
					if index <= num_previous_detections:

						if previous_detections[max_indices[index]]['color'] and cost[index][max_indices[index]] >= 0.5:

							detections[index]['color'] = previous_detections[max_indices[index]]['color']
							detections[index]['index'] = previous_detections[max_indices[index]]['index']

		# draw the original bounding boxes
		for o in detections:
			x1 = o['box']['top']
			x2 = o['box']['bottom']

			y1 = o['box']['left']
			y2 = o['box']['right']

			color = o['color']
			class_name = 'person' + str(o['index'])

			# Draw box
			cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

			(test_width, text_height), baseline = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 1)

			cv2.rectangle(frame, (x1, y1),(x1+test_width, y1-text_height-baseline), color, thickness=cv2.FILLED)

			cv2.putText(frame, class_name, (x1, y1-baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)

		print('Frame: {0}, Persons: {1}'.format(frame_counter, len(detections)))

		out.write(frame)
		cv2.imshow('frame', frame)

		if args['save']:
			img_name = 'videos/ViolaJonesFrames/frame_' + str(frame_counter) + '.png'
			cv2.imwrite(img_name, frame)

		previous_detections = detections
		num_previous_detections = len(detections)
		frame_counter += 1
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

finally:
	out.release()
