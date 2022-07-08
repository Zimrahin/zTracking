import cv2 as cv
import csv
import math
import numpy as np
import scipy.signal
from matplotlib import pyplot as plt
import argparse

agentLength = 5 	# cm (cube)
positions = []		# for perspective transform
countClicks = 0

def writeTrack(opt):

	def getMousePositions(event, x, y, flags, param):
		global positions, countClicks
		if event == cv.EVENT_LBUTTONDOWN:
			positions.append([x, y])
			cv.circle(frame, (x, y), 2, (0, 255, 0), -1)
			# print(positions)
			countClicks += 1

	# Read video
	vid = cv.VideoCapture(opt.fileName)
	if not vid.isOpened():
		exit(-1)
	fps = vid.get(cv.CAP_PROP_FPS)
	deltaT = 1./fps

	ret, frame = vid.read()
	height, width = frame.shape[:2]		# height and width in pixels

	# Select points
	cv.namedWindow('Select points')
	cv.setMouseCallback('Select points', getMousePositions)

	while countClicks <= 3:
		cv.imshow('Select points', frame)
		if cv.waitKey(20) & 0xFF == 27:
			break
	cv.destroyAllWindows()

	# write CSV file
	myFile = open(opt.outputPoints, 'w', newline = '')
	writer = csv.writer(myFile)
	for row in positions:
		writer.writerow(row)
	myFile.close()

	# Perspective transform
	src = np.float32([positions[0], positions[1], positions[2], positions[3]])
	dst = np.float32([positions[0], [positions[0][0], positions[1][1]], [positions[2][0], positions[0][1]], [positions[2][0], positions[1][1]]])
	M = cv.getPerspectiveTransform(src, dst) 	# Transformation matrix
	warped_frame = cv.warpPerspective(frame, M, (width, height))
	frame = cv.warpPerspective(frame, M, (width, height))

	# get agent length in pixels
	bbox = cv.selectROI('Pixels to centimetres', warped_frame, False)
	length = bbox[2] 	# pixel
	cv.destroyWindow('Pixels to centimetres')

	# initialise tracker1
	bbox11 = cv.selectROI('Tracking_v3', frame, False)
	tracker11 = cv.TrackerCSRT_create()
	tracker11.init(frame, bbox11)

	cv.destroyWindow('Tracking_v3')

	firstFrame = True
	seconds = 0.
	timeArr = []

	trackPoints = []

	# write CSV file
	myFile = open(opt.outputName, 'w', newline = '')
	writer = csv.writer(myFile)


	while vid.isOpened():
		ret, frame = vid.read()

		if ret:
			frame = cv.warpPerspective(frame, M, (width, height))
			ret, bbox11 = tracker11.update(frame)

			if ret:
				x11, y11, w11, h11 = int(bbox11[0]), int(bbox11[1]), int(bbox11[2]), int(bbox11[3])
				cv.rectangle(frame, (x11, y11), (x11 + w11, y11 + h11), (0, 0, 255), 2, 1)

			else:
				print('could not detect object')
				exit(-1)

			p11 = (x11 + w11 / 2., height - y11 - h11 / 2.)

			# write CSV file
			writer.writerow(p11)

			# Plot animation
			plt.figure(1)
			plt.cla()
			plt.gca().set_aspect('equal')
			plt.gca().set_ylim(0, height * agentLength / length)
			plt.gca().set_xlim(0, width * agentLength / length)
			plt.gca().imshow(cv.cvtColor(frame, cv.COLOR_BGR2RGB), extent=[0, width * agentLength / length, 0, height * agentLength / length])
			plt.plot(p11[0] * agentLength / length, p11[1] * agentLength / length, 'bo', linestyle="--", color='lime')
			plt.xlabel('(cm)')
			plt.ylabel('(cm)')
			plt.pause(0.000000001)

			# append into arrays
			timeArr = np.append(timeArr, seconds)

			seconds += deltaT
			firstFrame = False

			if cv.waitKey(1) == 27:
				break
		else:
			break
	vid.release()
	myFile.close()
	plt.show()

def parse_opt(known=False):
	parser = argparse.ArgumentParser()
	parser.add_argument('--fileName', default = "crop.mp4",type=str,help='Cropped video containing one lap')
	parser.add_argument('--outputName', default = "trackPoints.csv",type=str,help='CSV file containing track coordinates')
	parser.add_argument('--outputPoints', default = "cornerPoints.csv",type=str,help='CSV file containing corner coordinates')
	opt = parser.parse_known_args()[0] if known else parser.parse_args()
	return opt

def main(opt):
	writeTrack(opt)

if __name__ == '__main__':
	opt = parse_opt()
	main(opt)
