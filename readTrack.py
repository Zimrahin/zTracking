import cv2 as cv
import csv
import math
import numpy as np
import scipy.signal
from matplotlib import pyplot as plt
import argparse


agentLength = 10.5 	# cm (cube)

def readTrack(opt):

	# read CSV file
	positions = []		# for perspective transform
	with open(opt.cornerPoints, "r") as infile:
		reader = csv.reader(infile)
		for row in reader:
			row_float =[float(row[0]), float(row[1])]
			positions.append(row_float)

	# Read video
	vid = cv.VideoCapture(opt.fileName)
	if not vid.isOpened():
		exit(-1)
	fps = vid.get(cv.CAP_PROP_FPS)
	deltaT = 1./fps

	ret, frame = vid.read()
	height, width = frame.shape[:2]		# height and width in pixels

	# write CSV file
	myFile = open(opt.outputName, 'w', newline = '')
	writer = csv.writer(myFile)
	writer.writerow(['time(s)', 'distance12(cm)', 'distance23(cm)', 'distance34(cm)'])

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

	# read CSV file
	trackPoints = []
	with open(opt.trackName, "r") as infile:
		reader = csv.reader(infile)
		for row in reader:
			row_float =[float(row[0]), float(row[1])]
			trackPoints.append(row_float)
	#print(trackPoints)

	trackPoints_x = [x[0] for x in trackPoints]
	trackPoints_y = [y[1] for y in trackPoints]
	trackPoints_x_cm = [x[0] * agentLength / length for x in trackPoints]
	trackPoints_y_cm = [y[1] * agentLength / length for y in trackPoints]


	plt.figure(0)
	plt.cla()
	plt.gca().set_aspect('equal')
	plt.gca().set_ylim(0, height * agentLength / length)
	plt.gca().set_xlim(0, width * agentLength / length)
	plt.gca().imshow(cv.cvtColor(frame, cv.COLOR_BGR2RGB), extent=[0, width * agentLength / length, 0, height * agentLength / length])
	plt.plot(trackPoints_x_cm, trackPoints_y_cm, color='lime')
	plt.xlabel('(cm)')
	plt.ylabel('(cm)')
	plt.show()



	# initialise tracker1
	bbox1 = cv.selectROI('Tracking_v3', frame, False)
	tracker1 = cv.TrackerCSRT_create()
	tracker1.init(frame, bbox1)

	# initialise tracker2
	bbox2 = cv.selectROI('Tracking_v3', frame, False)
	tracker2 = cv.TrackerCSRT_create()
	tracker2.init(frame, bbox2)

	# initialise tracker3
	bbox3 = cv.selectROI('Tracking_v3', frame, False)
	tracker3 = cv.TrackerCSRT_create()
	tracker3.init(frame, bbox3)

	# initialise tracker4
	bbox4 = cv.selectROI('Tracking_v3', frame, False)
	tracker4 = cv.TrackerCSRT_create()
	tracker4.init(frame, bbox4)

	# initialise tracker5
	bbox5 = cv.selectROI('Tracking_v3', frame, False)
	tracker5 = cv.TrackerCSRT_create()
	tracker5.init(frame, bbox5)

	cv.destroyWindow('Tracking_v3')

	firstFrame = True
	seconds = 0.
	timeArr = []
	distance12Arr = []
	distance23Arr = []
	distance34Arr = []
	distance45Arr = []

	while vid.isOpened():
		ret, frame = vid.read()

		if ret:
			frame = cv.warpPerspective(frame, M, (width, height))
			ret, bbox2 = tracker2.update(frame)
			_, bbox3 = tracker3.update(frame)
			_, bbox1 = tracker1.update(frame)
			_, bbox4 = tracker4.update(frame)
			_, bbox5 = tracker5.update(frame)

			if ret:
				x1, y1, w1, h1 = int(bbox1[0]), int(bbox1[1]), int(bbox1[2]), int(bbox1[3])
				#cv.rectangle(frame, (x1, y1), (x1+w1, y1+h1), (0, 0, 255), 2, 1)
				x2, y2, w2, h2 = int(bbox2[0]), int(bbox2[1]), int(bbox2[2]), int(bbox2[3])
				#cv.rectangle(frame, (x2, y2), (x2 + w2, y2 + h2), (0, 0, 255), 2, 1)
				x3, y3, w3, h3 = int(bbox3[0]), int(bbox3[1]), int(bbox3[2]), int(bbox3[3])
				#cv.rectangle(frame, (x3, y3), (x3 + w3, y3 + h3), (0, 0, 255), 2, 1)
				x4, y4, w4, h4 = int(bbox4[0]), int(bbox4[1]), int(bbox4[2]), int(bbox4[3])
				x5, y5, w5, h5 = int(bbox5[0]), int(bbox5[1]), int(bbox5[2]), int(bbox5[3])
			else:
				print('could not detect object')
				exit(-1)

			p1 = (x1 + w1/2., height - y1 - h1/2.)
			p2 = (x2 + w2/2., height - y2 - h2/2.)
			p3 = (x3 + w3/2., height - y3 - h3/2.)
			p4 = (x4 + w4/2., height - y4 - h4/2.)
			p5 = (x5 + w5/2., height - y5 - h5/2.)
			# Find closest point in track
			points = [p1, p2, p3, p4, p5]
			pointsInTrack = []
			pointsInTrackPositions = []
			for point in points:
				minDistance = 10000.
				count = 0
				for trackPoint in trackPoints:
					euclideanDistance2 = (point[0] - trackPoint[0]) ** 2 + (point[1] - trackPoint[1]) ** 2
					if euclideanDistance2 < minDistance:
						minDistance = euclideanDistance2
						minPoint = trackPoint
						positionInTrack = count
					count += 1
				pointsInTrack.append(minPoint)
				pointsInTrackPositions.append(positionInTrack)
			#print(pointsInTrack)
			#print(pointsInTrackPositions)

			# Find non-euclidean distance
			# Distance 1-2
			distance12 = 0.
			if pointsInTrackPositions[0] <= pointsInTrackPositions[1]:
				for k in range(pointsInTrackPositions[0], pointsInTrackPositions[1] - 1):
					distance12 += math.sqrt((trackPoints[k][0] - trackPoints[k+1][0]) ** 2 + (trackPoints[k][1] - trackPoints[k+1][1]) ** 2)
			else:
				for k in range(0, pointsInTrackPositions[1]):
					distance12 += math.sqrt((trackPoints[k][0] - trackPoints[k+1][0]) ** 2 + (trackPoints[k][1] - trackPoints[k+1][1]) ** 2)
				for k in range(pointsInTrackPositions[0], len(trackPoints) - 1):
					distance12 += math.sqrt((trackPoints[k][0] - trackPoints[k+1][0]) ** 2 + (trackPoints[k][1] - trackPoints[k+1][1]) ** 2)
				# Add discontinuity
				distance12 += math.sqrt((trackPoints[0][0] - trackPoints[len(trackPoints) - 1][0]) ** 2 + (trackPoints[0][1] - trackPoints[len(trackPoints) - 1][1]) ** 2)

			# Distance 2-3
			distance23 = 0.
			if pointsInTrackPositions[1] <= pointsInTrackPositions[2]:
				for k in range(pointsInTrackPositions[1], pointsInTrackPositions[2] - 1):
					distance23 += math.sqrt((trackPoints[k][0] - trackPoints[k+1][0]) ** 2 + (trackPoints[k][1] - trackPoints[k+1][1]) ** 2)
			else:
				for k in range(0, pointsInTrackPositions[2]):
					distance23 += math.sqrt((trackPoints[k][0] - trackPoints[k+1][0]) ** 2 + (trackPoints[k][1] - trackPoints[k+1][1]) ** 2)
				for k in range(pointsInTrackPositions[1], len(trackPoints) - 1):
					distance23 += math.sqrt((trackPoints[k][0] - trackPoints[k+1][0]) ** 2 + (trackPoints[k][1] - trackPoints[k+1][1]) ** 2)
				# Add discontinuity
				distance23 += math.sqrt((trackPoints[0][0] - trackPoints[len(trackPoints) - 1][0]) ** 2 + (trackPoints[0][1] - trackPoints[len(trackPoints) - 1][1]) ** 2)

			# Distance 3-4
			distance34 = 0.
			if pointsInTrackPositions[2] <= pointsInTrackPositions[3]:
				for k in range(pointsInTrackPositions[2], pointsInTrackPositions[3] - 1):
					distance34 += math.sqrt((trackPoints[k][0] - trackPoints[k+1][0]) ** 2 + (trackPoints[k][1] - trackPoints[k+1][1]) ** 2)
			else:
				for k in range(0, pointsInTrackPositions[3]):
					distance34 += math.sqrt((trackPoints[k][0] - trackPoints[k+1][0]) ** 2 + (trackPoints[k][1] - trackPoints[k+1][1]) ** 2)
				for k in range(pointsInTrackPositions[2], len(trackPoints) - 1):
					distance34 += math.sqrt((trackPoints[k][0] - trackPoints[k+1][0]) ** 2 + (trackPoints[k][1] - trackPoints[k+1][1]) ** 2)
				# Add discontinuity
				distance34 += math.sqrt((trackPoints[0][0] - trackPoints[len(trackPoints) - 1][0]) ** 2 + (trackPoints[0][1] - trackPoints[len(trackPoints) - 1][1]) ** 2)

			# Distance 4-5
			distance45 = 0.
			if pointsInTrackPositions[3] <= pointsInTrackPositions[4]:
				for k in range(pointsInTrackPositions[3], pointsInTrackPositions[4] - 1):
					distance45 += math.sqrt((trackPoints[k][0] - trackPoints[k+1][0]) ** 2 + (trackPoints[k][1] - trackPoints[k+1][1]) ** 2)
			else:
				for k in range(0, pointsInTrackPositions[4]):
					distance45 += math.sqrt((trackPoints[k][0] - trackPoints[k+1][0]) ** 2 + (trackPoints[k][1] - trackPoints[k+1][1]) ** 2)
				for k in range(pointsInTrackPositions[3], len(trackPoints) - 1):
					distance45 += math.sqrt((trackPoints[k][0] - trackPoints[k+1][0]) ** 2 + (trackPoints[k][1] - trackPoints[k+1][1]) ** 2)
				# Add discontinuity
				distance45 += math.sqrt((trackPoints[0][0] - trackPoints[len(trackPoints) - 1][0]) ** 2 + (trackPoints[0][1] - trackPoints[len(trackPoints) - 1][1]) ** 2)

			distance12_cm = distance12 * agentLength / length
			distance23_cm = distance23 * agentLength / length
			distance34_cm = distance34 * agentLength / length
			distance45_cm = distance45 * agentLength / length

			# Plot animation
			plt.figure(1)
			plt.cla()
			plt.gca().set_aspect('equal')
			plt.gca().set_ylim(0, height * agentLength / length)
			plt.gca().set_xlim(0, width * agentLength / length)
			plt.gca().imshow(cv.cvtColor(frame, cv.COLOR_BGR2RGB), extent=[0, width * agentLength / length, 0, height * agentLength / length])
			# Distance 1-2
			if pointsInTrackPositions[0] <= pointsInTrackPositions[1]:
				plt.plot(trackPoints_x_cm[pointsInTrackPositions[0]:pointsInTrackPositions[1]], trackPoints_y_cm[pointsInTrackPositions[0]:pointsInTrackPositions[1]], color='lime')
			else:
				plt.plot(trackPoints_x_cm[0:pointsInTrackPositions[1]], trackPoints_y_cm[0:pointsInTrackPositions[1]], color='lime')
				plt.plot(trackPoints_x_cm[pointsInTrackPositions[0]:-1], trackPoints_y_cm[pointsInTrackPositions[0]:-1], color='lime')
				plt.plot([trackPoints_x_cm[-1], trackPoints_x_cm[0]], [trackPoints_y_cm[-1], trackPoints_y_cm[0]], color='lime')
			# Distance 2-3
			if pointsInTrackPositions[1] <= pointsInTrackPositions[2]:
				plt.plot(trackPoints_x_cm[pointsInTrackPositions[1]:pointsInTrackPositions[2]], trackPoints_y_cm[pointsInTrackPositions[1]:pointsInTrackPositions[2]], color='magenta')
			else:
				plt.plot(trackPoints_x_cm[0:pointsInTrackPositions[2]], trackPoints_y_cm[0:pointsInTrackPositions[2]], color='magenta')
				plt.plot(trackPoints_x_cm[pointsInTrackPositions[1]:-1], trackPoints_y_cm[pointsInTrackPositions[1]:-1], color='magenta')
				plt.plot([trackPoints_x_cm[-1], trackPoints_x_cm[0]], [trackPoints_y_cm[-1], trackPoints_y_cm[0]], color='magenta')
			# Distance 3-4
			if pointsInTrackPositions[2] <= pointsInTrackPositions[3]:
				plt.plot(trackPoints_x_cm[pointsInTrackPositions[2]:pointsInTrackPositions[3]], trackPoints_y_cm[pointsInTrackPositions[2]:pointsInTrackPositions[3]], color='yellow')
			else:
				plt.plot(trackPoints_x_cm[0:pointsInTrackPositions[3]], trackPoints_y_cm[0:pointsInTrackPositions[3]], color='yellow')
				plt.plot(trackPoints_x_cm[pointsInTrackPositions[2]:-1], trackPoints_y_cm[pointsInTrackPositions[2]:-1], color='yellow')
				plt.plot([trackPoints_x_cm[-1], trackPoints_x_cm[0]], [trackPoints_y_cm[-1], trackPoints_y_cm[0]], color='yellow')
			# Distance 4-5
			if pointsInTrackPositions[3] <= pointsInTrackPositions[4]:
				plt.plot(trackPoints_x_cm[pointsInTrackPositions[3]:pointsInTrackPositions[4]], trackPoints_y_cm[pointsInTrackPositions[3]:pointsInTrackPositions[4]], color='cyan')
			else:
				plt.plot(trackPoints_x_cm[0:pointsInTrackPositions[4]], trackPoints_y_cm[0:pointsInTrackPositions[4]], color='cyan')
				plt.plot(trackPoints_x_cm[pointsInTrackPositions[3]:-1], trackPoints_y_cm[pointsInTrackPositions[3]:-1], color='cyan')
				plt.plot([trackPoints_x_cm[-1], trackPoints_x_cm[0]], [trackPoints_y_cm[-1], trackPoints_y_cm[0]], color='cyan')

			plt.plot(pointsInTrack[0][0] * agentLength / length, pointsInTrack[0][1] * agentLength / length, 'bo', color='red')
			plt.plot(pointsInTrack[1][0] * agentLength / length, pointsInTrack[1][1] * agentLength / length, 'bo',  color='red')
			plt.plot(pointsInTrack[2][0] * agentLength / length, pointsInTrack[2][1] * agentLength / length, 'bo',  color='red')
			plt.plot(pointsInTrack[3][0] * agentLength / length, pointsInTrack[3][1] * agentLength / length, 'bo',  color='red')
			plt.plot(pointsInTrack[4][0] * agentLength / length, pointsInTrack[4][1] * agentLength / length, 'bo',  color='red')

			#plt.plot([p1[0] * agentLength / length, p2[0] * agentLength / length], [p1[1] * agentLength / length, p2[1] * agentLength / length], 'bo', linestyle="--", color='lime')
			plt.text((p1[0] + p2[0])/2. * agentLength / length, (p1[1] + p2[1])/2. * agentLength / length, str(round(distance12_cm, 1)), color='white')

			#plt.plot([p2[0] * agentLength / length, p3[0] * agentLength / length], [p2[1] * agentLength / length, p3[1] * agentLength / length], 'bo', linestyle="--", color='lime')
			plt.text((p2[0] + p3[0]) / 2. * agentLength / length, (p2[1] + p3[1]) / 2. * agentLength / length, str(round(distance23_cm, 1)), color='white')

			plt.text((p3[0] + p4[0]) / 2. * agentLength / length, (p3[1] + p4[1]) / 2. * agentLength / length, str(round(distance34_cm, 1)), color='white')

			plt.text((p4[0] + p5[0]) / 2. * agentLength / length, (p4[1] + p5[1]) / 2. * agentLength / length, str(round(distance45_cm, 1)), color='white')

			plt.xlabel('(cm)')
			plt.ylabel('(cm)')
			plt.pause(0.000000001)

			# append into arrays
			timeArr = np.append(timeArr, seconds)
			distance12Arr = np.append(distance12Arr, distance12_cm)
			distance23Arr = np.append(distance23Arr, distance23_cm)
			distance34Arr = np.append(distance34Arr, distance34_cm)
			distance45Arr = np.append(distance45Arr, distance45_cm)

			# write CSV file
			writer.writerow([seconds, distance12_cm, distance23_cm, distance34_cm, distance45_cm])

			seconds += deltaT
			firstFrame = False

			if cv.waitKey(1) == 27:
				break
		else:
			break
	vid.release()
	plt.show()


	# Butterworth filter
	b, a = scipy.signal.butter(1, 0.5)
	distance12Arr = scipy.signal.filtfilt(b, a, distance12Arr)
	distance23Arr = scipy.signal.filtfilt(b, a, distance23Arr)
	distance34Arr = scipy.signal.filtfilt(b, a, distance34Arr)
	distance45Arr = scipy.signal.filtfilt(b, a, distance45Arr)

	# Plot
	plt.figure(2)
	plt.plot(timeArr, distance12Arr, label='distance12')
	plt.plot(timeArr, distance23Arr, label='distance23')
	plt.plot(timeArr, distance34Arr, label='distance34')
	plt.plot(timeArr, distance45Arr, label='distance45')
	plt.xlabel('Time(s)')
	plt.ylabel('Distance(cm)')
	plt.title('Distance vs Time')
	plt.grid(True)
	plt.legend()
	plt.show()

def parse_opt(known=False):
	parser = argparse.ArgumentParser()
	parser.add_argument('--fileName', default = "prueba_1.mp4",type=str,help='Video containing the experiment')
	parser.add_argument('--trackName', default = "trackPoints.csv",type=str,help='CSV file containing track coordinates')
	parser.add_argument('--outputName', default = "results.csv",type=str,help='Results for plotting')
	parser.add_argument('--cornerPoints', default = "cornerPoints.csv",type=str,help='Track corners used for perspective transform')
	opt = parser.parse_known_args()[0] if known else parser.parse_args()
	return opt

def main(opt):
	readTrack(opt)

if __name__ == '__main__':
	opt = parse_opt()
	main(opt)
