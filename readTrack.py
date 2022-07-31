import cv2 as cv
import csv
import math
import numpy as np
import scipy.signal
from matplotlib import pyplot as plt
import argparse
from Tracker import Tracker


agentLength = 10.5 	# cm (cube)
N = 6 				# number of carts

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
	firstrow = ['time(s)']
	for i in range(N-1):
		firstrow.append('distance' + str(i+1) + str(i+2) + '(cm)')
	writer.writerow(firstrow)
	del(firstrow)	# free memory

	# Perspective transform
	src = np.float32([positions[0], positions[1], positions[2], positions[3]])
	dst = np.float32([positions[0], [positions[0][0], positions[1][1]], [positions[2][0], positions[0][1]], [positions[2][0], positions[1][1]]])
	M = cv.getPerspectiveTransform(src, dst) 	# Transformation matrix
	warped_frame = cv.warpPerspective(frame, M, (width, height))
	frame = cv.warpPerspective(frame, M, (width, height))

	# get agent length in pixels
	cv.namedWindow('Pixels to centimetres', cv.WINDOW_NORMAL)
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

	# initialise trackers
	cv.namedWindow('zTracking', cv.WINDOW_AUTOSIZE)
	trackers = []
	for i in range(N):
		trackers.append(Tracker(frame))

	cv.destroyWindow('zTracking')

	firstFrame = True
	seconds = 0.
	timeArr = []
	distancesArr = []
	colours_plot = ['lime', 'magenta', 'yellow', 'cyan', 'green', 'blue', 'red']

	while vid.isOpened():
		ret, frame = vid.read()

		if ret:
			points = []
			frame = cv.warpPerspective(frame, M, (width, height))
			for tracker in trackers:
				if(tracker.update(frame)):
					tracker.getCoordinates()
				else:
					print('could not detect object')
					exit(-1)
			
				tracker.getCentroid(height)
				points.append(tracker.p)

			# Find closest point in track
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


			# Find non-euclidean distance
			distances = []
			distances_cm = []
			temp_distance = 0.0
			for i in range(N - 1):
				if pointsInTrackPositions[i] <= pointsInTrackPositions[i+1]:
					for k in range(pointsInTrackPositions[i], pointsInTrackPositions[i+1] - 1):
						temp_distance += math.sqrt((trackPoints[k][0] - trackPoints[k+1][0]) ** 2 + (trackPoints[k][1] - trackPoints[k+1][1]) ** 2)
				else:
					for k in range(0, pointsInTrackPositions[i+1]):
						temp_distance += math.sqrt((trackPoints[k][0] - trackPoints[k+1][0]) ** 2 + (trackPoints[k][1] - trackPoints[k+1][1]) ** 2)
					for k in range(pointsInTrackPositions[i], len(trackPoints) - 1):
						temp_distance += math.sqrt((trackPoints[k][0] - trackPoints[k+1][0]) ** 2 + (trackPoints[k][1] - trackPoints[k+1][1]) ** 2)
					# Add discontinuity
					temp_distance += math.sqrt((trackPoints[0][0] - trackPoints[len(trackPoints) - 1][0]) ** 2 + (trackPoints[0][1] - trackPoints[len(trackPoints) - 1][1]) ** 2)

				distances.append(temp_distance)
				distances_cm.append(temp_distance * agentLength / length)
				# Clean temporal variable before next iteration
				temp_distance = 0.0

			
			# Plot animation
			plt.figure(1)
			plt.cla()
			plt.gca().set_aspect('equal')
			plt.gca().set_ylim(0, height * agentLength / length)
			plt.gca().set_xlim(0, width * agentLength / length)
			plt.gca().imshow(cv.cvtColor(frame, cv.COLOR_BGR2RGB), extent=[0, width * agentLength / length, 0, height * agentLength / length])
			
			# FOR distances
			for i in range(N-1):
				if pointsInTrackPositions[i] <= pointsInTrackPositions[i+1]:
					plt.plot(trackPoints_x_cm[pointsInTrackPositions[i]:pointsInTrackPositions[i+1]], trackPoints_y_cm[pointsInTrackPositions[i]:pointsInTrackPositions[i+1]], color=colours_plot[i])
				else:
					plt.plot(trackPoints_x_cm[0:pointsInTrackPositions[i+1]], trackPoints_y_cm[0:pointsInTrackPositions[i+1]], color=colours_plot[i])
					plt.plot(trackPoints_x_cm[pointsInTrackPositions[i]:-1], trackPoints_y_cm[pointsInTrackPositions[i]:-1], color=colours_plot[i])
					# Add discontinuity
					plt.plot([trackPoints_x_cm[-1], trackPoints_x_cm[0]], [trackPoints_y_cm[-1], trackPoints_y_cm[0]], color=colours_plot[i])
				# plot text
				plt.text((points[i][0] + points[i+1][0])/2. * agentLength / length, (points[i][1] + points[i+1][1])/2. * agentLength / length, str(round(distances_cm[i], 1)), color='white')

			# FOR points
			for i in range(N):
				plt.plot(pointsInTrack[i][0] * agentLength / length, pointsInTrack[i][1] * agentLength / length, 'bo', color='red')

			plt.xlabel('(cm)')
			plt.ylabel('(cm)')
			plt.pause(0.000000001)

			# append into arrays
			timeArr = np.append(timeArr, seconds)
			distancesArr.append(distances_cm)
			# write CSV file
			firstrow = [seconds]
			for i in range(N-1):
				firstrow.append(distances_cm[i])
			writer.writerow(firstrow)
			del(firstrow)	# free memory

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
	distancesArr = np.array(distancesArr)
	for i in range(N-1):
		distancesArr[:,i] = scipy.signal.filtfilt(b, a, distancesArr[:,i])

	# Plot
	plt.figure(2)
	for i in range(N-1):
		plt.plot(timeArr, distancesArr[:,i], label='distance {}-{} cm'.format(i+1, i+2))
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
