import cv2 as cv
import os
import argparse

def writeVideo(opt):
    image_folder = 'frames'
    fps = 30

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    frame = cv.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv.VideoWriter(opt.outputName + '.mp4', cv.VideoWriter_fourcc(*'avc1'), fps, (width,height))

    for image in images:
        video.write(cv.imread(os.path.join(image_folder, image)))

    cv.destroyAllWindows()
    video.release()

def parse_opt(known=False):
	parser = argparse.ArgumentParser()
	parser.add_argument('--outputName', default = "output", type=str,help='Name of output file')
	opt = parser.parse_known_args()[0] if known else parser.parse_args()
	return opt

def main(opt):
	writeVideo(opt)

if __name__ == '__main__':
	opt = parse_opt()
	main(opt)