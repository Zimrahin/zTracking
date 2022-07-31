import cv2 as cv

WINDOW_NAME = 'zTracking'


class Tracker:
    def __init__(self, frame):
        
        self.bbox = cv.selectROI(WINDOW_NAME, frame, False)
        self.tracker = cv.TrackerCSRT_create()
        self.tracker.init(frame, self.bbox)
        self.x = None
        self.y = None
        self.w = None
        self.h = None
        self.p = None


    def update(self,frame):
        ret, self.bbox = self.tracker.update(frame)
        return ret

    def getCoordinates(self):
        self.x, self.y, self.w, self.h = int(self.bbox[0]), int(self.bbox[1]), int(self.bbox[2]), int(self.bbox[3])

    def getCentroid(self, height):
        self.p = (self.x + self.w/2., height - self.y - self.h/2.)