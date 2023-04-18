import cv2
import dlib
from screeninfo import get_monitors
import threading
from copy import deepcopy
import numpy as np


def get_resolution():
    for m in get_monitors():
        if m.is_primary:
            h, w = m.height, m.width
    return h, w


def process_frame(image):
    image = cv2.resize(image, None, fx=0.4, fy=0.4)
    detector = dlib.get_frontal_face_detector()
    faces = detector(image)

    return len(faces)


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

        # Here we extract the primary screen resolution to adjust the size of the video frame
        # Note: if the camera resolution is less than the screen resolution, then it will be set to max cam resolution
        self.h, self.w = get_resolution()
        self.video.set(3, self.w)
        self.video.set(4, self.h)
        (self.grabbed, self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        self.video.release()

    # We perform our image processing here
    def get_frame(self):
        image = self.frame
        img_h, img_w, _ = image.shape
        num_faces = process_frame(deepcopy(image))

        image = cv2.putText(image,
                            f'Faces Detected: {num_faces}',
                            (int(0.1*img_w), int(0.1*img_h)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 0, 0),
                            2,
                            cv2.LINE_AA)

        image = cv2.resize(image, None, fx=1.5, fy=1.5)
        print(image.shape)
        _, jpeg = cv2.imencode('.jpeg', image)
        return jpeg.tobytes()

    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

