from django.shortcuts import render
from django.views.decorators import gzip
from django.http import StreamingHttpResponse
import cv2
import threading
import dlib
from django.core.paginator import Paginator
from . models import Song

from screeninfo import get_monitors

@gzip.gzip_page
def index(request):
    try:
        cam = VideoCamera()
        return StreamingHttpResponse(gen(cam), content_type="multipart/x-mixed-replace;boundary=frame")
        # return StreamingHttpResponse(gen(cam), content_type="image/jpeg")
    except:
        pass
    return render(request, 'index.html')


def get_resolution():
    for m in get_monitors():
        if m.is_primary:
            h, w = m.height, m.width
    return h, w


def process_frame(image):
    image = cv2.resize(image, None, fx=0.4, fy=0.4)
    detector = dlib.get_frontal_face_detector()
    faces = detector(image)

    # for face in faces:
    #     # Getting the x1,y1 and x2,y2 coordinates of the face detected
    #     x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
    #     image = cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 1)

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

    def get_frame(self):
        image = self.frame
        img_h, img_w, _ = image.shape
        # h, w = get_resolution()
        num_faces = process_frame(image.copy())
        image = cv2.putText(image,
                            f'Faces Detected: {num_faces}',
                            (int(0.1*img_w), int(0.1*img_h)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 0, 0),
                            2,
                            cv2.LINE_AA)


        # image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)

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


def play(request):
    paginator = Paginator(Song.objects.all(), 1)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    context = {"page_obj": page_obj}
    return render(request, "play.html", context)
