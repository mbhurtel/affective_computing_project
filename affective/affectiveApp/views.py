from django.shortcuts import render
from django.views.decorators import gzip
from django.http import StreamingHttpResponse
from django.core.paginator import Paginator
from . models import Song
from . import utils as ut
from django.views.decorators.csrf import csrf_exempt
import json
import time
from django.views import View
import platform

cam = ut.VideoCamera()

def index(request):
    global cam
    print("Cam Object", cam)
    if cam.final_hand_gesture and cam.final_hand_gesture == "reset":
        if platform.system() == "Windows":
            cam.video.release()
        cam = ut.VideoCamera()
        request.GET._mutable = True
        request.GET.pop("page", "")
        request.GET._mutable = False

    page_number = request.GET.get('page')

    print("Cam Final Emotion", cam.final_emotion)

    if (cam and cam.final_emotion):
        cam.reloaded = True
        paginator = Paginator(Song.objects.filter(genre=cam.final_emotion), 1)
        cam.final_hand_gesture = "play"  # To make the music play initially when fetched for the first time.
        print("Sending Songs for emotion: ", cam.final_emotion)
    else:
        paginator = Paginator(Song.objects.all(), 1)
        print("Sending all Songs")
    page_obj = paginator.get_page(page_number)
    num_pages = paginator.num_pages
    context = {"page_obj": page_obj, 'total': num_pages, 'current_page' :  page_number if page_number else 1}
    return render(request, "index.html", context=context)


@gzip.gzip_page
def live(request):
    try:
        global cam
        # cam = ut.VideoCamera()
        return StreamingHttpResponse(ut.gen(cam), content_type="multipart/x-mixed-replace;boundary=frame")
    except Exception:
        print(f"Oops!: {Exception}")
        pass
    return render(request, 'index.html')


def event_stream():
    initial_data = {}
    while True:
        data = {
            "final_emotion": cam.final_emotion,
            "is_music_on": cam.is_music_on,
            'final_hand_gesture': cam.final_hand_gesture,
            "reloaded": cam.reloaded
        }

        # if cam.final_emotion:
        #     data["songs"] = list(Song.objects.filter(genre=cam.final_emotion).values())

        if ut.hasChanged(initial_data, data):
            # print("Previous Hand Gesture: ", initial_data.get("final_hand_gesture", "None"))
            # print("Final Hand Gesture:", cam.final_hand_gesture)
            initial_data = data
            data = json.dumps(data)
            if cam.final_hand_gesture == "vUp" or cam.final_hand_gesture == "vDown":
                cam.final_hand_gesture = None
            yield "\ndata: {} \n\n".format(data)
        time.sleep(1)


class MesssageStreamView(View):
    @csrf_exempt
    def get(self, request):
        response = StreamingHttpResponse(event_stream())
        response["Content-Type"] = 'text/event-stream'
        return response


# Page to play music
def play(request):
    paginator = Paginator(Song.objects.all(), 1)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    context = {"page_obj": page_obj}
    return render(request, "play.html", context)

# @csrf_exempt
# @gzip.gzip_page
# def expression(request):
#     uri = json.loads(request.body)['image_uri']
#     expression = getExpression(uri)
#     return JsonResponse({"mood": expression})


def fetch_songs(request):
    if request.GET.get("emotion"):
        emotion = request.GET.get("emotion")
        paginator = Paginator(Song.filter.all(genre=emotion), 1)
        page_number = request.GET.get('page')
        page_obj = paginator.get_page(page_number)
        context = {"page_obj": page_obj}
        print("Songs Fetched")
        return render(request, "index.html", context=context)


