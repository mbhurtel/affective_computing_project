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

cam = ut.VideoCamera()

def index(request):
    if cam and cam.final_emotion:
        cam.reloaded = True
        paginator = Paginator(Song.objects.filter(genre=cam.final_emotion), 1)
        print("Sending Songs for emotion: ", cam.final_emotion)
        cam.final_emotion = None
        # cam.final_hand_gesture = None
    else:
        paginator = Paginator(Song.objects.all(), 1)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    context = {"page_obj": page_obj}
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
            'final_hand_gesture' : cam.final_hand_gesture,
            "reloaded" : cam.reloaded
        }
        
        # if cam.final_emotion:
        #     data["songs"] = list(Song.objects.filter(genre=cam.final_emotion).values())
        
        print("Final Hand Gesture:", cam.final_hand_gesture)
        print("Previous Hand Gesture: ", initial_data.get("final_hand_gesture", "None"))
        if ut.hasChanged(initial_data, data):
            initial_data = data
            data = json.dumps(data)
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
    print("Fetching Songs")
    if request.GET.get("emotion"):
        emotion = request.GET.get("emotion")
        paginator = Paginator(Song.filter.all(genre=emotion), 1)
        page_number = request.GET.get('page')
        page_obj = paginator.get_page(page_number)
        context = {"page_obj": page_obj}
        print("Songs Fetched")
        return render(request, "index.html", context=context)


