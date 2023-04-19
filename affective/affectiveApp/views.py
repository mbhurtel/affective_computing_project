from django.shortcuts import render
from django.views.decorators import gzip
from django.http import StreamingHttpResponse
from django.core.paginator import Paginator
from . models import Song
from . import utils as ut

def index(request):
    paginator = Paginator(Song.objects.all(), 1)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    context = {"page_obj": page_obj}
    return render(request, "index.html", context=context)

@gzip.gzip_page
def live(request):
    try:
        cam = ut.VideoCamera()
        return StreamingHttpResponse(ut.gen(cam), content_type="multipart/x-mixed-replace;boundary=frame")
    except Exception:
        print(f"Oops!: {Exception}")
        pass
    return render(request, 'index.html')


# Page to play music
def play(request):
    paginator = Paginator(Song.objects.all(), 1)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    context = {"page_obj": page_obj}
    return render(request, "play.html", context)
