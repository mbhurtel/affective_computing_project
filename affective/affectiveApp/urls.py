from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("live", views.live, name = "live_camera"),
    path("play/", views.play, name="play")
]