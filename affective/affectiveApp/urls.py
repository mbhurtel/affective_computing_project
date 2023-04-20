from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("live/", views.live, name = "live_camera"),
    path("play/", views.play, name="play"),
    path("message/", views.MesssageStreamView.as_view(), name="message"),
    path("fetch/", views.fetch_songs, name="fetch_song"),
]