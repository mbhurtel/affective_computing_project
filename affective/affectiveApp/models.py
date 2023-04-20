from django.db import models


# Create your models here.
class Song(models.Model):
    class Genre(models.TextChoices):
        angry = 'angry', ('angry')
        disgust = 'disgust', ('disgust')
        fear = 'fear', ('fear')
        happy = 'happy', ('happy')
        neutral = 'neutral', ('neutral')
        sad = 'sad', ('sad')
        surprise = 'surprise', ('surprise')

    title = models.TextField()
    artist = models.TextField()
    genre = models.CharField(
        max_length = 10,
        choices=Genre.choices,
        default=Genre.neutral
    )
    # likes = models.IntegerField()
    image = models.ImageField()
    audio_file = models.FileField(blank=True, null=True)
    audio_link = models.CharField(max_length=200, blank=True, null=True)
    duration = models.CharField(max_length=20)
    paginate_by = 2

    def __str__(self):
        return self.title
