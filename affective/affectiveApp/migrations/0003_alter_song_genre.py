# Generated by Django 4.2 on 2023-04-19 23:31

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('affectiveApp', '0002_song_genre'),
    ]

    operations = [
        migrations.AlterField(
            model_name='song',
            name='genre',
            field=models.CharField(choices=[('angry', 'angry'), ('disgust', 'disgust'), ('fear', 'fear'), ('happy', 'happy'), ('neutral', 'neutral'), ('sad', 'sad'), ('surprise', 'surprise')], default='neutral', max_length=10),
        ),
    ]
