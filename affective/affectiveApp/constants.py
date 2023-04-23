import tensorflow as tf
import mediapipe as mp

face_detect = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
mp_hands = mp.solutions.hands
hand_detect = mp_hands.Hands()  # Default parameters are set
hand_draw = mp.solutions.drawing_utils

emotion_detector = tf.keras.models.load_model("affectiveApp/trained_models/facial_emotion_reconizer.h5")
gesture_detector = tf.keras.models.load_model("affectiveApp/trained_models/hand_gesture_recognizer.h5")

emotions_classes = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Playing songs text prompts
play_prompts = {
                "play": "Playing song for you. Enjoy...",
                "pause": "Current song paused!",
                "stop": "Current song stopped!",
                "next": "Playing next song for you. Enjoy...",
                "prev": "Playing previous song for you. Enjoy...",
                "reset": "Resetting the process...",
                "others" : ""
                }

"""
Songs for Fearful:
Roar: Katy Perry
Fight Song: Rachel Platten
Dreamer: Ozzy Osbourne
Get Up Stand Up: Bob Marley
Believer: Imagine Dragons

Songs for Happy:
Happy: Pharrel Williams
Neon Moon (Remix): DJ Noiz, Brooks & Dunn
Shalala Lala: Vengaboys
Born To Be Wild: Steppenwolf

Songs for Sad:
Let it be: The Beatles
Coldplay: Fix You Lyrics
Believer: Imagine Dragon
Don't Worry Be Happy: Bobby McFerrins
Blowin' In The Wind: Bob Dylan

Songs for neutral:
Hotel California: Eagles
Stairway to Heaven: Led Zeppelin

Songs for surprise:
Top Of The World: The Carpenters
Price Tag: Jessie J

Songs for disgust:
Let it be: The Beatles
Imagine: John Lennon
Don't Worry Be Happy: Bobby McFerrin
Don't Stop Me Now: Queen
Uptown Funk: Mark Ronson ft. Bruno Mars
"""
