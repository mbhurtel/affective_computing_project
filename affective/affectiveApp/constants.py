import string
import tensorflow as tf
import mediapipe as mp

face_detect = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
hand_detect = mp.solutions.hands.Hands()  # Default parameters are set

emotion_detector = tf.keras.models.load_model("affectiveApp/trained_models/facial_emotion_reconizer.h5")
gesture_detector = tf.keras.models.load_model("affectiveApp/trained_models/hand_gesture_recognizer.h5")

emotions_classes = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
letters_map = dict(enumerate(string.ascii_uppercase))
music_btn_cls = {
    'A': "stop",
    'M': "stop",
    'S': "stop",
    'N': "stop",
    'Y': "pause",
    'L': "pause",
    'O': "play",
    'F': 'play',
    'G': "next",
    'H': "next",
    'T': "next",
    'P': "next",
    'U': "prev",
    'V': "prev"
}