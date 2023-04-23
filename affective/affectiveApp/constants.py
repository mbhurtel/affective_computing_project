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
