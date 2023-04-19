import cv2
import dlib
from screeninfo import get_monitors
import threading
from copy import deepcopy
import tensorflow as tf
import numpy as np

my_model = tf.keras.models.load_model("affectiveApp/custom_augmented_50.h5")
Classes = ["angry" , "disgust" , "fear", "happy", "neutral", "sad", "surprise"]
def get_resolution():
    for m in get_monitors():
        if m.is_primary:
            h, w = m.height, m.width
    return h, w


def detect_faces(image):
    # image = cv2.resize(image, None, fx=0.4, fy=0.4)
    detector = dlib.get_frontal_face_detector()
    faces = detector(image)
    for face in faces:
        # Getting the x1,y1 and x2,y2 coordinates of the face detected
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        image = cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 1)

    return image, faces

def detect_facial_emotion(clone, faces):
    # TODO: Facial Emotion Recognition Code Here
    emotion_list = []
  
    for face in faces:
        print("Face Type: ",type(face))
        print("Face: ",face)
        face_image = clone[face.top():face.bottom(), face.left():face.right()]
        img = cv2.resize(face_image, (48, 48))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        X = np.array(img).reshape(-1, 48, 48, 1)
        X = X / 255.0
        print(X.shape)
        predicted_emotion = my_model.predict(X)

        emotion = Classes[np.argmax(predicted_emotion)]
        print(f"Detected Emotion {emotion}")
        emotion_list.append(emotion)
    return emotion_list


def get_final_max_pred(pred_cat_list):
    pred_cat_count = {pred_cat: 0 for pred_cat in set(pred_cat_list)}
    for pred_cat in pred_cat_list:
        pred_cat_count[pred_cat] += 1
    return max(pred_cat_count, key=pred_cat_count.get)


def check_hand_detection(image):
    # TODO: Here we use the hand detecteion xml to check whether a hand is detected or not
    return False


def detect_hand_gesture(hand_bbox):
    # TODO: Hand Gesture Recognition Code Here
    return "pause"


def get_hand_gesture_and_annotate(image):
    img_h, img_w, _ = image.shape

    play_text = "Playing a song for you. Enjoy..."
    print(play_text)
    image = cv2.putText(image,
                        play_text,
                        (int(0.06 * img_w), int(0.1 * img_h)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 0, 0),
                        2,
                        cv2.LINE_AA)

    hand_bbox_coords = (int(0.75 * img_w), int(0.3 * img_h)), \
                       (int(0.95 * img_w), int(0.7 * img_h))

    image = cv2.rectangle(image,
                          hand_bbox_coords[0],
                          hand_bbox_coords[1],
                          (0, 255, 0),
                          thickness=2)

    # Check whether a hand is detected in the given frame or not
    is_hand_detected = check_hand_detection(image)

    # hand is detected then we check the hand_gesture
    if not is_hand_detected:
        hand_gesture = None
        hand_detect_text = "No hand detected!"
        hand_text_color = (255, 0, 0)
    else:
        # Extracting the bounding box of the hand_region
        hand_bbox = image[hand_bbox_coords[0][1]: hand_bbox_coords[1][1],
                          hand_bbox_coords[0][0]: hand_bbox_coords[1][0]]

        hand_gesture = detect_hand_gesture(hand_bbox)
        hand_detect_text = "Hand detected! Checking gesture..."
        hand_text_color = (0, 0, 255)

    print(hand_detect_text)
    image = cv2.putText(image,
                        hand_detect_text,
                        (int(hand_bbox_coords[0][0] - (0.02 * img_w)), int(hand_bbox_coords[0][0] - (0.02 * img_w))),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        hand_text_color,
                        2,
                        cv2.LINE_AA)

    # TODO: Show controls of play, pause, stop, next and prev

    return image, is_hand_detected, hand_gesture


def annotate_initials(image,
                      num_faces,
                      img_w,
                      img_h,
                      ):
    face_text_x_scale = 0.04
    face_text_y_scale = 0.1

    image = cv2.putText(image,
                        f'Faces Detected: {num_faces} => Checking emotions...',
                        (int(face_text_x_scale * img_w), int(face_text_y_scale * img_h)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 0, 0),
                        2,
                        cv2.LINE_AA)

    return image


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

        # Here we extract the primary screen resolution to adjust the size of the video frame
        # Note: if the camera resolution is less than the screen resolution, then it will be set to max cam resolution
        self.h, self.w = get_resolution()
        self.video.set(3, self.w)
        self.video.set(4, self.h)

        self.fps = self.video.get(cv2.CAP_PROP_FPS)

        self.start_face_count = 0
        self.emotion_list = []
        self.final_emotion = None
        self.is_music_on = False
        self.hand_gesture_list = []
        self.final_hand_gesture = None
        self.is_hand_detected = True

        (self.grabbed, self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        self.video.release()

    # We perform our image processing here
    def get_frame(self):
        image = self.frame
        img_h, img_w, _ = image.shape
        image = cv2.flip(image, 1)  # flipping to remove the mirror effect

        if not self.is_music_on:
            # We check the final emotions every 5 seconds
            if not len(self.emotion_list) == self.fps * 1:
                image, faces = detect_faces(deepcopy(image))  # detecting number of faces
                frame_emotions = detect_facial_emotion(deepcopy(image), faces)
                self.emotion_list += frame_emotions
            else:
                self.final_emotion = get_final_max_pred(self.emotion_list)
                print(f"Your emotion is: {self.final_emotion}. Now playing music...")
                self.is_music_on = True

        if not self.final_emotion:
            image = annotate_initials(image, len(faces), img_w, img_h)  # annotating the video frames for face detection

        # We extract the isMusicPlaying flag from the frontend
        if self.is_music_on:
            image, is_hand_detected, hand_gesture = get_hand_gesture_and_annotate(image)

        if len(self.hand_gesture_list) == self.fps * 2:
            self.final_hand_gesture = get_final_max_pred(self.hand_gesture_list)

        image = cv2.resize(image, None, fx=1, fy=1)  # resizing the video frame to 1080P
        _, jpeg = cv2.imencode('.jpeg', image)
        return jpeg.tobytes()

    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

