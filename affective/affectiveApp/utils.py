import cv2
from screeninfo import get_monitors
import threading
from copy import deepcopy
import numpy as np

from . import constants as ct


def get_resolution():
    for m in get_monitors():
        if m.is_primary:
            h, w = m.height, m.width
    return h, w


def detect_faces(image):
    h, w, _ = image.shape
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = ct.face_detect.process(img_rgb)
    face_coords = []
    if faces.detections:
        for face_det in faces.detections:
            bb = face_det.location_data.relative_bounding_box
            xmin, ymin = int(bb.xmin * w), int(bb.ymin * h)
            xmax, ymax = xmin + int(bb.width * w), ymin + int(bb.height * h)
            face_coords.append([xmin, xmax, ymin, ymax])
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

    return image, face_coords


def preprocess_image(img):
    print(img)
    print(img.shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (48, 48))
    X = np.array(img).reshape(-1, 48, 48, 1)
    X = X / 255.0
    return X


def detect_facial_emotion(clone, faces):
    # TODO: Facial Emotion Recognition Code Here
    emotion_list = []
    for face in faces:
        # face_image = clone[face.top():face.bottom(), face.left():face.right()]
        face_image = clone[face[2]:face[3], face[0]:face[1]]
        # print(face_image)
        if len(face_image):
            img = preprocess_image(face_image)
            predicted_emotion = ct.emotion_detector.predict(img)
            emotion = ct.emotions_classes[np.argmax(predicted_emotion)]
            emotion_list.append(emotion)
    return emotion_list


def get_final_max_pred(pred_cat_list):
    pred_cat_count = {pred_cat: 0 for pred_cat in set(pred_cat_list)}
    for pred_cat in pred_cat_list:
        pred_cat_count[pred_cat] += 1
    return max(pred_cat_count, key=pred_cat_count.get)


def check_hand_detection(hand_bbox):
    img_rgb = cv2.cvtColor(hand_bbox, cv2.COLOR_BGR2RGB)
    results = ct.hand_detect.process(img_rgb)
    return True if results.multi_hand_landmarks else False


def detect_hand_gesture(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (28,28))
    view_image = np.expand_dims(img, axis=2)
    test_image = view_image[np.newaxis, ...]
    preds = ct.gesture_detector.predict(test_image, verbose=0)
    pred_class = ct.letters_map[np.argmax(preds, axis=1)[0]]
    music_class = ct.music_btn_cls[pred_class] if pred_class in ct.music_btn_cls.keys() else "others"
    return music_class


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

    hand_bbox_coords = (int(0.05 * img_w), int(0.3 * img_h)), \
                       (int(0.3 * img_w), int(0.7 * img_h))

    # Extracting the bounding box of the hand_region
    hand_bbox = image[hand_bbox_coords[0][1]: hand_bbox_coords[1][1],
                      hand_bbox_coords[0][0]: hand_bbox_coords[1][0]]

    # Check whether a hand is detected in the given frame or not
    is_hand_detected = check_hand_detection(hand_bbox)

    # hand is detected then we check the hand_gesture
    if not is_hand_detected:
        hand_gesture = "No gesture"
        hand_text_color = (0, 0, 255)
    else:
        hand_text_color = (0, 255, 0)
        hand_gesture = detect_hand_gesture(hand_bbox)

    image = cv2.rectangle(image,
                          hand_bbox_coords[0],
                          hand_bbox_coords[1],
                          hand_text_color,
                          thickness=3)

    image = cv2.putText(image,
                        hand_gesture,
                        (int(hand_bbox_coords[0][0]), int(hand_bbox_coords[0][1] - (0.05 * img_h))),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        hand_text_color,
                        2,
                        cv2.LINE_AA)

    # TODO: Show controls of play, pause, stop, next and prev
    return image, hand_gesture


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
        self.reloaded = False

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
            if not len(self.emotion_list) == self.fps * 5:
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
            image, hand_gesture = get_hand_gesture_and_annotate(image)
            if hand_gesture != "others":
                self.hand_gesture_list.append(hand_gesture)

        if len(self.hand_gesture_list) == self.fps * 2:
            self.final_hand_gesture = get_final_max_pred(self.hand_gesture_list)

        # image = cv2.resize(image, None, fx=1, fy=1)  # resizing the video frame to 1080P
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


def hasChanged(data1, data2):
    if type(data1) == type(data2):
        for k in data2.keys():
            if data1.get(k, "") != data2[k]:
                return True
        return False
    return True