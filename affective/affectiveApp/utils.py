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
            predicted_emotion = ct.emotion_detector.predict(img, verbose=False)
            emotion = ct.emotions_classes[np.argmax(predicted_emotion)]
            emotion_list.append(emotion)
    return emotion_list


def get_final_max_pred(pred_cat_list):
    pred_cat_count = {pred_cat: 0 for pred_cat in set(pred_cat_list)}
    for pred_cat in pred_cat_list:
        pred_cat_count[pred_cat] += 1
    return max(pred_cat_count, key=pred_cat_count.get)


def detect_hands_and_landmarks(hand_bbox, hand_no=0):
    img_rgb = cv2.cvtColor(hand_bbox, cv2.COLOR_BGR2RGB)
    results = ct.hand_detect.process(img_rgb)
    lm_dict = {}
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[hand_no]
        for ID, lm in enumerate(hand.landmark):
            h, w, c = hand_bbox.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            lm_dict[ID] = (cx, cy)
        for hand_lmks in results.multi_hand_landmarks:
            ct.hand_draw.draw_landmarks(hand_bbox, hand_lmks, ct.mp_hands.HAND_CONNECTIONS)

    return lm_dict, hand_bbox


def detect_hand_gesture(lm_dict):
    check_play_stop = []
    check_thumbs_prev_next = []
    check_pause = []
    check_reset = []
    for ID, (_, cy) in lm_dict.items():
        if ID != 4:
            if lm_dict[ID][1] > lm_dict[4][1]:
                check_play_stop.append(True)
            else:
                check_play_stop.append(False)

            if lm_dict[ID][0] > lm_dict[4][0]:
                check_thumbs_prev_next.append(True)
            else:
                check_thumbs_prev_next.append(False)

        if lm_dict[6][1] > lm_dict[7][1] > lm_dict[8][1]:
            if ID not in [6, 7, 8]:
                if lm_dict[ID][1] > lm_dict[6][1]:
                    check_pause.append(True)
                else:
                    check_pause.append(False)

        if ID != 12:
            if lm_dict[ID][1] > lm_dict[12][1]:
                check_reset.append(True)
            else:
                check_reset.append(False)

    hand_gesture = ""
    if check_play_stop:
        if all(check_play_stop):
            hand_gesture = "play"  # thumbs up
        if not any(check_play_stop):
            hand_gesture = "stop"  # thumbs down

    if check_thumbs_prev_next:
        if all(check_thumbs_prev_next):
            hand_gesture = "prev"  # thumbs left
        if not any(check_thumbs_prev_next):
            hand_gesture = "next"  # thumbs right

    if check_pause:
        if all(check_pause):
            hand_gesture = "pause"  # only fore finger up

    if check_reset:
        if all(check_reset):
            hand_gesture = "reset"  # talk to my hands

    return hand_gesture


def get_hand_gesture_and_annotate(image, hand_bbox_coords):
    img_h, img_w, _ = image.shape

    # Here we put the music control buttons after the music is played
    controls = deepcopy(list(ct.play_prompts.keys()))
    controls.remove("others")

    y = int(0.85 * img_h), int(0.95 * img_h)
    x_start = 0.1
    xbox_w = int(0.1 * img_w)
    gap = 0.04
    x_coord = [(control,
                (int((x_start * (i + 1) * img_w) + (gap * i * img_w)),
                 int((x_start * (i + 1) * img_w) + xbox_w + (gap * i * img_w)))
                ) for i, control in enumerate(controls)]

    for control_name, x in x_coord:
        control_img = cv2.imread(f"../../affective_computing_project/affective/gestures/{control_name}.JPG")
        target_box = image[y[0]: y[1], x[0]: x[1]]
        control_img = cv2.resize(control_img, target_box.shape[:2][::-1], interpolation=cv2.INTER_AREA)
        image[y[0]: y[1], x[0]: x[1]] = control_img
        cv2.rectangle(image, (x[0], y[0]), (x[1], y[1]), (255, 255, 0), 1)
        image = cv2.putText(image,
                            control_name,
                            (int(x[0] + (0.02 * img_w)), int(y[1] + (0.03 * img_h))),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.80,
                            (0, 0, 255),
                            2,
                            cv2.LINE_AA)

    # Extracting the bounding box of the hand_region
    hand_bbox = image[hand_bbox_coords[0][1]: hand_bbox_coords[1][1],
                      hand_bbox_coords[0][0]: hand_bbox_coords[1][0]]

    # Check whether a hand is detected in the given frame or not
    lm_dict, hand_bbox = detect_hands_and_landmarks(hand_bbox)

    # hand is detected then we check the hand_gesture
    if not lm_dict:
        hand_gesture = None
        hand_text_color = (0, 0, 255)
    else:
        hand_text_color = (0, 255, 0)
        hand_gesture = detect_hand_gesture(lm_dict)

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
                        (0, 255, 0),
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
        self.reloaded = False

        (self.grabbed, self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        self.video.release()

    # We perform our image processing here
    def get_frame(self):
        image = self.frame
        img_h, img_w, _ = image.shape
        hand_bbox_pos = "right"

        if hand_bbox_pos == "right":
            w_scale = (0.7, 0.95)
        else:
            w_scale = (0.05, 0.30)

        hand_bbox_coords = (int(w_scale[0] * img_w), int(0.3 * img_h)), \
                           (int(w_scale[1] * img_w), int(0.7 * img_h))

        image = cv2.flip(image, 1)  # flipping to remove the mirror effect

        if not self.is_music_on:
            # We check the final emotions every 5 seconds
            if not len(self.emotion_list) == self.fps * 2:
                image, faces = detect_faces(deepcopy(image))  # detecting number of faces
                frame_emotions = detect_facial_emotion(deepcopy(image), faces)
                self.emotion_list += frame_emotions
            else:
                self.final_emotion = get_final_max_pred(self.emotion_list)
                self.is_music_on = True
                self.final_hand_gesture = "play"

        if not self.final_emotion and not self.is_music_on:
            image = annotate_initials(image, len(faces), img_w, img_h)  # annotating the video frames for face detection

        # We extract the isMusicPlaying flag from the frontend
        if self.is_music_on:
            image, hand_gesture = get_hand_gesture_and_annotate(image, hand_bbox_coords)

            if len(self.hand_gesture_list) != self.fps * 1 and hand_gesture:
                self.hand_gesture_list.append(hand_gesture)
            elif len(self.hand_gesture_list) == self.fps * 1:
                self.final_hand_gesture = get_final_max_pred(self.hand_gesture_list)
                self.hand_gesture_list = []
                print(f"Your emotion is: {self.final_emotion}. Now playing music...")

        if self.final_hand_gesture:
            play_text = ct.play_prompts[self.final_hand_gesture]
            image = cv2.putText(image,
                                play_text,
                                (int(0.06 * img_w), int(0.1 * img_h)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (255, 0, 0),
                                2,
                                cv2.LINE_AA)

        image = cv2.resize(image, None, fx=0.9, fy=0.9)  # resizing the video frame to 1080P
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