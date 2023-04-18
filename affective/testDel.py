import cv2
from screeninfo import get_monitors

def get_resolution():
    for m in get_monitors():
        if m.is_primary:
            h, w = m.height, m.width
    return h, w

cap = cv2.VideoCapture(0)

print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
h, w = get_resolution()
# print(h, w)
cap.set(3, w)  # cv2.CAP_PROP_FRAME_WIDTH
cap.set(4, h) #cv2.CAP_PROP_FRAME_HEIGHT

print(cap.get(3))
print(cap.get(4))
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
    cv2.imshow("frame", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()