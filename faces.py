import numpy as np
import cv2
import pickle
import os

face_cascade = cv2.CascadeClassifier(
    'cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("recognizer.yml")

labels = {}
with open("labels.pickle", "rb") as f:
    labels = pickle.load(f)
labels = {v: k for k, v in labels.items()}
# inverting dictionary


# Video Encoding, might require additional installs
VIDEO_TYPE = {
    'avi': cv2.VideoWriter_fourcc(*'XVID'),
    'mp4': cv2.VideoWriter_fourcc(*'XVID'),
    # 'mp4': cv2.VideoWriter_fourcc(*'H264')
}

# Standard Video Dimensions Sizes
STD_DIMENSIONS = {
    "480p": (640, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "4k": (3840, 2160),
}


def get_video_type(file_name):  # to encode video in given extension
    file_name, ext = os.path.splitext(file_name)
    if ext in VIDEO_TYPE:
        return VIDEO_TYPE[ext]
    return VIDEO_TYPE['avi']

# grab resolution dimensions and set video capture to it.


def get_dims(reso):
    if reso in STD_DIMENSIONS:
        width, height = STD_DIMENSIONS[reso]
    # change the current capture device to the resulting resolution
    cap.set(3, width)
    cap.set(4, height)
    return width, height


cap = cv2.VideoCapture(0)
out = cv2.VideoWriter('video.mp4', get_video_type(
    'video.mp4'), 25, get_dims('480p'))

while True:
    # capture frame-by-frame
    ret, frame = cap.read()

    cv2.imwrite("image.png", frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # haarcascades only work on grayscaled images
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in faces:
        # region of interest
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # recognizing face (can be done using deep learning as well)
        id_, conf = recognizer.predict(roi_gray)
        # returns the label and confidence
        if conf >= 45:
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(frame, name, (x, y-10), font,
                        0.7, color, stroke, cv2.LINE_AA)

        # drawing a rectangle around region of interest
        color = (255, 0, 0)  # BGR
        stroke = 1  # thickness
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, stroke)

    out.write(frame)

    # Display resulting frame
    cv2.imshow('frame', frame)
    # stop display
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
