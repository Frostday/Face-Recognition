import os
from PIL import Image
import numpy as np
import cv2
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "images")

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
x_train = []
y_labels = []

for root, dirs, files in os.walk(IMAGE_DIR):
    # root is path to folder names like Dhruv Garg
    for f in files:
        # f are images inside a folder like 1.jpg inside Dhruv Garg
        if f.endswith("png") or f.endswith("jpg") or f.endswith("jpeg"):
            label = os.path.basename(root).replace(" ", "-").lower()
            path = os.path.join(root, f)
            # print(label, path)

            # assigning a unique number id to every label
            if label not in label_ids:
                label_ids[label] = current_id
                current_id += 1

            id_ = label_ids[label]
            
            # pre-processing image
            # loading image L converts image to grayscale
            pil_image = Image.open(path).convert("L")
            final_image = pil_image
            # resizing
            # size = (720, 720)
            # final_image = pil_image.resize(size, Image.ANTIALIAS)
            # convert image to numpy array
            image_arr = np.array(final_image, "uint8")

            # finding region of interest in training images
            faces = face_cascade.detectMultiScale(image_arr, scaleFactor=1.5, minNeighbors=5)

            for (x, y, w, h) in faces:
                # print(path)
                roi = image_arr[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)
                # print(roi.shape, id_)

print(label_ids)
print(y_labels)
# to see how many images were actuallyy recognized

# saving label ids
with open("labels.pickle", "wb") as f:
    pickle.dump(label_ids, f)

# training the recognizer
recognizer.train(x_train, np.array(y_labels))
recognizer.save("recognizer.yml")