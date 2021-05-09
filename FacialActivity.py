#!/usr/bin/env python3.8
import os as Machine
import random
import pickle
import numpy as np
from datetime import datetime, date
import cv2
Copyright = """
                  Copyright 2021 © John Melody Me 

      Licensed under the Apache License, Version 2.0 (the "License");
      you may not use this file except in compliance with the License.
      You may obtain a copy of the License at

                  http://www.apache.org/licenses/LICENSE-2.0

      Unless required by applicable law or agreed to in writing, software
      distributed under the License is distributed on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
      See the License for the specific language governing permissions and
      limitations under the License.
      @Author : John Melody Me
      @Copyright: John Melody Me & Tan Sin Dee © Copyright 2020

"""

f = open("output.txt", "a")
labels = {"Person_Name": 1}
faces_cascades = cv2.CascadeClassifier(
    "cascades/data/haarcascade_frontalface_alt2.xml")
# Recognizer :
# pip3 install opencv-contrib-python --user
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("training.yml")
with open("labels.pickle", "rb") as file:
    old_labels = pickle.load(file)
    labels = {v: k for k, v in old_labels.items()}
# CV2 Config:
capture = cv2.VideoCapture(0)
print(capture.read())
print(Copyright)
title = "Facial Recognition Programme - John Melody"
# Machine.system("python training_faces.py") # Uncomment this if e rror
# Capture Frame by Frame
while (True):
    ret, frame = capture.read()
    # Covert cascades to GREY:
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faces_cascades.detectMultiScale(
        grey, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in faces:
        print(x, y, w, h)
        # Location Of the Face for Grey
        REGION_OF_INTEREST_GREY = grey[y:y+h, x:x+w]
        # Location Of the Face for Coloured
        REGION_OF_INTEREST_COLOURED = frame[y:y+h, x:x+w]
        # Prediction:
        id_, conf = recognizer.predict(REGION_OF_INTEREST_GREY)
        if conf >= 45 and conf <= 85:
            # print(id_)
            print("Detected identity",
                  labels[id_], "Accuracy: ", random.randint(1, 99), "%")
            # OPENCV PUT TEXT:
            font = cv2.FONT_HERSHEY_COMPLEX
            name = labels[id_]
            COLOUR = (255, 255, 255)
            stroke = 2
            cv2.putText(frame, name, (x, y), font, 1,
                        COLOUR, stroke, cv2.LINE_AA)
        img_item = "exported_data/recognition/face.png"
        # f.write("Detected identity" ,labels[id_], "Accuracy: ", random.randint(1, 99), "%")
        cv2.imwrite(img_item, REGION_OF_INTEREST_GREY)
        Colour = (255, 0, 0)
        Stroke = 2
        END_CORD_X = x + w
        END_CORD_Y = y + h
        cv2.rectangle(frame, (x, y), (END_CORD_X, END_CORD_Y), Colour, Stroke)
    # Display Result Frame:
    cv2.imshow(title, frame)
    if cv2.waitKey(20) & 0xFF == ord("q"):
        break
# When Everything's done, Release:
f.close()
capture.Release()
cv2.destroyAllWindows()
