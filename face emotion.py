import cv2 as c
import numpy as np
import face_recognition as fc
import os
from deepface import DeepFace as df
from datetime import datetime as dt
#EMOTION
face_cascade = c.CascadeClassifier(c.data.haarcascades +'haarcascade_frontalface_default.xml')

path = 'basic images'
images = []
classes = []
mylist = os.listdir(path)

#print(mylist)

for item in mylist:
    currentImage = c.imread(f'{path}\{item}')
    images.append(currentImage)
    classes.append(os.path.splitext(item)[0])
# print(classes)
# print(images)

def findEncodings(image):
    encodeList = []
    for img in image:
        img = c.cvtColor(img, c.COLOR_BGR2RGB)
        encode = fc.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def dataRec(name, emotion):
    with open('face emotion.csv', 'r+') as f:
        myDataList = f.readlines()
        # print(myDataList)

        nameList = []
        # nameList.clear()

        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = dt.now()
            dtString = now.strftime('%H:%S:%M')
            f.writelines(f'\n{name},{emotion},{dtString}')

encodedList = findEncodings(images)
#print(len(encodedList))
# print(encodedList)

cap = c.VideoCapture(0)

while True:

    try:
        ret, frame = cap.read()
        img = c.resize(frame, (800, 600))
        # img = c.resize(frame, (0,0), None, 0.25, 0.25)          #one fourth of the size...0.25

        # EMOTION
        result = df.analyze(img, actions=['emotion'])

        gray = c.cvtColor(img, c.COLOR_BGR2RGB)

        faceLocations = fc.face_locations(gray)
        encodedFrame = fc.face_encodings(gray, faceLocations)

        for faceLoc, encode in zip(faceLocations, encodedFrame):
            mathches = fc.compare_faces(encodedList, encode)
            faceDis = fc.face_distance(encodedList, encode)
            # print(faceDis)

            matchIndex = np.argmin(faceDis)  # with minumum face distance

            if mathches[matchIndex]:
                name = classes[matchIndex]
                print(name)

                # faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                #
                # for (x, y, w, h) in faces:
                #     c.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)

                dis_emotion = result['dominant_emotion']
                nameText = f'Name: {name}'
                emotion = f'Emotion: {dis_emotion}'

                y1, x2, y2, x1 = faceLoc
                # y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                c.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # c.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), c.FILLED)
                c.putText(img, nameText, (x1, y2 + 21), c.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 2)
                c.putText(img, emotion, (x1, y2 + 41), c.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 2)

                dataRec(name, dis_emotion)

        c.imshow('Image', img)

    except:
        c.imshow('Image', img)

    #c.waitKey(1)
    k = c.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
c.destroyAllWindows()




