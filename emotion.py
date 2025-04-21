import cv2 as c
from deepface import DeepFace

img = c.imread(r'basic images\Elon Musk.jpg')
img_m = c.imread(r'C:\Users\Hp\PycharmProjects\FYP\basic images\Abid Farid.jpeg')

predictions = DeepFace.analyze(img_m)
print(predictions)
print(predictions['dominant_emotion'])

# REALTIME EMOTION
# import cv2 as c
# from deepface import DeepFace
#
# faceCascade = c.CascadeClassifier(c.data.haarcascades + 'haarcascade_frontalface_default.xml')
# # faceCascade = c.CascadeClassifier(r'E:\udemy\guide\haarcascades\haarcascade_frontalface_default')
#
# cap = c.VideoCapture(0)
# if not cap.isOpened:
#     raise IOError('cannot open webcam')
#
# # while True:
# while cap.isOpened:
#     try:
#         ret, frame = cap.read()
#
#         result = DeepFace.analyze(frame, actions=['emotion'])
#
#         gray = c.cvtColor(frame, c.COLOR_BGR2GRAY)
#
#         faces = faceCascade.detectMultiScale(gray, 1.1, 4)
#         for (x, y, w, h) in faces:
#             c.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
#
#         c.putText(frame, result['dominant_emotion'], (50, 50), c.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2, c.LINE_4)
#
#         c.imshow('Video', frame)
#     except:
#         c.imshow('Video', frame)
#
#     k = c.waitKey(30) & 0xff
#     if k == 27:
#         break
#
# cap.release()
# c.destroyAllWindows()
#
#
