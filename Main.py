import cv2
from simpleFacerec import SimpleFacerec
from datetime import datetime
import os
import glob

#Encode faces
sfr = SimpleFacerec()
sfr.load_encoding_images("C:/Users/anany/OneDrive/Documents/ML/Face Recognition/images/")
now = datetime.now()
currentdate = now.strftime("%Y-%m-%d")
#creating csv file
f = open('C:/Users/anany/OneDrive/Documents/ML/Face Recognition/Attendance/' + currentdate + '.csv', 'w+', newline = '')
#Load Camera
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    #Detect Faces
    face_location, face_names = sfr.detect_known_faces(frame)
    sfr.add_to_csv(f, now, face_names)
    for face_loc, name in zip(face_location, face_names):
        y1, x1, y2, x2 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
        cv2.putText(frame, name, (1, y1-10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if(key == 27):
        break
cap.release()
cv2.destroyAllWindows()
f.close()