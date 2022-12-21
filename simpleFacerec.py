import face_recognition
import cv2
import os
import glob
import numpy as np
from datetime import datetime
import csv

class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.frame_resizing = 0.25
        self.students = []

    def load_encoding_images(self, images_path):
        images_path = glob.glob(os.path.join(images_path, "*.*"))
        print("{} encoding images found.".format(len(images_path)))
        for img_path in images_path:
            img = cv2.imread(img_path)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            basename = os.path.basename(img_path)
            (filename, ext) = os.path.splitext(basename)
            img_encoding = face_recognition.face_encodings(rgb_img)[0]
            self.known_face_encodings.append(img_encoding)
            self.known_face_names.append(filename)
            with open("encodings.txt", "w") as file:
                file.write("\n")
                for pixel in img_encoding:
                    file.write(str(pixel) + " ")
                file.write(filename + "\n")
        print("Encoding images loaded")
        self.students = self.known_face_names.copy()

    def load_encoded_images(self):
        with open("encodings.txt", "r") as file:
            for line in file:
                imgEncodings = []
                enc = file.readline()
                enc = enc.split()
                name = enc[-2] + enc[-1]
                enc = enc[:-2]
                for value in enc:
                    imgEncodings.append(float(value))
                self.known_face_encodings.append(imgEncodings)
                self.known_face_names.append(name)

    def detect_known_faces(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
            face_names.append(name)
        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_resizing
        return face_locations.astype(int), face_names

    def add_to_csv(self, f, now, face_names):
        lnwrite = csv.writer(f)
        for name in face_names:
            if name in self.students:
                self.students.remove(name)
                print(self.students)
                currenttime = now.strftime("%H-%M-%S")
                lnwrite.writerow([name, currenttime])