from datetime import datetime
import cv2
import face_recognition
import os
import numpy as np


PATH = "resources"
imgs = []
image_files = os.listdir(PATH)
images = []
person_names = []

for img_file in image_files:
    img = cv2.imread(f"{PATH}/{img_file}")
    images.append(img)
    person_name = img_file.split(".")[0]
    person_names.append(person_name)



def mark_attendance_list(name):
    with open("attendance.csv", "r+") as f:
        current_list = f.readlines()

        names = []

        for item in current_list:
            entry = item.split(",")
            names.append(entry[0])

        if name not in names:
            curr_date = datetime.now()
            formatted_time = curr_date.strftime("%H:%M:%S")
            f.writelines(f"\n{name}, {formatted_time}") 



def faceEncodings(img_list):
    encoding_list = []

    for image in img_list:
        encoding = face_recognition.face_encodings(image)[0]
        encoding_list.append(encoding)

    return encoding_list


known_encoding_list = faceEncodings(images)

cap = cv2.VideoCapture(0)


while True:
    success, img = cap.read()

    # if not success:
    #     break


    test_face_location = face_recognition.face_locations(img)
    test_encoded = face_recognition.face_encodings(img, test_face_location)


    for encoded_face, location in zip(test_encoded, test_face_location):
        matches = face_recognition.compare_faces(known_encoding_list, test_encoded[0])
        face_distances = face_recognition.face_distance(known_encoding_list, test_encoded[0])

        # print(face_distances)

        match_index = np.argmin(face_distances)

        if matches[match_index]:
            name = person_names[match_index]

            y1, x2, y2, x1 = location
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
            cv2.putText(img, name, (x1+8, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

            mark_attendance_list(name)


    cv2.imshow("Image", img)
    cv2.waitKey(1)

cap.release()