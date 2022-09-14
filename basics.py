import cv2
import face_recognition

elon_musk = cv2.imread("resources/Elon_Musk.jpg")
test_img = cv2.imread("resources/Bill_Gates.jpg")

# top, right, bottom, left
face_location = face_recognition.face_locations(elon_musk)[0]
test_face_location = face_recognition.face_locations(test_img)[0]

cv2.rectangle(elon_musk,
        (face_location[3], face_location[0]),
        (face_location[1], face_location[2]),
        (255, 255, 0),
        3        
    )

cv2.rectangle(test_img,
        (test_face_location[3], test_face_location[0]),
        (test_face_location[1], test_face_location[2]),
        (255, 255, 0),
        3        
    )

encodedElon = face_recognition.face_encodings(elon_musk)[0]
encodedTest = face_recognition.face_encodings(test_img)[0]
# print(encodedTest)


comparison_result = face_recognition.compare_faces([encodedElon], encodedTest)
print(comparison_result)


face_distances = face_recognition.face_distance([encodedElon], encodedTest)
print(face_distances)



cv2.imshow("Elon Musk", elon_musk)
cv2.imshow("Test Image", test_img)

cv2.waitKey(10000)          #images will close after 10 seconds
