import cv2
import face_recognition

imgTata = face_recognition.load_image_file('images/akshay 12345.jpeg')
imgTata = cv2.cvtColor(imgTata,cv2.COLOR_BGR2RGB)

imgTest = face_recognition.load_image_file('images/sumit 1122.jpeg')
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgTata)[0]
encodeTata = face_recognition.face_encodings(imgTata)[0]
cv2.rectangle(imgTata,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)
result = face_recognition.compare_faces([encodeTata],encodeTest)

dist = face_recognition.face_distance([encodeTata],encodeTest)

print(result)
print(dist)

cv2.imshow('virat',imgTata)
cv2.imshow('elonmusk',imgTest)

cv2.waitKey(0)