import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

directory = 'images'
photos = []
sList = []
studentID=[]
trainingSetList = os.listdir(directory)

for dataset in trainingSetList:
    current_Img = cv2.imread(f'{directory}/{dataset}')
    photos.append(current_Img)
    p = os.path.splitext(dataset)
    id = p[0].split()
    sList.append(id[0])
    studentID.append(id[1])

print('total data :')
print(len(sList))

def faceEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def attendance(name,id):
    days=1
    with open('markAttendance.csv', 'r+') as file:
        myDataList = file.readlines()
        nameList = []
        for row in myDataList:

            entry = row.split(' , ')
            if(entry[0]==name):
                days=days+1
            nameList.append(entry[0])
        if name not in nameList:
            time_now = datetime.now()
            tStr = time_now.strftime('%H:%M:%S')
            dStr = time_now.strftime('%d/%m/%Y')
            file.writelines(f'\n{name} , {tStr} , {dStr} , {id} ,   {days}')

encodeListKnown = faceEncodings(photos)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    faces = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    faces = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)

    facesCurrentFrame = face_recognition.face_locations(faces)
    encodesCurrentFrame = face_recognition.face_encodings(faces, facesCurrentFrame)

    for encodeFace, faceLoc in zip(encodesCurrentFrame, facesCurrentFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        # print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = sList[matchIndex].upper()
            id = studentID[matchIndex]
            print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 223, 0), 2)
            cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (212,175,55), cv2.FILLED)
            cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_ITALIC, 1, (255, 255, 255), 2)
            attendance(name,id)

    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) == 13:
        break

 # readfile = pd.read_csv("C:\\Users\\asus\\Documents\\csv-to-xl\\Attendance.csv")
 # readfile.to_excel("C:\\Users\\asus\\Documents\\csv-to-xl\\Attendance.xlsx", index = None, header=True)

cap.release()
cv2.destroyAllWindows()