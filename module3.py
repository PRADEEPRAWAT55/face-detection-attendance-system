import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

def countAttendance(id):
    with open('markAttendance.csv', 'r+') as f:
         list = f.readlines()


# p = os.path.splitext(imagedataset)
#     id = p[0].split()
#     sList.append(id[0])
#     studentID.append(id[1])