import cv2
import numpy as np
import face_recognition
import os
import requests

path=os.path.join(os.getcwd(),"DataModels")
images=[]
names=[]
persons=os.listdir(path)

for person in persons:
    current=cv2.imread(f'{path}\{person}')
    images.append(current)
    names.append(os.path.splitext(person)[0])

def findEncodings(images):
    encodeList=[]
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnowns=findEncodings(images)


ip=input("Enter IP:")
while True:
    img=requests.get("http://"+ip+":8080/shot.jpg")
    vid=np.array(bytearray(img.content),dtype=np.uint8)
    img=cv2.imdecode(vid,-1)
    imgSmall=cv2.resize(img,(0,0),None,0.25,0.25)

    facesCurFrame=face_recognition.face_locations(imgSmall)
    encodesCurFrame=face_recognition.face_encodings(imgSmall,facesCurFrame)

    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        match=face_recognition.compare_faces(encodeListKnowns,encodeFace)
        faceDis=face_recognition.face_distance(encodeListKnowns,encodeFace)
        matchIndex=np.argmin(faceDis)
        if match[matchIndex]:
            name=names[matchIndex].upper()
            print(name)
            y1,x2,y2,x1=faceLoc
            imgSmall=imgSmall[y1:y2,x1:x2]
    cv2.imshow("frame",imgSmall)
    cv2.waitKey(1)

    if key==27:
        break

cv2.destroyAllWindows()