import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# But what if we have a huge data. For that, we cant write codes of imread/imshow for all the images
# Hence we do the following:-  ( we imported os to do our task )

folder= 'My Classroom'
images = []
classNames = []
myList = os.listdir(folder)
print(myList)

# This function adds all images in images_array and names in classNames.
for filename in myList:
    img=cv2.imread( os.path.join( folder,filename ) )
    if img is not None:
        images.append(img)
        classNames.append( os.path.splitext(filename)[0]  )   # we have used split_text to avoid ".jgp" in the name

print(classNames)
print("\n")

# This fn generates the encoded version of each image and saves those in the array
def findEncodings(images):
    elist= []
    for img in images:
        img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        code= face_recognition.face_encodings(img)[0]
        elist.append(code)
    return elist

def markAttendance(name):
    with open('attendance.csv', 'r+') as f:
        myData= f.readlines()
        namelist=[]
        for line in myData:
            entry= line.split(',')  # splitting on the basis of comma
            namelist.append( entry[0] )
        # Now we'll check if current name is present or not:-
        if name not in namelist:
            now=datetime.now()
            dtstring= now.strftime('%H:%M:%S') # hours,minutes,seconds as 00:00:00
            f.writelines(f'\n{name},{dtstring}')



encodedListKnown= findEncodings(images)
print( " ENCODING COMPLETED   " + "\n")

#Now we'll take input from webcam
cap= cv2.VideoCapture(0)
while True:
    success,img=cap.read()
    imgS= cv2.resize(img,(0,0),None,0.25,0.25) #To reduce the img, hence speeden up the process
    imgS= cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurrFrame= face_recognition.face_locations(imgS)
    encodesCurrFrame= face_recognition.face_encodings(imgS,facesCurrFrame)

    # Now we'll compare this encoding with the main dataset.
    for encodeface,faceloc in zip(encodesCurrFrame,facesCurrFrame):
        matches= face_recognition.compare_faces( encodedListKnown , encodeface )
        faceDist= face_recognition.face_distance( encodedListKnown , encodeface )
        #print(faceDist)
        matchIndex= np.argmin(faceDist)
        # Now we have found out the matching person.

        if matches[matchIndex]:
            name= classNames[matchIndex].upper()  # to make everything in uppercase
            markAttendance(name)
            print(name)
            y1,x2,y2,x1 =  faceloc
            y1, x2, y2, x1 = y1*4 , x2*4 , y2*4 , x1*4   # because we scaled the image previosly
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)

    cv2.imshow("WEBCAM",img)
    cv2.waitKey(1)


cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()