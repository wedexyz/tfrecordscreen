import cv2
import pyrebase
import time

import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

#firebase
firebaseConfig={
    "apiKey": "AIzaSyBYVwVZUBLXSm7iR5Fp6k-dziJGEuhwExk",
    "authDomain": "megaboth007.firebaseapp.com",
    "databaseURL": "https://megaboth007.firebaseio.com",
    "projectId": "megaboth007",
    "storageBucket": "megaboth007.appspot.com",
    "messagingSenderId": "942424390212",
    "appId": "1:942424390212:web:c3622743b0fba57b5a1a11"
    }
firebase=pyrebase.initialize_app(firebaseConfig)
db=firebase.database()


#thres = 0.45 # Threshold to detect object
thres = 0.7
cap = cv2.VideoCapture('walking.mp4')
cap.set(3,1280)
cap.set(4,720)
cap.set(10,70)

classNames= []
classFile = 'coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


while True:
    success,img = cap.read()
    classIds, confs, bbox = net.detect(img,confThreshold=thres)
    #print(classIds,bbox)
    cv2.line(img,(1280,360),(0,360),(0,255,0),2)
    cv2.line(img,(640,0),(640,720),(0,255,0),2)

    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            cv2.rectangle(img,box,color=(0,255,0),thickness=2)
            cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30), cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,0),1)
            cv2.putText(img,str(round(confidence*100,2))+' %',(box[0]+100,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,255),1)

            ## Menghitung Jarak ##
            x = box[0]
            y = box[1]
            h = box[2]
            w = box[3]
           
            lebar = w / 20
            yeye = y + h/3
            reye = x + (w/2) - (w/5)
            leye = x + (w/2) + (w/5)
            space = leye - reye
            f = 690
            r = 10
            distance = f * r / space
            distance_in_cm = int(distance)
            cv2.putText(img, str(distance_in_cm)+' cm', (box[0]+200,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,255),1)
            print(distance)
        ''''
            x=box[0]
            if x >500 and x <700 :
                cv2.line(img,(1280,360),(0,360),(255,0,255),4)
                cv2.line(img,(640,0),(640,720),(255,0,255),4)
                print("tengah")
                db.child("suara").set("0")
                #time.sleep(1)
            elif x >=300 and x <=550  :
                cv2.line(img,(1280,360),(0,360),(255,255,255),4)
                cv2.line(img,(640,0),(640,720),(255,255,255),4)
                print ("kiri") 
                db.child("suara").set("4")
               # time.sleep(3)
            elif x >=750 and x <=900:
                cv2.line(img,(1280,360),(0,360),(0,255,255),4)
                cv2.line(img,(640,0),(640,720),(0,255,255),4)
                db.child("suara").set("3")
                print("kanan")
                #time.sleep(3)
            else :
                db.child("suara").set("1")
                  print("jalan")
             '''
            
    cv2.imshow("Output",img)
    cv2.waitKey(1)