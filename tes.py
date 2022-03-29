## Original script written by Mbahmbul ##
## Contact me if you find any problem ##
## Email : alwansyah.al888@gmail.com ##
## Github : https://github.com/mbahmbul ##
import cv2

## checking ##
cap = cv2.VideoCapture(0)
ret = cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
ret = cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')



def drawBoxAndWriteText(findfaces):
    for (x, y, w, h) in findfaces:
        
        ## Menghitung Jarak ##
        eyeradius = w / 20
        yeye = y + h/3
        reye = x + (w/2) - (w/5)
        leye = x + (w/2) + (w/5)
        space = leye - reye
        f = 690
        r = 10
        distance = f * r / space
        distance_in_cm = int(distance)
        ##

        cv2.rectangle(color, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(color, str(distance_in_cm)+' cm', (x+5, y+h-5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0))

while(True):
    ret, color = cap.read()
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    drawBoxAndWriteText(faces)
    cv2.imshow('color', color)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()