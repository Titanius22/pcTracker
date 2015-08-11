"""
TODO
-DONE-make "reset" test happen only every couple of seconds to save calculations
-Done-make "reset" caibrate to new color
-make function to tell servos to move 
-make option to switch to face tracking



"""
import numpy as np
import cv2
import time

# How ofter will it check for 'reset'
resetCounter = 5

# Resolution
camWidth = 320
camHeight = 240

# Track box size
trackWidth = 100
trackHeight = 80

# Hue detction range
lower_hue = np.array([hueMin, 50, 50], dtype=np.uint8)
upper_hue = np.array([hueMax,255,255], dtype=np.uint8)

"""
HUE VALUES
Red = 0-15
Orangle 15-25
Yellow 25-35
Green 35-80
Cyan 85-95
Blue 100-135
Purple 140-155
Red 160-180
"""
def Calibrate_NewObject():
    # Give time for object to back away
    time.sleep(.4)
    
    # take first frame of the video
    ret,frame = cap.read()

    # setup initial location of window
    r,h,c,w = int(.5*(camHeight-trackHeight)), trackHeight, int(.5*(camWidth-trackWidth)), trackWidth  # Defined above. used int() to keep type integer
    track_window = (c,r,w,h)

    # set up the ROI for tracking
    roi = frame[r:r+h, c:c+w] #creat smaller frame
    hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV) #convert to HSV

    mask = cv2.inRange(hsv_roi, lower_hue, upper_hue) #np.array((0., 60.,32.)), np.array((180.,255.,255.))
    roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180]) #([hsv_roi],[0],mask,[180],[0,180])
    cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
    
    return track_window, roi_hist

# created a camera object and connects to webcam
cap = cv2.VideoCapture(0)

# take first frame of the video
ret,frame = cap.read()

# setup initial location of window
r,h,c,w = int(.5*(camHeight-trackHeight)), trackHeight, int(.5*(camWidth-trackWidth)), trackWidth  # Defined above. used int() to keep type integer
track_window = (c,r,w,h)

# set up the ROI for tracking
roi = frame[r:r+h, c:c+w] #creat smaller frame
hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV) #convert to HSV

mask = cv2.inRange(hsv_roi, lower_hue, upper_hue) #np.array((0., 60.,32.)), np.array((180.,255.,255.))
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180]) #([hsv_roi],[0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

# Bitwise-AND mask and original image
res = cv2.bitwise_and(roi, roi, mask= mask)

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while(True):
    ret ,frame = cap.read()

    if True: #ret == True:
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        
        preShift = track_window

        # Apply meanshift to get the new location
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)
        
        # The change  X and Y. posative isup and to the right
        deltaPos = cv2.subtract((track_window[0],track_window[1]), (preShift[0], preShift[1]))
        
  
        # Draw it on image
        x,y,w,h = track_window
        img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
        #img2 = cv2.rectangle(frame, (48,48), (152,132), 255,2)
        #img2 = cv2.rectangle(frame, (48,48), (49,48), 255,0)
        print "Its working " + str(x) + " " + str(ret) 
        
        cv2.imshow('img2',img2)
        
        k = cv2.waitKey(60) & 0xff
        if k == 27:
            #cv2.imwrite("ball.jpg",img2[50:130,50:150])
            #cv2.imwrite("ballInImage.jpg",dst)
            #cv2.imwrite("me.jpg",img2)
            #cv2.imwrite("circleInImage.jpg",img2)

            break
        else:
            cv2.imwrite(chr(k)+".jpg",img2)
            
        # Used as a 'reset' button to start tracking new object TODO
        resetCounter = resetCounter -1
        if resetCounter == 0:
            test = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #frameMean = int((cv2.meanStdDev(test))[0])
            frameStdDev = int((cv2.meanStdDev(test))[1])
            if frameStdDev < 10:
                track_window, roi_hist = Calibrate_NewObject()
            resetCounter = 5     
    else:
        break

cv2.destroyAllWindows()
cap.release()