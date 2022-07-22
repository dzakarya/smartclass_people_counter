from repositories.object_counter.centroidtracker import CentroidTracker

import numpy as np
import cv2
 
# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cv2.startWindowThread()

# open webcam video stream
cap = cv2.VideoCapture(0)
peopleCt = CentroidTracker(maxDisappeared=80, maxDistance=60)
# the output will be written to output.avi
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # using a greyscale picture, also for faster detection
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # detect people in the image
    # returns the bounding boxes for the detected objects
    boxes, weights = hog.detectMultiScale(frame,
                                    winStride=(4, 4),
                                    padding=(4, 4),
                                    scale=1.05)

    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

    rect_people = []
    for (xA, yA, xB, yB) in boxes:
        rect_people.append((xA,yA,xB,yB))
        # display the detected boxes in the colour picture
        cv2.rectangle(frame, (xA, yA), (xB, yB),
                          (0, 255, 0), 2)
    
    peopleCt.update(rect_people)
    print(peopleCt.nextObjectID)
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
# finally, close the window
cv2.destroyAllWindows()
cv2.waitKey(1)