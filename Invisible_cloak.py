import cv2
import numpy as np

cap = cv2.VideoCapture(0)
back = cv2.imread('image.jpeg')

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        red = np.uint8([[[0,0,255]]])
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        l_red = np.array([0,100,100])
        u_red = np.array([10,255,255])

        mask = cv2.inRange(hsv, l_red, u_red)
        #cv2.imshow("mask",mask)

        part1 = cv2.bitwise_and(back, back, mask=mask)
        #cv2.imshow("part1", part1)

        mask = cv2.bitwise_not(mask)
        
        part2 = cv2.bitwise_and(frame, frame, mask=mask)
        #cv2.imshow("part2",part2)

        kernel = np.ones((5,5),np.uint8)
        
        #cv2.imshow("cloak",part1+part2)
        opening = cv2.morphologyEx(part1+part2, cv2.MORPH_OPEN, kernel)
        cv2.imshow("cloak",opening)
        if cv2.waitKey(5) == ord('q'):
            cv2.imwrite("image.jpeg", back)
            break
        
cap.release()
cv2.destroyAllWindows()
