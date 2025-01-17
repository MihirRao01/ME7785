import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #apply GaussianBlur to the frame to reduce noise and imporove ege detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detect edges using Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find contours from the edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw bounding boxes around detected objects
    for contour in contours:
        # Filter small contours based on area
        if cv2.contourArea(contour) > 500:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


    # Display the resulting frame
    cv2.imshow('frame',frame)
    cv2.imshow('blurred',blurred)
    cv2.imshow('edges',edges)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()