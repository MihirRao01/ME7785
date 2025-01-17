import numpy as np
import cv2

# Initialize video capture
cap = cv2.VideoCapture(0)

# Refined HSV color ranges for detecting pure red
lower_red_1 = np.array([0, 150, 150])   # First range for pure red (lower hue)
upper_red_1 = np.array([5, 255, 255])  # Upper bound for first range
lower_red_2 = np.array([170, 150, 150]) # Second range for pure red (upper hue)
upper_red_2 = np.array([180, 255, 255]) # Upper bound for second range

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Exiting...")
        break

    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create two masks for red (due to wraparound in HSV)
    mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)

    # Combine the masks
    mask = cv2.bitwise_or(mask1, mask2)

    # Apply a bitwise AND to keep only the red parts of the image
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes around detected objects
    for contour in contours:
        # Filter small contours based on area
        if cv2.contourArea(contour) > 500:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Red Object", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the original frame with bounding boxes
    cv2.imshow('Detected Red Object', frame)

    # Display the mask for debugging
    cv2.imshow('Mask', mask)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
