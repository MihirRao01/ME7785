import numpy as np
import cv2

# Global variables for HSV range
lower_hsv = None
upper_hsv = None
selected_hsv = None  

def click_event(event,x,y,flags,param):
    global selected_hsv
    if event == cv2.EVENT_LBUTTONDOWN:
        # get the hsv values of the pixel
        # hsv = cv2.cvtColor(event, cv2.COLOR_BGR2HSV)
        hsv_frame = param["hsv_frame"]
        selected_hsv = hsv_frame[y, x]
        print(f"Selected Pixel Coordinates: {x,y}")
        print(f"Clicked HSV: {selected_hsv}")

# driver function 
def main():
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('Detected Object')
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting...")
            break
                
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        cv2.setMouseCallback('Detected Object',click_event,{"hsv_frame":hsv_frame})
       
        
        if selected_hsv is not None:
            #define a range around the selected hsv value
           # Detect white objects
            if selected_hsv[1] < 50 and selected_hsv[2] > 200:  # Low Saturation, High Value
                lower_hsv = np.array([0, 0, 200])  # Focus on high brightness
                upper_hsv = np.array([179, 50, 255])  # Allow full Hue range, low Saturation
            else:
                # Use normal thresholds for colored objects
                tolerance = np.array([10, 50, 50])
                lower_hsv = np.maximum(selected_hsv - tolerance, [0, 0, 0])
                upper_hsv = np.minimum(selected_hsv + tolerance, [179, 255, 255])
            
            # Create a mask for the hsv color selected
            mask = cv2.inRange(hsv_frame, lower_hsv, upper_hsv)

            # Apply a bitwise AND to keep only the selected parts of the image
            cv2.bitwise_and(frame, frame, mask=mask)

            # Find contours in the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Draw bounding boxes around detected objects
            for contour in contours:
                # Filter small contours based on area
                if cv2.contourArea(contour) > 500:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, "Detected Object", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
            # Display the mask for debugging
            cv2.imshow('Mask', mask)

        # Display the original frame with bounding boxes
        cv2.imshow('Detected Object', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()
    
main()