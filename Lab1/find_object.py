import numpy as np
import cv2

# Global variable for HSV range
lower_hsv = None
upper_hsv = None
selected_hsv = None  

def click_event(event, x, y, flags, param):
    global selected_hsv
    if event == cv2.EVENT_LBUTTONDOWN:
        # Get the HSV value of the clicked pixel
        hsv_frame = param["hsv_frame"]
        selected_hsv = hsv_frame[y, x]
        # print(f"Selected Pixel Coordinates: {x,y}")
        print(f"Clicked HSV: {selected_hsv}")

# Driver function 
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

        cv2.setMouseCallback('Detected Object', click_event, {"hsv_frame": hsv_frame})

        if selected_hsv is not None:
            # Extract the Value (brightness) component from the selected HSV value
            selected_value = selected_hsv[2]

            # Determine if the object is dark or bright
            if selected_value < 100:  # Dark object (like black phone)
                tolerance = np.array([10, 50, 100])  # Lower tolerance for dark objects
            else:  # Bright object (like tennis ball)
                tolerance = np.array([10, 50, 50])  # Higher tolerance for bright objects
            
            # Define HSV range with tolerance
            lower_hsv = np.maximum(selected_hsv - tolerance, [0, 0, 0])
            upper_hsv = np.minimum(selected_hsv + tolerance, [179, 255, 255])
            
            # Create a mask for the HSV color selected
            mask = cv2.inRange(hsv_frame, lower_hsv, upper_hsv)

            # Apply the mask to the frame (keep only the selected object)
            result = cv2.bitwise_and(frame, frame, mask=mask)

            # Find contours in the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Filter out small objects by contour area
            min_contour_area =750  # Minimum area of contour to be considered as an object
            filtered_contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]

            # If no contours are found, we skip the drawing part
            if len(filtered_contours) > 0:
                # Sort contours based on area to pick the largest one (potentially the clicked object)
                largest_contour = max(filtered_contours, key=cv2.contourArea)

                # Get bounding box around the largest contour
                x, y, w, h = cv2.boundingRect(largest_contour)

                # Draw bounding box around the detected object
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Detected Object", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                print(f"Center Pixel Coordinates: {(x+w)/2,(y+h)/2}")

            # Display the mask for debugging purposes
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
