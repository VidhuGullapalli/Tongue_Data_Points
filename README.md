# Tongue_Data_Points
Datapoints useful for ML on a tongue surface.
import cv2
import dlib
import numpy as np##arrays
import os
import bz2
import shutil
from imutils import face_utils

# Decompress shape predictor file if needed ---
compressed_file = '/Users/vidhu/Downloads/shape_predictor_68_face_landmarks.dat.bz2'
decompressed_file = '/Users/vidhu/Downloads/shape_predictor_68_face_landmarks.dat'

if not os.path.exists(decompressed_file):
    print("Decompressing shape predictor file...")
    with bz2.BZ2File(compressed_file) as fr, open(decompressed_file, 'wb') as fw:
        shutil.copyfileobj(fr, fw)
    print("Decompression complete.")

#  Load Dlib's face detector and landmark predictor ---
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(decompressed_file)

# : Initialize ORB keypoint detector ---
orb = cv2.ORB_create()

def detect_tongue_surface(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)
        mouth = shape[50:1000]

        top_lip = shape[62]
        bottom_lip = shape[66]
        mouth_open = np.linalg.norm(top_lip - bottom_lip) > 10  # adjust threshold as needed

        if not mouth_open:
            continue  # skip if mouth is closed

        # Compute the mouth region bounding box (refine to focus below lips)
        (x, y, w, h) = cv2.boundingRect(mouth)
        y_lower = y + int(h * 0.6)  # Focus on the lower part of the mouth region (below lips)
        mouth_roi = frame[y_lower:y + h, x:x + w]  # Refined region of interest

        #  Color-based segmentation for tongue surface detection ---
        hsv = cv2.cvtColor(mouth_roi, cv2.COLOR_BGR2HSV)

        # Define the refined range for tongue color (adjust for the tongue surface color)
        lower_tongue = np.array([0, 30, 50])  # Lower bound of the tongue color
        upper_tongue = np.array([15, 255, 255])  # Upper bound of the tongue color

        # Create a mask based on the color range (filtering out non-tongue regions)
        mask = cv2.inRange(hsv, lower_tongue, upper_tongue)

        # Apply the mask to the original mouth region
        result = cv2.bitwise_and(mouth_roi, mouth_roi, mask=mask)

        #  Edge detection for texture (surface of the tongue) ---
        edges = cv2.Canny(mask, 100, 200)  # Edge detection to find texture

        # Combine edge detection and mask to isolate tongue surface better
        combined = cv2.bitwise_and(edges, edges, mask=mask)

        #  Find the contour of the detected tongue surface ---
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find the largest contour, which should be the tongue surface
            largest_contour = max(contours, key=cv2.contourArea)

            # Draw the contour and a circle at the center of the tongue
            cv2.drawContours(frame, [largest_contour + (x, y_lower)], -1, (0, 255, 0), 2)

            # Calculate the center of the tongue surface and mark it
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.circle(frame, (cX + x, cY + y_lower), 5, (0, 0, 255), -1)
                cv2.putText(frame, "Tongue Surface", (cX + x + 10, cY + y_lower),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return frame

#  Capture from webcam ---
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
else:
    print("Webcam is ready.")

print("Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    frame = detect_tongue_surface(frame)
    cv2.imshow("Tongue Surface Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quitting...")
        break

cap.release()
cv2.destroyAllWindows()
