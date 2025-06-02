import cv2
import dlib
import numpy as np
from scipy.spatial import distance
import winsound  # For beep sound (Windows)

# Calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Thresholds and constants
EAR_THRESHOLD = 0.25  # Threshold for EAR to detect blinking or drowsiness
CONSECUTIVE_FRAMES = 20  # Number of consecutive frames below threshold to trigger alert

# Initialize counters
frame_counter = 0

# Load pre-trained dlib face detector and facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Indices for eyes in facial landmarks
LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))

# Start video capture
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Mirror the frame horizontally
    frame = cv2.flip(frame, 1)

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray_frame)

    if len(faces) == 0:
        cv2.putText(frame, "No Face Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        for face in faces:
            # Get facial landmarks
            landmarks = predictor(gray_frame, face)
            landmarks_points = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(68)]

            # Extract left and right eye landmarks
            left_eye = [landmarks_points[i] for i in LEFT_EYE]
            right_eye = [landmarks_points[i] for i in RIGHT_EYE]

            # Convert to NumPy array
            left_eye_np = np.array(left_eye, dtype=np.int32)
            right_eye_np = np.array(right_eye, dtype=np.int32)

            # Calculate EAR for both eyes
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0

            # Draw eye contours using convexHull
            cv2.polylines(frame, [cv2.convexHull(left_eye_np)], True, (0, 255, 0), 1)
            cv2.polylines(frame, [cv2.convexHull(right_eye_np)], True, (0, 255, 0), 1)

            # Check if EAR is below threshold
            if avg_ear < EAR_THRESHOLD:
                frame_counter += 1
                if frame_counter >= CONSECUTIVE_FRAMES:
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    # Trigger continuous beep sound
                    winsound.Beep(1000, 500)  # Frequency = 1000 Hz, Duration = 500 ms
            else:
                frame_counter = 0

    # Display the frame
    cv2.imshow("Driver Drowsiness Detection", frame)

    # Break loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
video_capture.release()
cv2.destroyAllWindows()