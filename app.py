import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import distance

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Camera setup
cap = cv2.VideoCapture(0)
_, frame = cap.read()
img_h, img_w = frame.shape[:2]

# Camera matrix estimation
focal_length = img_w
camera_matrix = np.array([
    [focal_length, 0, img_w/2],
    [0, focal_length, img_h/2],
    [0, 0, 1]
], dtype="double")

dist_coeffs = np.zeros((4, 1))

# 3D model points for head pose estimation
model_points = np.array([
    (0.0, 0.0, 0.0),          # Nose tip
    (0.0, -330.0, -65.0),     # Chin
    (-225.0, 170.0, -135.0),  # Left eye
    (225.0, 170.0, -135.0),   # Right eye
    (-150.0, -150.0, -125.0), # Left mouth
    (150.0, -150.0, -125.0)   # Right mouth
])

MAX_HEAD_YAW = 63
MAX_HEAD_PITCH = 160

GAZE_CONSEC_FRAMES = 1

looking_away_counter = 0
looking_away = False


while cap.isOpened():
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)

    if not success:
        continue

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get 2D image points for head pose
            image_points = np.array([
                [face_landmarks.landmark[4].x * img_w, face_landmarks.landmark[4].y * img_h],   # Nose tip
                [face_landmarks.landmark[152].x * img_w, face_landmarks.landmark[152].y * img_h],  # Chin
                [face_landmarks.landmark[133].x * img_w, face_landmarks.landmark[133].y * img_h],  # Left eye
                [face_landmarks.landmark[362].x * img_w, face_landmarks.landmark[362].y * img_h],  # Right eye
                [face_landmarks.landmark[61].x * img_w, face_landmarks.landmark[61].y * img_h],   # Left mouth
                [face_landmarks.landmark[291].x * img_w, face_landmarks.landmark[291].y * img_h]   # Right mouth
            ], dtype="double")

            # Head pose estimation
            _, rotation_vec, _ = cv2.solvePnP(
                model_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            # Convert rotation vector to angles
            rmat, _ = cv2.Rodrigues(rotation_vec)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
            pitch, yaw, roll = angles

            head_moved = (abs(yaw) > MAX_HEAD_YAW ) or (abs(pitch) < MAX_HEAD_PITCH)
            
            if (head_moved):
                looking_away_counter += 1
                if looking_away_counter >= GAZE_CONSEC_FRAMES and not looking_away:
                    looking_away = True
            else:
                looking_away_counter = max(0, looking_away_counter - 1)
                if looking_away_counter == 0 and looking_away:
                    looking_away = False

            # Visualization
            status_color = (0, 255, 0) if not looking_away else (0, 0, 255)
            
            # Draw head pose info
            cv2.putText(frame, f"Head YAW: {yaw:.1f} (Max: {MAX_HEAD_YAW})", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 1)
            cv2.putText(frame, f"Head PITCH: {pitch:.1f} (Max: {MAX_HEAD_PITCH})", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 1)
            # Draw status
            cv2.putText(frame, "STATE: " + ("LOOKING AWAY!" if looking_away else "FOCUSED"), 
                        (img_w-250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    cv2.imshow('Precision Gaze Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()