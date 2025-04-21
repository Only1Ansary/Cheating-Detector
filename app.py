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
eyes_closed_counter = 0
looking_away = False
eyes_closed = False

def eye_aspect_ratio(eye_landmarks):
    # Calculate Eye Aspect Ratio (EAR)
    A = distance.euclidean(eye_landmarks[1], eye_landmarks[5])
    B = distance.euclidean(eye_landmarks[2], eye_landmarks[4])
    C = distance.euclidean(eye_landmarks[0], eye_landmarks[3])
    return (A + B) / (2.0 * C)

def get_eye_position(eye_landmarks, img_w, img_h):
    # Calculate eye center and movement from neutral position
    eye_center = np.mean(eye_landmarks, axis=0)
    neutral_position = (img_w/2, img_h/2)  # Center of screen
    horizontal_diff = (eye_center[0] - neutral_position[0]) / neutral_position[0] * 100
    vertical_diff = (eye_center[1] - neutral_position[1]) / neutral_position[1] * 100
    return horizontal_diff, vertical_diff

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

            # Eye landmarks for gaze detection
            left_eye = np.array([(face_landmarks.landmark[i].x * img_w, face_landmarks.landmark[i].y * img_h) 
                                for i in [33, 160, 158, 133, 153, 144]], dtype="float32")
            right_eye = np.array([(face_landmarks.landmark[i].x * img_w, face_landmarks.landmark[i].y * img_h) 
                                 for i in [362, 385, 387, 263, 373, 380]], dtype="float32")

            # Calculate Eye Aspect Ratio
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0

            # Get eye movement from center
            left_eye_h, left_eye_v = get_eye_position(left_eye, img_w, img_h)
            right_eye_h, right_eye_v = get_eye_position(right_eye, img_w, img_h)
            avg_eye_h = (left_eye_h + right_eye_h) / 2
            avg_eye_v = (left_eye_v + right_eye_v) / 2

            head_moved = (abs(yaw) > MAX_HEAD_YAW ) or (abs(pitch) < MAX_HEAD_PITCH)
            
            if (head_moved) and not eyes_closed:
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