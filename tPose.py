import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def is_t_pose(landmarks, image_width, image_height):
    # Get coordinates
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
    right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

    # Convert normalized landmarks to pixel coordinates
    def to_pixel(landmark):
        return int(landmark.x * image_width), int(landmark.y * image_height)

    ls = to_pixel(left_shoulder)
    rs = to_pixel(right_shoulder)
    le = to_pixel(left_elbow)
    re = to_pixel(right_elbow)
    lw = to_pixel(left_wrist)
    rw = to_pixel(right_wrist)

    # Check horizontal alignment for T-pose (y-values roughly same)
    tolerance = 30
    is_left_arm_horizontal = abs(ls[1] - le[1]) < tolerance and abs(le[1] - lw[1]) < tolerance
    is_right_arm_horizontal = abs(rs[1] - re[1]) < tolerance and abs(re[1] - rw[1]) < tolerance

    return is_left_arm_horizontal and is_right_arm_horizontal

# Start webcam
cap = cv2.VideoCapture(0)
capture_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        if is_t_pose(results.pose_landmarks.landmark, frame.shape[1], frame.shape[0]):
            cv2.putText(frame, "T-Pose detected! Press SPACE to capture.", (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Stand in T-Pose to capture.", (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2)

    cv2.imshow("T-Pose Capture", frame)

    key = cv2.waitKey(10)
    if key == 27:  # ESC to quit
        break
    elif key == 32:  # Spacebar
        if results.pose_landmarks and is_t_pose(results.pose_landmarks.landmark, frame.shape[1], frame.shape[0]):
            filename = f"tpose_capture_{int(time.time())}.png"
            cv2.imwrite(filename, frame)
            print(f"[✓] T-Pose image saved as {filename}")
        else:
            print("[!] T-Pose not detected. Try again.")

cap.release()
cv2.destroyAllWindows()
