import cv2
import mediapipe as mp
import ffmpeg
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def process_video(input_path, output_path, marker_hex, line_hex, size_config, selected_joints, 
                  min_detection_confidence=0.85, min_tracking_confidence=0.85, visibility_threshold=0.15):
    """ 選択された関節のみを解析し、精度を向上させたモーションキャプチャ処理 """
    
    cap = cv2.VideoCapture(input_path)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(output_path, fourcc, 30, (frame_width, frame_height))

    marker_color = tuple(int(marker_hex.lstrip("#")[i:i+2], 16) for i in (4, 2, 0))
    line_color = tuple(int(line_hex.lstrip("#")[i:i+2], 16) for i in (4, 2, 0))

    marker_thickness = size_config["thickness"]
    marker_radius = size_config["circle_radius"]

    # **選択された関節のみ解析**
    joints_map = {
        "首": [mp_pose.PoseLandmark.NOSE],
        "上肢": [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
                 mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.RIGHT_ELBOW,
                 mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST],
        "下肢": [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP,
                 mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.RIGHT_KNEE,
                 mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_ANKLE],
        "体幹": [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP,
                 mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER]
    }

    selected_landmarks = []
    for key, enabled in selected_joints.items():
        if enabled:
            selected_landmarks.extend(joints_map[key])

    with mp_pose.Pose(static_image_mode=False, 
                      min_detection_confidence=min_detection_confidence, 
                      min_tracking_confidence=min_tracking_confidence) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                for lm_idx in selected_landmarks:
                    landmark = results.pose_landmarks.landmark[lm_idx]
                    h, w, _ = frame.shape
                    x, y = int(landmark.x * w), int(landmark.y * h)

                    if landmark.visibility < visibility_threshold:
                        continue

                    cv2.circle(frame, (x, y), marker_radius, marker_color, -1)

                for connection in mp_pose.POSE_CONNECTIONS:
                    if connection[0] in selected_landmarks and connection[1] in selected_landmarks:
                        cv2.line(frame, 
                                 (int(results.pose_landmarks.landmark[connection[0]].x * w), 
                                  int(results.pose_landmarks.landmark[connection[0]].y * h)),
                                 (int(results.pose_landmarks.landmark[connection[1]].x * w), 
                                  int(results.pose_landmarks.landmark[connection[1]].y * h)),
                                 line_color, marker_thickness)

            out.write(frame)

    cap.release()
    out.release()

