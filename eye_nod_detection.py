import cv2
import mediapipe as mp
import numpy as np
import math
import time

# --------------------- MediaPipe Setup ---------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,  # Enables Iris landmarks
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Drawing utils (optional, for debug visualization)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# --------------------- 3D Model for solvePnP ---------------------
# We'll use 6 canonical landmarks for head pose estimation (in millimeters).
face_3d_model = np.array([
    [0.0,    0.0,    0.0],     # Nose tip
    [0.0,   -63.6,  -12.0],    # Chin
    [-34.0,  32.0,  -26.0],    # Left eye outer corner
    [34.0,   32.0,  -26.0],    # Right eye outer corner
    [-26.0, -28.0,  -22.0],    # Left mouth corner
    [26.0,  -28.0,  -22.0],    # Right mouth corner
], dtype=np.float64)

# Corresponding Face Mesh indices for those 6 points:
# (Note: These are approximate; check official MediaPipe FaceMesh diagrams for your version.)
mp_indices = [4, 152, 33, 263, 61, 291]

# For IRIS: 4 main landmarks for each iris (5th can be index 472 or 477, but often 4 are sufficient).
left_iris_indices  = [468, 469, 470, 471]
right_iris_indices = [473, 474, 475, 476]

# Eye contour indices (optional, to draw eye outlines and define bounding box).
# These sets are from the official MediaPipe docs, but you can adapt as needed.
left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 
                    133, 246, 161, 160, 159, 158, 157, 173]
right_eye_indices = [263, 249, 390, 373, 374, 380, 381, 382,
                     362, 466, 388, 387, 386, 385, 384, 398]

# --------------------- Webcam & Detection Vars ---------------------
cap = cv2.VideoCapture(0)  # Change to 1 or 2 if you have multiple cams

# Head Nod Detection Variables
previous_pitch = None
nod_count = 0
nod_threshold = 15.0     # degrees change required to consider a nod
last_nod_time = 0.0
cooldown = 1.0           # 1 second cooldown between nod counts

# Optional smoothing factor for pitch
alpha = 0.3  # 0.0 = no smoothing, 1.0 = heavy smoothing
smoothed_pitch = None

# Iris-based Gaze thresholds
# We'll say "looking forward" if the iris center is within 0.25 of the bounding box width from center.
iris_center_threshold_ratio = 0.25

# We'll also define a loose head orientation threshold for skipping if user is turned sideways
max_yaw_for_gaze = 50  # if yaw is beyond ±50°, we won't even bother with "eye contact"

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera read failed. Exiting.")
        break

    # Flip horizontally for a selfie-like view
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    # Text placeholders
    pitch_yaw_roll_text = ""
    eye_contact_text = "No Face Detected"
    nod_text = f"Nods: {nod_count}"

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]

        # --------------------- (A) Head Pose with solvePnP ---------------------
        # Gather the 2D coordinates for the 6 canonical landmarks
        landmarks_2d = []
        for idx in mp_indices:
            lm = face_landmarks.landmark[idx]
            x, y = int(lm.x * w), int(lm.y * h)
            landmarks_2d.append([x, y])
        landmarks_2d = np.array(landmarks_2d, dtype=np.float64)

        # Construct the camera matrix
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0,            center[0]],
            [0,            focal_length, center[1]],
            [0,            0,            1        ]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)  # no distortion

        # Solve for the head pose
        success, rotation_vec, translation_vec = cv2.solvePnP(
            face_3d_model,
            landmarks_2d,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        pitch, yaw, roll = 0.0, 0.0, 0.0
        if success:
            rot_matrix, _ = cv2.Rodrigues(rotation_vec)
            proj_matrix = np.hstack((rot_matrix, translation_vec))
            _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)
            pitch, yaw, roll = euler_angles.flatten()

            # Smooth the pitch if desired
            if smoothed_pitch is None:
                smoothed_pitch = pitch
            else:
                smoothed_pitch = alpha * pitch + (1 - alpha) * smoothed_pitch

            pitch_yaw_roll_text = (f"Pitch: {smoothed_pitch:.2f}, "
                                   f"Yaw: {yaw:.2f}, "
                                   f"Roll: {roll:.2f}")

        # --------------------- (B) Head Nod Detection ---------------------
        if previous_pitch is not None:
            delta_pitch = smoothed_pitch - previous_pitch
            if abs(delta_pitch) > nod_threshold:
                now = time.time()
                if now - last_nod_time > cooldown:
                    nod_count += 1
                    last_nod_time = now
        previous_pitch = smoothed_pitch

        # --------------------- (C) Iris-based Gaze ---------------------
        # Extract left and right iris landmarks
        left_iris_points = [face_landmarks.landmark[i] for i in left_iris_indices]
        right_iris_points = [face_landmarks.landmark[i] for i in right_iris_indices]

        # Compute average (x,y) for left/right iris
        def mean_xy(lmarks):
            pts = [(lm.x * w, lm.y * h) for lm in lmarks]
            return np.mean(pts, axis=0).astype(int)

        left_iris_center  = mean_xy(left_iris_points)
        right_iris_center = mean_xy(right_iris_points)

        # (Optional) Eye bounding boxes
        def bounding_box(indices):
            coords = [(face_landmarks.landmark[i].x * w,
                       face_landmarks.landmark[i].y * h)
                      for i in indices]
            coords = np.array(coords, dtype=np.int32)
            x_, y_, ww_, hh_ = cv2.boundingRect(coords)
            return (x_, y_, ww_, hh_)

        # Left eye bounding box
        lx, ly, lw_, lh_ = bounding_box(left_eye_indices)
        # Right eye bounding box
        rx, ry, rw_, rh_ = bounding_box(right_eye_indices)

        # For each eye, measure distance from iris center to bounding box center
        def eye_gaze_check(eye_x, eye_y, box_x, box_y, box_w, box_h):
            center_x = box_x + box_w // 2
            center_y = box_y + box_h // 2
            dx = eye_x - center_x
            dy = eye_y - center_y
            dist = np.sqrt(dx**2 + dy**2)
            # if dist < some fraction of box width, we consider it "looking forward"
            threshold = iris_center_threshold_ratio * box_w
            return dist < threshold

        left_gaze = eye_gaze_check(left_iris_center[0], left_iris_center[1], lx, ly, lw_, lh_)
        right_gaze = eye_gaze_check(right_iris_center[0], right_iris_center[1], rx, ry, rw_, rh_)

        # We'll say "eyes forward" if BOTH eyes meet the threshold
        eyes_forward = left_gaze and right_gaze

        # -------------- Combine Head + Eye Logic for "Eye Contact" --------------
        # Let's define some rules:
        # 1) The user's face is detected
        # 2) The user's yaw isn't extremely large (i.e. not turned far away)
        # 3) The iris is near center in both eyes
        # If you want a pitch check too, you can do abs(pitch) < some_thresh.

        # For demonstration, let's just do a yaw-based filter and then check eyes:
        if success:  # face found
            if abs(yaw) < max_yaw_for_gaze and eyes_forward:
                eye_contact_text = "Eye Contact"
            else:
                eye_contact_text = "No Eye Contact"
        else:
            eye_contact_text = "No Eye Contact"

        # (Optional) Visualization: Draw circles for iris centers, bounding boxes, etc.
        cv2.circle(frame, tuple(left_iris_center), 3, (0, 255, 0), -1)
        cv2.circle(frame, tuple(right_iris_center), 3, (0, 255, 0), -1)

        # Draw bounding boxes for eyes (optional)
        cv2.rectangle(frame, (lx, ly), (lx + lw_, ly + lh_), (255, 0, 0), 1)
        cv2.rectangle(frame, (rx, ry), (rx + rw_, ry + rh_), (255, 0, 0), 1)

        # (Optional) Draw face mesh
        mp_drawing.draw_landmarks(
            frame,
            face_landmarks,
            mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
        )

    # --------------------- Render Text ---------------------
    cv2.putText(frame, pitch_yaw_roll_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, eye_contact_text, (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, nod_text, (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Iris Gaze & Head Nod Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Esc to exit
        break

cap.release()
cv2.destroyAllWindows()
