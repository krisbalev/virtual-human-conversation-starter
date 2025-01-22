import cv2
import mediapipe as mp
import numpy as np
import time

class EyeNodDetector:
    """
    Opens a webcam feed (via OpenCV), processes frames with MediaPipe FaceMesh
    to detect face landmarks, draws overlays, and returns booleans about nod detection
    and eye contact.
    """
    def __init__(self,
                 camera_index=0,
                 nod_threshold=15.0,
                 cooldown=1.0,
                 alpha=0.3,
                 iris_center_threshold_ratio=0.25,
                 max_yaw_for_gaze=50):
        # Open the camera
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open webcam")

        # Store thresholds and parameters
        self.nod_threshold = nod_threshold
        self.cooldown = cooldown
        self.alpha = alpha
        self.iris_center_threshold_ratio = iris_center_threshold_ratio
        self.max_yaw_for_gaze = max_yaw_for_gaze

        # Init MediaPipe FaceMesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,  # for iris detection
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # For head pose estimation via solvePnP:
        self.face_3d_model = np.array([
            [0.0,    0.0,    0.0],     # Nose tip
            [0.0,   -63.6,  -12.0],    # Chin
            [-34.0,  32.0,  -26.0],    # Left eye outer corner
            [34.0,   32.0,  -26.0],    # Right eye outer corner
            [-26.0, -28.0,  -22.0],    # Left mouth corner
            [26.0,  -28.0,  -22.0],    # Right mouth corner
        ], dtype=np.float64)

        # Corresponding MediaPipe FaceMesh indices:
        self.mp_indices = [4, 152, 33, 263, 61, 291]

        # Iris
        self.left_iris_indices  = [468, 469, 470, 471]
        self.right_iris_indices = [473, 474, 475, 476]

        # Eye contours
        self.left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155,
                                 133, 246, 161, 160, 159, 158, 157, 173]
        self.right_eye_indices = [263, 249, 390, 373, 374, 380, 381, 382,
                                  362, 466, 388, 387, 386, 385, 384, 398]

        # Head nod variables
        self.previous_pitch = None
        self.last_nod_time = 0.0
        self.smoothed_pitch = None
        self.nod_count = 0  # Just for display

    def get_cues_and_show(self):
        """
        1) Reads a frame from the webcam
        2) Does face/iris detection
        3) Calculates nod detection & eye contact
        4) Draws results on the frame
        5) Shows the frame in a window
        6) Returns (nod_detected, eye_contact, should_quit)
        """
        ret, frame = self.cap.read()
        if not ret:
            print("Could not read from webcam.")
            return False, False, True

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        nod_detected = False
        eye_contact = False

        pitch_yaw_roll_text = ""
        eye_contact_text = "No Face Detected"
        nod_text = f"Nods: {self.nod_count}"

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]

            # ---------- Head Pose with solvePnP ----------
            landmarks_2d = []
            for idx in self.mp_indices:
                lm = face_landmarks.landmark[idx]
                x, y = int(lm.x * w), int(lm.y * h)
                landmarks_2d.append([x, y])
            landmarks_2d = np.array(landmarks_2d, dtype=np.float64)

            focal_length = w
            center = (w / 2, h / 2)
            camera_matrix = np.array([
                [focal_length, 0,            center[0]],
                [0,            focal_length, center[1]],
                [0,            0,            1]
            ], dtype=np.float64)
            dist_coeffs = np.zeros((4, 1), dtype=np.float64)

            success, rotation_vec, translation_vec = cv2.solvePnP(
                self.face_3d_model,
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
                if self.smoothed_pitch is None:
                    self.smoothed_pitch = pitch
                else:
                    self.smoothed_pitch = (
                        self.alpha * pitch +
                        (1 - self.alpha) * self.smoothed_pitch
                    )

                pitch_yaw_roll_text = (f"Pitch: {self.smoothed_pitch:.2f}, "
                                       f"Yaw: {yaw:.2f}, "
                                       f"Roll: {roll:.2f}")

            # ---------- Nod Detection ----------
            if self.previous_pitch is not None and self.smoothed_pitch is not None:
                delta_pitch = self.smoothed_pitch - self.previous_pitch
                if abs(delta_pitch) > self.nod_threshold:
                    now = time.time()
                    if now - self.last_nod_time > self.cooldown:
                        self.nod_count += 1
                        nod_detected = True
                        self.last_nod_time = now
            self.previous_pitch = self.smoothed_pitch

            # ---------- Iris-based Eye Contact ----------
            def mean_xy(lmarks):
                pts = [(lm.x * w, lm.y * h) for lm in lmarks]
                return np.mean(pts, axis=0).astype(int)

            left_iris_pts = [face_landmarks.landmark[i] for i in self.left_iris_indices]
            right_iris_pts = [face_landmarks.landmark[i] for i in self.right_iris_indices]
            left_iris_center  = mean_xy(left_iris_pts)
            right_iris_center = mean_xy(right_iris_pts)

            def bounding_box(indices):
                coords = [(face_landmarks.landmark[i].x * w,
                           face_landmarks.landmark[i].y * h)
                          for i in indices]
                coords = np.array(coords, dtype=np.int32)
                x_, y_, ww_, hh_ = cv2.boundingRect(coords)
                return (x_, y_, ww_, hh_)

            lx, ly, lw_, lh_ = bounding_box(self.left_eye_indices)
            rx, ry, rw_, rh_ = bounding_box(self.right_eye_indices)

            def eye_gaze_check(eye_x, eye_y, box_x, box_y, box_w, box_h):
                center_x = box_x + box_w // 2
                center_y = box_y + box_h // 2
                dx = eye_x - center_x
                dy = eye_y - center_y
                dist = np.sqrt(dx**2 + dy**2)
                threshold = self.iris_center_threshold_ratio * box_w
                return dist < threshold

            left_gaze = eye_gaze_check(left_iris_center[0], left_iris_center[1], lx, ly, lw_, lh_)
            right_gaze = eye_gaze_check(right_iris_center[0], right_iris_center[1], rx, ry, rw_, rh_)
            eyes_forward = left_gaze and right_gaze

            # Eye contact means: face is found, yaw < threshold, and eyes forward
            if success and abs(yaw) < self.max_yaw_for_gaze and eyes_forward:
                eye_contact = True
                eye_contact_text = "Eye Contact"
            else:
                eye_contact_text = "No Eye Contact"
        else:
            eye_contact_text = "No Face Detected"

        # ---------- Drawing Overlays ----------
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            self.mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                self.mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )

            # Draw iris centers if we found them
            if eye_contact_text != "No Face Detected":
                color = (0, 255, 0) if eye_contact else (0, 255, 255)
                cv2.circle(frame, tuple(left_iris_center), 3, color, -1)
                cv2.circle(frame, tuple(right_iris_center), 3, color, -1)

                # Eye bounding boxes
                cv2.rectangle(frame, (lx, ly), (lx + lw_, ly + lh_), (255, 0, 0), 1)
                cv2.rectangle(frame, (rx, ry), (rx + rw_, ry + rh_), (255, 0, 0), 1)

        cv2.putText(frame, pitch_yaw_roll_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, eye_contact_text, (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, nod_text, (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Iris Gaze & Head Nod Detection", frame)

        # Press ESC to quit
        key = cv2.waitKey(1) & 0xFF
        should_quit = (key == 27)

        return nod_detected, eye_contact, should_quit

    def release(self):
        """ Safely release camera and close window """
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
