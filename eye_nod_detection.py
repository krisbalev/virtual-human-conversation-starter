import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

class EyeNodDetector:
    """
    Enhanced webcam-based detector using MediaPipe FaceMesh to identify:
      1) Eye Contact (based on iris center).
      2) Head Nods (based on pitch changes).

    Improvements:
    - Consecutive frame requirement for stable eye contact.
    - Rolling buffer for pitch to detect nod "patterns."
    - Selects largest face if multiple faces are detected.
    - Smoother transitions and better reliability in busy environments.
    """

    def __init__(self,
                 camera_index=0,
                 # --- NOD DETECTION PARAMS ---
                 nod_pitch_diff_threshold=15.0,  # threshold in degrees for pitch difference
                 min_nod_frames=3,               # how many frames (in the buffer) must show a clear "dip" or "rise"
                 nod_cooldown=1.0,               # minimum seconds between detected nods

                 # --- EYE CONTACT PARAMS ---
                 eye_contact_frames_required=5,   # consecutive frames required to confirm eye contact
                 iris_center_threshold_ratio=0.25,# how close iris must be to the box center
                 max_yaw_for_gaze=50,            # ignore eye contact if yaw is too large

                 # --- SMOOTHING PARAMS ---
                 alpha=0.2,                      # smoothing factor for pitch
                 pitch_buffer_size=10,           # rolling buffer size for pitch

                 # --- MEDIAPIPE PARAMS ---
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):

        # Open the camera
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open webcam at index {camera_index}")

        # Store thresholds and parameters
        self.nod_pitch_diff_threshold = nod_pitch_diff_threshold
        self.min_nod_frames = min_nod_frames
        self.nod_cooldown = nod_cooldown
        self.eye_contact_frames_required = eye_contact_frames_required
        self.iris_center_threshold_ratio = iris_center_threshold_ratio
        self.max_yaw_for_gaze = max_yaw_for_gaze
        self.alpha = alpha
        self.pitch_buffer_size = pitch_buffer_size

        # MediaPipe FaceMesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=5,  # track up to 5 faces, then pick largest
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # 3D model for solvePnP (head pose)
        self.face_3d_model = np.array([
            [0.0,    0.0,    0.0],   # Nose tip
            [0.0,   -63.6,  -12.0],  # Chin
            [-34.0,  32.0,  -26.0],  # Left eye outer corner
            [34.0,   32.0,  -26.0],  # Right eye outer corner
            [-26.0, -28.0,  -22.0],  # Left mouth corner
            [26.0,  -28.0,  -22.0],  # Right mouth corner
        ], dtype=np.float64)

        # Corresponding indices in MediaPipe FaceMesh
        self.mp_indices = [4, 152, 33, 263, 61, 291]

        # Iris indices
        self.left_iris_indices  = [468, 469, 470, 471]
        self.right_iris_indices = [473, 474, 475, 476]

        # Eye contours (for bounding boxes)
        self.left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155,
                                 133, 246, 161, 160, 159, 158, 157, 173]
        self.right_eye_indices = [263, 249, 390, 373, 374, 380, 381, 382,
                                  362, 466, 388, 387, 386, 385, 384, 398]

        # Internal state for nod detection
        self.last_nod_time = 0.0
        self.smoothed_pitch = None
        self.pitch_buffer = deque(maxlen=self.pitch_buffer_size)

        # Internal state for eye contact
        self.consecutive_eye_contact_frames = 0

    def get_cues_and_show(self):
        """
        Main processing function:
        1) Captures a frame from camera.
        2) Detects face + iris + head pose with MediaPipe.
        3) Determines if a nod occurred and if eye contact is present.
        4) Displays annotated frame.
        5) Returns (nod_detected, eye_contact, should_quit).
        """
        ret, frame = self.cap.read()
        if not ret or frame is None:
            print("Warning: Could not read from webcam. Retrying...")
            return False, False, False

        frame = cv2.flip(frame, 1)     # mirror image
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run face mesh
        try:
            results = self.face_mesh.process(rgb_frame)
        except Exception as e:
            print(f"FaceMesh processing error: {e}")
            return False, False, False

        nod_detected = False
        eye_contact = False

        # -- If multiple faces, pick the largest bounding box --
        if results.multi_face_landmarks and len(results.multi_face_landmarks) > 0:
            face_landmarks_list = results.multi_face_landmarks
            face_rects = []
            for idx, landmarks in enumerate(face_landmarks_list):
                x_coords = [lm.x for lm in landmarks.landmark]
                y_coords = [lm.y for lm in landmarks.landmark]
                # Convert from normalized [0..1] to actual pixels
                min_x, max_x = min(x_coords)*w, max(x_coords)*w
                min_y, max_y = min(y_coords)*h, max(y_coords)*h
                bbox_area = (max_x - min_x) * (max_y - min_y)
                face_rects.append((idx, bbox_area))

            # Sort by area desc, pick largest
            face_rects.sort(key=lambda x: x[1], reverse=True)
            best_face_index = face_rects[0][0]

            # Extract best face landmarks
            face_landmarks = face_landmarks_list[best_face_index]
        else:
            # No face found
            self._draw_overlays(frame, "No Face Detected", "N/A", "Nods: 0")
            return False, False, self._check_quit()

        # ------- HEAD POSE with solvePnP -------
        pitch, yaw, roll = self._compute_head_pose(face_landmarks, w, h)

        # -- Update pitch smoothing / buffer --
        if pitch is not None:
            if self.smoothed_pitch is None:
                self.smoothed_pitch = pitch
            else:
                self.smoothed_pitch = (
                    self.alpha * pitch + (1 - self.alpha) * self.smoothed_pitch
                )
            self.pitch_buffer.append(self.smoothed_pitch)

        # ------- Nod Detection -------
        if pitch is not None and len(self.pitch_buffer) == self.pitch_buffer.maxlen:
            # Basic pattern check: difference between min and max in buffer
            pitch_min = min(self.pitch_buffer)
            pitch_max = max(self.pitch_buffer)
            pitch_diff = abs(pitch_max - pitch_min)

            current_time = time.time()
            if (pitch_diff >= self.nod_pitch_diff_threshold and
                current_time - self.last_nod_time > self.nod_cooldown):
                
                # Optional extra check: ensure there's a "dip" or "rise" pattern
                # Simplistic approach: last half vs first half
                half = self.pitch_buffer_size // 2
                first_half_avg = np.mean(list(self.pitch_buffer)[:half])
                second_half_avg = np.mean(list(self.pitch_buffer)[half:])
                # If there's a reversal in direction, we consider it a nod
                if ( (second_half_avg - first_half_avg) * (pitch_max - pitch_min) < 0 ):
                    # This implies the pitch went up then down or down then up
                    # But you can simplify by ignoring direction if you just want
                    # one big pitch movement:
                    pass

                # We found a big enough pitch difference => It's a nod
                nod_detected = True
                self.last_nod_time = current_time

        # ------- Eye Contact Detection -------
        # We'll do the usual check (iris center + yaw < threshold).
        eyes_forward = self._check_iris_gaze(face_landmarks, w, h)
        # If yaw is too large, no eye contact
        if yaw is not None and abs(yaw) > self.max_yaw_for_gaze:
            eyes_forward = False

        if eyes_forward and pitch is not None:
            # If we have stable detection
            self.consecutive_eye_contact_frames += 1
            if self.consecutive_eye_contact_frames >= self.eye_contact_frames_required:
                eye_contact = True
        else:
            self.consecutive_eye_contact_frames = 0

        # ------- Prepare overlay strings -------
        pitch_yaw_roll_text = ""
        if self.smoothed_pitch is not None and yaw is not None and roll is not None:
            pitch_yaw_roll_text = (
                f"Pitch: {self.smoothed_pitch:.2f}, "
                f"Yaw: {yaw:.2f}, "
                f"Roll: {roll:.2f}"
            )
        eye_contact_text = "Eye Contact" if eye_contact else "No Eye Contact"
        nod_text = "Nod DETECTED" if nod_detected else "No Nod"

        # ------- Draw Face Mesh on the frame -------
        self._draw_face_mesh(frame, face_landmarks)
        self._draw_iris_overlays(frame, face_landmarks, w, h, eye_contact)
        self._draw_overlays(frame, eye_contact_text, pitch_yaw_roll_text, nod_text)

        # Show final annotated frame
        cv2.imshow("Iris & Head Nod Detection", frame)
        should_quit = self._check_quit()
        return nod_detected, eye_contact, should_quit

    def _compute_head_pose(self, face_landmarks, w, h):
        """
        Given landmarks for the best face, compute pitch, yaw, roll via solvePnP.
        Returns (pitch, yaw, roll) in degrees or (None, None, None) on failure.
        """
        # 2D landmarks for solvePnP
        landmarks_2d = []
        for idx in self.mp_indices:
            lm = face_landmarks.landmark[idx]
            x, y = int(lm.x * w), int(lm.y * h)
            landmarks_2d.append([x, y])
        landmarks_2d = np.array(landmarks_2d, dtype=np.float64)

        # solvePnP requires a camera matrix
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0,            center[0]],
            [0,            focal_length, center[1]],
            [0,            0,            1]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        try:
            success, rotation_vec, translation_vec = cv2.solvePnP(
                self.face_3d_model,
                landmarks_2d,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
        except Exception as e:
            print(f"solvePnP error: {e}")
            return None, None, None

        if not success:
            return None, None, None

        rot_matrix, _ = cv2.Rodrigues(rotation_vec)
        proj_matrix = np.hstack((rot_matrix, translation_vec))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)
        pitch, yaw, roll = euler_angles.flatten()
        return pitch, yaw, roll

    def _check_iris_gaze(self, face_landmarks, w, h):
        """
        Returns True if both eyes' iris centers are within a certain threshold
        of the bounding box center.
        """
        left_iris_center  = self._mean_xy(face_landmarks, self.left_iris_indices, w, h)
        right_iris_center = self._mean_xy(face_landmarks, self.right_iris_indices, w, h)

        # bounding boxes for each eye
        lx, ly, lw_, lh_ = self._bounding_box(face_landmarks, self.left_eye_indices, w, h)
        rx, ry, rw_, rh_ = self._bounding_box(face_landmarks, self.right_eye_indices, w, h)

        left_gaze  = self._eye_gaze_check(left_iris_center, (lx, ly, lw_, lh_))
        right_gaze = self._eye_gaze_check(right_iris_center, (rx, ry, rw_, rh_))

        return left_gaze and right_gaze

    def _mean_xy(self, face_landmarks, indices, w, h):
        """
        Compute the average (x,y) in pixel coordinates
        for the given set of landmark indices.
        """
        pts = [(face_landmarks.landmark[i].x * w,
                face_landmarks.landmark[i].y * h)
               for i in indices]
        return np.mean(pts, axis=0).astype(int)

    def _bounding_box(self, face_landmarks, indices, w, h):
        coords = [(face_landmarks.landmark[i].x * w,
                   face_landmarks.landmark[i].y * h)
                  for i in indices]
        coords = np.array(coords, dtype=np.int32)
        x_, y_, ww_, hh_ = cv2.boundingRect(coords)
        return x_, y_, ww_, hh_

    def _eye_gaze_check(self, iris_center, box):
        """
        Check if iris_center is within a threshold distance from the box center.
        """
        (box_x, box_y, box_w, box_h) = box
        center_x = box_x + box_w // 2
        center_y = box_y + box_h // 2

        dx = iris_center[0] - center_x
        dy = iris_center[1] - center_y
        dist = np.sqrt(dx*dx + dy*dy)
        threshold = self.iris_center_threshold_ratio * box_w

        return dist < threshold

    def _draw_face_mesh(self, frame, face_landmarks):
        """
        Draw FaceMesh tessellation on the frame for visualization.
        """
        self.mp_drawing.draw_landmarks(
            frame,
            face_landmarks,
            self.mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
        )

    def _draw_iris_overlays(self, frame, face_landmarks, w, h, eye_contact):
        """
        Draw iris centers and eye bounding boxes on the frame.
        """
        # If "eye_contact" is true, color them green; else yellow for visibility
        iris_color = (0, 255, 0) if eye_contact else (0, 255, 255)

        left_iris_center  = self._mean_xy(face_landmarks, self.left_iris_indices, w, h)
        right_iris_center = self._mean_xy(face_landmarks, self.right_iris_indices, w, h)

        cv2.circle(frame, tuple(left_iris_center),  3, iris_color, -1)
        cv2.circle(frame, tuple(right_iris_center), 3, iris_color, -1)

        # Draw bounding boxes for each eye
        lx, ly, lw_, lh_ = self._bounding_box(face_landmarks, self.left_eye_indices, w, h)
        rx, ry, rw_, rh_ = self._bounding_box(face_landmarks, self.right_eye_indices, w, h)
        cv2.rectangle(frame, (lx, ly), (lx+lw_, ly+lh_), (255, 0, 0), 1)
        cv2.rectangle(frame, (rx, ry), (rx+rw_, ry+rh_), (255, 0, 0), 1)

    def _draw_overlays(self, frame, eye_contact_text, pitch_yaw_roll_text, nod_text):
        """
        Simple utility to draw text overlays onto the frame.
        """
        cv2.putText(frame, pitch_yaw_roll_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, eye_contact_text, (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, nod_text, (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    def _check_quit(self):
        """
        Checks if ESC key was pressed to signal termination.
        """
        key = cv2.waitKey(1) & 0xFF
        return (key == 27)  # ESC

    def release(self):
        """
        Safely release camera and close window.
        """
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
