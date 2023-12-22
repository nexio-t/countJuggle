import cv2
from ultralytics import YOLO
import numpy as np

class CountJuggles:
    def __init__(self):
        self.model = YOLO("/Users/tomasgear/Desktop/Projects/Development/countJuggle/best.pt")
        self.pose_model = YOLO("yolov8s-pose.pt")

        video_path = "/Users/tomasgear/Desktop/Projects/Development/countJuggle/videos/edited/juggle_sample_2.mp4"
        self.cap = cv2.VideoCapture(video_path)

        # Combined juggle counting
        self.juggle_count = 0
        self.recent_juggle_detected = False
        self.juggle_detection_cooldown = 10  # Adjusted cooldown frames
        self.frame_counter_since_last_juggle = 0

        # Proximity-based counting
        self.body_index = {"left_ankle": 15, "right_ankle": 16, "left_knee": 13, "right_knee": 14, "head": 0}
        self.proximity_threshold = 100  # Increased proximity threshold

        # Trajectory-based counting
        self.prev_y_center = None
        self.moving_up = False
        self.movement_threshold = 15  # Increased movement threshold
        self.max_upward_movement = 0

        self.frame_skip = 1
        self.frame_counter = 0

    def run(self):
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if success:
                self.frame_counter += 1

                if self.frame_counter >= self.frame_skip:
                    self.frame_counter = 0
                    ball_position = None

                    pose_results = self.pose_model(frame, verbose=False, conf=0.5)
                    pose_annotated_frame = pose_results[0].plot()

                    results_list = self.model(frame, verbose=False, conf=0.60)
                    for results in results_list:
                        for bbox in results.boxes.xyxy:
                            x1, y1, x2, y2 = bbox[:4]
                            ball_position = (x1 + x2) / 2, (y1 + y2) / 2
                            cv2.rectangle(pose_annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            cv2.putText(pose_annotated_frame, "Soccer Ball", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    keypoints_data = pose_results[0].keypoints.data
                    if keypoints_data.shape[1] > max([value for key, value in self.body_index.items() if isinstance(value, int)]):
                        body_parts = {key: keypoints_data[0, value, :2] for key, value in self.body_index.items() if isinstance(value, int)}

                        if ball_position is not None:
                            self.update_juggle_count(ball_position, body_parts)

                    self.display_text(pose_annotated_frame, f'Juggles: {self.juggle_count}')
                    cv2.imshow("YOLOv8 Inference", pose_annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            else:
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def update_juggle_count(self, ball_position, body_parts):
        proximity_detected = any(np.linalg.norm(np.array(pos) - np.array(ball_position)) < self.proximity_threshold for pos in body_parts.values())

        # Trajectory analysis
        trajectory_detected = False
        x_center, y_center = ball_position
        if self.prev_y_center is not None:
            movement = y_center - self.prev_y_center
            if movement < -self.movement_threshold:
                self.max_upward_movement = min(self.max_upward_movement, movement)
