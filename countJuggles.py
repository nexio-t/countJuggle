import cv2
from ultralytics import YOLO
import numpy as np
from datetime import datetime

class CountJuggles:
    def __init__(self):

        # ---CHANGE THIS LINE TO THE PATH OF YOUR WEIGHTS--- #
        self.model = YOLO("/path/to/weights/best.pt")
        self.pose_model = YOLO("yolov8s-pose.pt")
        
        # ---CHANGE THIS LINE TO THE PATH OF YOUR VIDEO--- #
        video_path = "/path/to/video.mp4"
        self.cap = cv2.VideoCapture(video_path)

        # Combined juggle counting
        self.juggle_count = 0
        self.recent_juggle_detected = False
        self.juggle_detection_cooldown = 10 
        self.frame_counter_since_last_juggle = 0

        # Video output
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
        self.out = None  

        # Proximity-based counting
        self.body_index = {"left_ankle": 15, "right_ankle": 16, "left_knee": 13, "right_knee": 14, "head": 0}
        self.proximity_threshold = 300  

        # Trajectory-based counting
        self.prev_y_center = None
        self.moving_up = False
        self.movement_threshold = 20
        self.max_upward_movement = 0

        self.frame_skip = 0
        self.frame_counter = 0

    def run(self):
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if success:
                self.frame_counter += 1

                if self.out is None:
                    frame_height, frame_width = frame.shape[:2]
                    now = datetime.now()
                    timestamp = now.strftime("%m-%d-%Y-%H%M%S")
                    output_path =  f"/Users/tomasgear/Desktop/Projects/Development/countJuggle/videos/output/{timestamp}.mp4" 
                    self.out = cv2.VideoWriter(output_path, self.fourcc, 20.0, (frame_width, frame_height))

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
                            cv2.putText(pose_annotated_frame, "Soccer Ball", (int(x1), int(y1) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, 
                            (255, 255, 255), 3)  

                    keypoints_data = pose_results[0].keypoints.data
                    if keypoints_data.shape[1] > max([value for key, value in self.body_index.items() if isinstance(value, int)]):
                        body_parts = {key: keypoints_data[0, value, :2] for key, value in self.body_index.items() if isinstance(value, int)}

                        if ball_position is not None:
                            self.update_juggle_count(ball_position, body_parts)

                    self.display_text(pose_annotated_frame, f'Juggles: {self.juggle_count}')

                    self.out.write(pose_annotated_frame)  
                    cv2.imshow("YOLOv8 Inference", pose_annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            else:
                break

        self.out.release() 
        self.cap.release()
        cv2.destroyAllWindows()

    def update_juggle_count(self, ball_position, body_parts):
        proximity_detected = any(np.linalg.norm(np.array(pos) - np.array(ball_position)) < self.proximity_threshold for pos in body_parts.values())

        body_x_positions, body_y_positions = zip(*body_parts.values())
        min_x, max_x = min(body_x_positions), max(body_x_positions)
        min_y, max_y = min(body_y_positions), max(body_y_positions)
        person_bbox = (min_x, min_y, max_x, max_y)

        ball_near_person = self.is_ball_near_person(ball_position, person_bbox)

        trajectory_detected = False
        _, y_center = ball_position
        if self.prev_y_center is not None:
            movement = y_center - self.prev_y_center
            
            if movement < -self.movement_threshold:
                self.max_upward_movement = min(self.max_upward_movement, movement)
                self.moving_up = True
            elif self.moving_up:
                if abs(movement) < self.movement_threshold:  
                    trajectory_detected = True
                    self.moving_up = False
                    self.max_upward_movement = 0
     
        if trajectory_detected and not self.recent_juggle_detected and (proximity_detected or ball_near_person):
            self.juggle_count += 1
            self.recent_juggle_detected = True
            self.frame_counter_since_last_juggle = 0

        if self.recent_juggle_detected:
            self.frame_counter_since_last_juggle += 1
            if self.frame_counter_since_last_juggle > self.juggle_detection_cooldown:
                self.recent_juggle_detected = False

        self.prev_y_center = y_center

    def is_ball_near_person(self, ball_position, person_bbox):
        x_center, y_center = ball_position
        min_x, min_y, max_x, max_y = person_bbox

        buffer = 50 
        if (min_x - buffer <= x_center <= max_x + buffer) and (min_y - buffer <= y_center <= max_y + buffer):
            return True
        return False

    def display_text(self, frame, text):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 3
        thickness = 5
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

        padding = 60

        top_left = (10, 50)
        bottom_right = (10 + text_size[0] + padding, 50 + text_size[1] + padding)

        cv2.rectangle(frame, top_left, bottom_right, (255, 255, 255), -1)

        cv2.putText(frame, text, 
                    (10 + padding // 2, 50 + text_size[1] + padding // 2), 
                    font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

if __name__ == "__main__":
    count_juggles = CountJuggles()
    count_juggles.run()
