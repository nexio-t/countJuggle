import cv2
from ultralytics import YOLO
import numpy as np

# juggle sample 1 - WORKS!!!
# juggle sample 6? - very close - TRY WITH NEW MODEL
# juggle sample 7 - WORKS!!!!

# for some reason new model isn't working as well with sample 1
# Tomorrow: 
# 1. make it more sensitive to the ball movement on video 6 below 
# 2. Once you do that make sure it works with 6 and 7 
# 3. Then try with sample 1, or revert sample 1 to the best.pt model


# December 21 current
# sample 2 -- DONE 
# sample 7 -- DONE - need to cut off before the last second as it counts 21 juggles
# sample 1 is not working at all -- the ball detection just isn't working
# sample 6 -- DONE, works until about juggle 10 - so DONE

### - Create Gif that shows video 1 and video 3 with the .best model as it more accurately tracks the ball

class CountJuggles:
    def __init__(self):
        self.model = YOLO("/Users/tomasgear/Desktop/Projects/Development/countJuggle/best_renamed.pt")
        self.pose_model = YOLO("yolov8s-pose.pt")

        video_path = "/Users/tomasgear/Desktop/Projects/Development/countJuggle/videos/edited/juggle_sample_5.mp4"
        self.cap = cv2.VideoCapture(video_path)

        # Combined juggle counting
        self.juggle_count = 0
        self.recent_juggle_detected = False
        self.juggle_detection_cooldown = 10  # frames to wait before considering a new juggle
        self.frame_counter_since_last_juggle = 0

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

    # def update_juggle_count(self, ball_position, body_parts):
    #     proximity_detected = any(np.linalg.norm(np.array(pos) - np.array(ball_position)) < self.proximity_threshold for pos in body_parts.values())

    #     # Trajectory analysis
    #     trajectory_detected = False
    #     x_center, y_center = ball_position
    #     if self.prev_y_center is not None:
    #         movement = y_center - self.prev_y_center
    #         print('movement is: ', movement, 'y_center is: ', y_center, 'prev_y_center is: ', self.prev_y_center, 'movement < -self.movement_threshold is: ', movement < -self.movement_threshold)
    #         if movement < -self.movement_threshold:
    #             print('condition one')
    #             self.max_upward_movement = min(self.max_upward_movement, movement)
    #             self.moving_up = True
    #         elif movement > self.movement_threshold and self.moving_up:
    #             print('condition two ===========')
    #             if abs(self.max_upward_movement) > self.movement_threshold:
    #                 trajectory_detected = True
    #             self.moving_up = False
    #             self.max_upward_movement = 0 


    #     print('proximity_detected is: ', proximity_detected, 'trajectory_detected', trajectory_detected)
    #     # Combined juggle count logic
    #     if trajectory_detected and not self.recent_juggle_detected and proximity_detected:
    #         self.juggle_count += 1
    #         self.recent_juggle_detected = True
    #         self.frame_counter_since_last_juggle = 0

    #     if self.recent_juggle_detected:
    #         self.frame_counter_since_last_juggle += 1
    #         if self.frame_counter_since_last_juggle > self.juggle_detection_cooldown:
    #             self.recent_juggle_detected = False
            

    #     self.prev_y_center = y_center

    def update_juggle_count(self, ball_position, body_parts):
        proximity_detected = any(np.linalg.norm(np.array(pos) - np.array(ball_position)) < self.proximity_threshold for pos in body_parts.values())

         # Calculate the person's bounding box
        body_x_positions, body_y_positions = zip(*body_parts.values())
        min_x, max_x = min(body_x_positions), max(body_x_positions)
        min_y, max_y = min(body_y_positions), max(body_y_positions)
        person_bbox = (min_x, min_y, max_x, max_y)

        # Check if the ball is within or near the person's bounding box
        ball_near_person = self.is_ball_near_person(ball_position, person_bbox)

        # Trajectory analysis
        trajectory_detected = False
        x_center, y_center = ball_position
        if self.prev_y_center is not None:
            movement = y_center - self.prev_y_center
            print('movement is: ', movement, 'y_center is: ', y_center, 'prev_y_center is: ', self.prev_y_center, 'movement < -self.movement_threshold is: ', movement < -self.movement_threshold)
            
            if movement < -self.movement_threshold:
                print('condition one')
                self.max_upward_movement = min(self.max_upward_movement, movement)
                self.moving_up = True
            elif self.moving_up:
                # New condition: Check if the ball's upward movement has slowed down significantly
                if abs(movement) < self.movement_threshold:  # Adjust the multiplier as needed
                    print('condition peak detected')
                    trajectory_detected = True
                    self.moving_up = False
                    self.max_upward_movement = 0
                # elif movement > self.movement_threshold:
                #     print('condition two ===========')
                #     if abs(self.max_upward_movement) > self.movement_threshold:
                #         trajectory_detected = True
                #     self.moving_up = False
                #     self.max_upward_movement = 0

        print('proximity_detected is: ', proximity_detected, 'trajectory_detected', trajectory_detected)
        print('trajectory_detected is: ', trajectory_detected, 'self.recent_juggle_detected', not self.recent_juggle_detected, "proximity_detected", proximity_detected)
        # Combined juggle count logic
        if trajectory_detected and not self.recent_juggle_detected and (proximity_detected or ball_near_person):
            print('+++++++ JUGGLE INCREMENTED ++++++')
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

        buffer = 50  # Adjust this buffer distance as needed
        if (min_x - buffer <= x_center <= max_x + buffer) and (min_y - buffer <= y_center <= max_y + buffer):
            return True
        return False

    def display_text(self, frame, text):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        thickness = 3
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

        padding = 20

        top_left = (10, 50)
        bottom_right = (10 + text_size[0] + padding, 50 + text_size[1] + padding)

        cv2.rectangle(frame, top_left, bottom_right, (255, 255, 255), -1)

        cv2.putText(frame, text, 
                    (10 + padding // 2, 50 + text_size[1] + padding // 2), 
                    font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

if __name__ == "__main__":
    count_juggles = CountJuggles()
    count_juggles.run()
