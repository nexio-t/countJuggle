import cv2
from ultralytics import YOLO
import numpy as np
from datetime import datetime

# December 18: This is only counting juggles based on movement of the ball 
# Prematurely counts the juggle before it makes contact with the players body 
# Also outputing video with annotations

# TODOS: 
# Create necessary output directory in setup script
# Fix premature counting -- include exclusion of contact with arms or hands as a juggle t
# Update script to use a more extensively trained model - 30-50 epochs
class CountJuggles:
    def __init__(self):
        self.model = YOLO("/Users/tomasgear/Desktop/Projects/Development/countJuggle/best.pt")
        self.pose_model = YOLO("yolov8s-pose.pt")

        video_path = "/Users/tomasgear/Desktop/Projects/Development/countJuggle/videos/edited/juggle_sample_2.mp4"
        # video_path = "/Users/tomasgear/Desktop/Projects/Development/countJuggle/videos/edited/juggle_sample_3.mp4"
        # video_path = "/Users/tomasgear/Desktop/Projects/Development/countJuggle/videos/edited/juggle_sample_1.mp4"


        self.cap = cv2.VideoCapture(video_path)

        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
        self.out = None  

        self.prev_ball_position = None
        self.juggle_count = 0
        self.frame_skip = 1
        self.frame_counter = 0

        self.prev_y_center = None
        self.moving_up = False
        self.movement_threshold = 10
        self.max_upward_movement = 0

        self.wait_frames = 10  
        self.downward_frames = 0  
        self.juggle_state = 'Ascending'  


    def run(self):
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if success:
                if self.out is None:
                    frame_height, frame_width = frame.shape[:2]
                    now = datetime.now()
                    timestamp = now.strftime("%m-%d-%Y-%H%M%S")
                    # output_path =  f"/Users/tomasgear/Desktop/Projects/Development/countJuggle/videos/output/{timestamp}.mp4" 
                    # self.out = cv2.VideoWriter(output_path, self.fourcc, 20.0, (frame_width, frame_height))

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

                            label = "Soccer Ball"
                            cv2.putText(pose_annotated_frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    if ball_position is not None:
                        self.update_juggle_count(ball_position)

                    text = f'Juggles: {self.juggle_count}'
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1.5
                    thickness = 3
                    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

                    padding = 20

                    top_left = (10, 50)
                    bottom_right = (10 + text_size[0] + padding, 50 + text_size[1] + padding)

                    cv2.rectangle(pose_annotated_frame, top_left, bottom_right, (255, 255, 255), -1)

                    cv2.putText(pose_annotated_frame, text, 
                                (10 + padding // 2, 50 + text_size[1] + padding // 2), 
                                font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

                    # self.out.write(pose_annotated_frame)

                    cv2.imshow("YOLOv8 Inference", pose_annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            else:
                break

        self.cap.release()
        self.out.release()  
        cv2.destroyAllWindows()

    def update_juggle_count(self, ball_position):
        x_center, y_center = ball_position

        if self.prev_y_center is not None:
            movement = y_center - self.prev_y_center

            if self.juggle_state == 'Ascending':
                if movement < -self.movement_threshold:
                    self.juggle_state = 'Peak'

            elif self.juggle_state == 'Peak':
                if movement > self.movement_threshold:
                    self.juggle_state = 'Descending'

            elif self.juggle_state == 'Descending':
                if movement < -self.movement_threshold:
                    self.juggle_count += 1
                    self.juggle_state = 'Ascending'

        self.prev_y_center = y_center

        print('Ball position: ', ball_position)
        print(f"Juggle Count: {self.juggle_count}")


if __name__ == "__main__":
    count_juggles = CountJuggles()
    count_juggles.run()
