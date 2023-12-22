import cv2
from ultralytics import YOLO
import numpy as np


# Video 1: super inaccurate - way under counts
# Video 2: super inaccurate - way OVER counts
# Video 3: severly undercounts

class CountJuggles:
    def __init__(self):
        self.model = YOLO("/Users/tomasgear/Desktop/Projects/Development/countJuggle/best.pt")
        self.pose_model = YOLO("yolov8s-pose.pt")

        # video_path = "/Users/tomasgear/Desktop/Projects/Development/countJuggle/videos/edited/juggle_sample_1.mp4"
        video_path = "/Users/tomasgear/Desktop/Projects/Development/countJuggle/videos/edited/juggle_sample_2.mp4"
        # video_path = "/Users/tomasgear/Desktop/Projects/Development/countJuggle/videos/edited/juggle_sample_3.mp4"
        self.cap = cv2.VideoCapture(video_path)

        self.has_juggled_recently = False

        # Expanded to include knees and head keypoints
        self.body_index = {"left_ankle": 15, "right_ankle": 16, "left_knee": 13, "right_knee": 14, "head": 0}
        self.prev_ball_position = None
        self.juggle_count = 0
        self.proximity_threshold = 300  
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

                            label = "Soccer Ball"
                            cv2.putText(pose_annotated_frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    keypoints_data = pose_results[0].keypoints.data
                    if keypoints_data.shape[1] > max([value for key, value in self.body_index.items() if isinstance(value, int)]):
                        body_parts = {key: keypoints_data[0, value, :2] for key, value in self.body_index.items() if isinstance(value, int)}

                        if ball_position is not None:
                            self.update_juggle_count(ball_position, body_parts)
                            self.prev_ball_position = ball_position  

                    text = f'Juggles: {self.juggle_count}'
                    self.display_text(pose_annotated_frame, text)

                    cv2.imshow("YOLOv8 Inference", pose_annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            else:
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def update_juggle_count(self, ball_position, body_parts):
        proximity = any(np.linalg.norm(np.array(pos) - np.array(ball_position)) < self.proximity_threshold for pos in body_parts.values())

        print('Proximity to a body part: ', proximity)

        if self.prev_ball_position is not None:
            if not self.has_juggled_recently and proximity:
                self.juggle_count += 1
                self.has_juggled_recently = True
            elif not proximity:
                self.has_juggled_recently = False  
        else:
            if proximity:
                self.juggle_count += 1 

        self.prev_ball_position = ball_position  

        print('Ball position: ', ball_position)
        print(f"Juggle Count: {self.juggle_count}")

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
