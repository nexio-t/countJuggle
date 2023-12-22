import cv2
from ultralytics import YOLO
import numpy as np

### December 16, 2023 
### Script is working and accurately counting juggles after increasing proximity threshold to 70 pixels
### Next: Try to find a way to count juggles agnostic of proximity to ankles, meaning just based on air movement alone
class CountJuggles:
    def __init__(self):
        self.model = YOLO("/Users/tomasgear/Desktop/Projects/Development/countJuggle/best.pt")
        self.pose_model = YOLO("yolov8s-pose.pt")

        # video_path = "/Users/tomasgear/Desktop/Projects/Development/countJuggle/videos/edited/juggle_sample_1.mp4"
        video_path = "/Users/tomasgear/Desktop/Projects/Development/countJuggle/videos/edited/juggle_sample_2.mp4"
        # video_path = "/Users/tomasgear/Desktop/Projects/Development/countJuggle/videos/edited/juggle_sample_3.mp4"

        self.cap = cv2.VideoCapture(video_path)

        self.has_juggled_recently = False

        self.body_index = {"left_ankle": 15, "right_ankle": 16}
        self.prev_ball_position = None
        self.juggle_count = 0
        self.proximity_threshold = 70  
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
                    if keypoints_data.shape[1] > max(self.body_index.values()):
                        left_ankle = keypoints_data[0, self.body_index["left_ankle"], :2]
                        right_ankle = keypoints_data[0, self.body_index["right_ankle"], :2]

                        if ball_position is not None:
                            self.update_juggle_count(ball_position, left_ankle, right_ankle)
                            self.prev_ball_position = ball_position  

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

                    # font = cv2.FONT_HERSHEY_SIMPLEX
                    # cv2.putText(pose_annotated_frame, f'Juggles: {self.juggle_count}', 
                    #             (10, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.imshow("YOLOv8 Inference", pose_annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            else:
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def update_juggle_count(self, ball_position, left_ankle, right_ankle):
        distance_to_left = np.linalg.norm(np.array(left_ankle) - np.array(ball_position))
        distance_to_right = np.linalg.norm(np.array(right_ankle) - np.array(ball_position))

        proximity_to_ankle = distance_to_left < self.proximity_threshold or distance_to_right < self.proximity_threshold

        print('proximity_to_ankle: ', proximity_to_ankle)

        if self.prev_ball_position is not None:
            if not self.has_juggled_recently and proximity_to_ankle:
                self.juggle_count += 1
                self.has_juggled_recently = True
            elif not proximity_to_ankle:
                self.has_juggled_recently = False  
        else:
            if proximity_to_ankle:
                self.juggle_count += 1 

        self.prev_ball_position = ball_position  

        print('Distance to left ankle: ', distance_to_left)
        print('Distance to right ankle: ', distance_to_right)
        print('Ball position: ', ball_position)
        print(f"Juggle Count: {self.juggle_count}")



if __name__ == "__main__":
    count_juggles = CountJuggles()
    count_juggles.run()
