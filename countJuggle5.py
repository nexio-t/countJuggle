import cv2
from ultralytics import YOLO
import numpy as np

class CountJuggles:
    def __init__(self):
        # Load the YOLO models for ball detection and pose estimation
        self.model = YOLO("/Users/tomasgear/Desktop/Projects/Development/countJuggle/best.pt")
        self.pose_model = YOLO("yolov8s-pose.pt")

        video_path = "/Users/tomasgear/Desktop/Projects/Development/countJuggle/videos/edited/juggling_sample_1.mp4"
        self.cap = cv2.VideoCapture(video_path)

        # Indices for ankle keypoints
        self.body_index = {"left_ankle": 15, "right_ankle": 16}

        # Initialize variables for juggling counting
        self.prev_ball_position = None
        self.juggle_count = 0

        # Frame skipping for performance optimization
        self.frame_skip = 2
        self.frame_counter = 0

    def run(self):
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if success:
                self.frame_counter += 1

                if self.frame_counter >= self.frame_skip:
                    self.frame_counter = 0

                    # Pose estimation
                    pose_results = self.pose_model(frame, verbose=False, conf=0.5)
                    pose_annotated_frame = pose_results[0].plot()

                    # Extract keypoints data
                    keypoints_data = pose_results[0].keypoints.data

                    # Check if the specific keypoints (left and right ankles) are detected
                    if keypoints_data.shape[1] > max(self.body_index.values()):
                        # Extract and print ankle positions
                        left_ankle = keypoints_data[0, self.body_index["left_ankle"], :2]
                        right_ankle = keypoints_data[0, self.body_index["right_ankle"], :2]

                        print(f"Left Ankle: ({left_ankle[0]:.2f}, {left_ankle[1]:.2f})")
                        print(f"Right Ankle: ({right_ankle[0]:.2f}, {right_ankle[1]:.2f})")
                    else:
                        print("Specific keypoints (left and right ankles) are not detected.")

                    # Ball detection
                    results_list = self.model(frame, verbose=False, conf=0.60)
                    for results in results_list:
                        for bbox in results.boxes.xyxy:
                            x1, y1, x2, y2 = bbox[:4]
                            ball_position = (x1 + x2) / 2, (y1 + y2) / 2
                            print(f"Soccer ball coordinates: (x={ball_position[0]:.2f}, y={ball_position[1]:.2f})")

                            self.update_juggle_count(ball_position)
                            self.prev_ball_position = ball_position

                            # Annotate the frame with the bounding box of the ball
                            cv2.rectangle(pose_annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                    # Display the juggle count and pose on the frame
                    cv2.imshow("YOLOv8 Inference", pose_annotated_frame)

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            else:
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def update_juggle_count(self, ball_position):
        if self.prev_ball_position is not None:
            if ball_position[1] < self.prev_ball_position[1]:
                self.juggle_count += 1
            print(f"Juggle Count: {self.juggle_count}")

if __name__ == "__main__":
    count_juggles = CountJuggles()
    count_juggles.run()
