import cv2
from ultralytics import YOLO

# Issues: Right now, the video is very slow and glitchy. The video needs to be smooth. 
# SOLVED Issue 2: Why is a sports ball label being used? I called it soccer ball when I trained it (I used the wrong model)
# Issue 3: I need to make sure I'm using GPU and CPU

class CountJuggles:
    def __init__(self):
        # Load the YOLO model for ball detection
        self.model = YOLO("/Users/tomasgear/Desktop/Projects/Development/countJuggle/best.pt")
        self.pose_model = YOLO("yolov8s-pose.pt")

        video_path = "/Users/tomasgear/Desktop/Projects/Development/countJuggle/videos/edited/juggling_sample_1.mp4"
        self.cap = cv2.VideoCapture(video_path)

        # Initialize variables for juggling counting
        self.prev_ball_position = None
        self.juggle_count = 0

        # Frame skipping
        self.frame_skip = 2  # process every 5th frame, adjust as needed
        self.frame_counter = 0

    def run(self):
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if success:
                self.frame_counter += 1

                # Process the frame only if the frame counter reaches the frame skip count
                if self.frame_counter >= self.frame_skip:
                    self.frame_counter = 0  # Reset frame counter
                    results_list = self.model(frame, verbose=False, conf=0.60)

                    for results in results_list:
                        for bbox in results.boxes.xyxy:
                            x1, y1, x2, y2 = bbox[:4]

                            ball_position = (x1 + x2) / 2, (y1 + y2) / 2
                            print(f"Soccer Ball coordinates: (x={ball_position[0]:.2f}, y={ball_position[1]:.2f})")

                            self.update_juggle_count(ball_position)

                            self.prev_ball_position = ball_position

                        annotated_frame = results.plot()

                        # Display the juggle count on the frame

                        cv2.imshow("YOLOv8 Inference", annotated_frame)

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            else:
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def update_juggle_count(self, ball_position):
        # Update your juggle counting logic here
        if self.prev_ball_position is not None:
            # Example logic, replace with your own
            if ball_position[1] < self.prev_ball_position[1]:
                self.juggle_count += 1
            print(f"Juggle Count: {self.juggle_count}")

if __name__ == "__main__":
    count_juggles = CountJuggles()
    count_juggles.run()
