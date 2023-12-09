import cv2
from ultralytics import YOLO

class BallTracker:
    def __init__(self):
        # Load the YOLO model for ball detection
        self.model = YOLO("/Users/tomasgear/Desktop/Projects/Development/countJuggles.pt")

        # Open the video file or webcam
        self.cap = cv2.VideoCapture("path/to/your/video.mp4")  # Change to 0 or 1 for webcam

    def run(self):
        # Process frames from the video
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                break

            # Detect the ball
            results = self.model(frame, conf=0.65)

            # Process each detection
            for bbox in results.xyxy[0]:
                x1, y1, x2, y2 = bbox[:4]

                # Calculate center coordinates of the ball
                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2

                # Annotate and display the frame
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.circle(frame, (int(x_center), int(y_center)), 5, (0, 0, 255), -1)
                cv2.imshow("Ball Tracking", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # Release resources
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    ball_tracker = BallTracker()
    ball_tracker.run()
