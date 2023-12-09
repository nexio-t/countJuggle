import cv2
from ultralytics import YOLO
import os

class CountJuggles:
    def __init__(self):
        print('loading model...')
        # Load the YOLO model for ball detection
        self.model = YOLO("/Users/tomasgear/Desktop/Projects/Development/countJuggle/countJuggles.pt")

        video_path = "/Users/tomasgear/Desktop/Projects/Development/countJuggle/videos/edited/juggling_sample_1.mp4"
        if not os.path.exists(video_path):
            print(f"Error: Video file not found at {video_path}")
            return

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            print("Error: Failed to open video file.")
            return

        print('Video file opened successfully.')

    def run(self):
    # Process frames from the video
        while self.cap.isOpened():
            print('processing video file...')
            success, frame = self.cap.read()
            if not success:
                break

            # Detect the ball
            results = self.model(frame, conf=0.65)

            # Inside your run method
            for result in results.xyxy[0]:  # results.xyxy[0] is a tensor of shape (n, 6) where n is the number of detections
                if result[5] == 0:  # Assuming class '0' is for the ball; modify as per your model's classes
                    x1, y1, x2, y2 = int(result[0]), int(result[1]), int(result[2]), int(result[3])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw a rectangle


            # Display the frame
            cv2.imshow("Video", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release resources
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    count_juggles = CountJuggles()
    count_juggles.run()
