import cv2
from ultralytics import YOLO

class CountJuggles:
    def __init__(self):
        self.model = YOLO("/Users/tomasgear/Desktop/Projects/Development/countJuggle/best.pt")
        video_path = "/Users/tomasgear/Desktop/Projects/Development/countJuggle/videos/edited/juggle_sample_2.mp4"
        # video_path = "/Users/tomasgear/Desktop/Projects/Development/countJuggle/videos/edited/juggle_sample_3.mp4"
        # video_path = "/Users/tomasgear/Desktop/Projects/Development/countJuggle/videos/edited/juggling_sample_1.mp4"
        self.cap = cv2.VideoCapture(video_path)

    def run(self):
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if success:
                results_list = self.model(frame, verbose=False, conf=0.40)
                annotated_frame = frame.copy()

                for results in results_list:
                    for bbox in results.boxes.xyxy:
                        x1, y1, x2, y2 = bbox[:4]
                        
                        label = "Soccer Ball" 

                        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        
                        label_text = f"{label}"
                        cv2.putText(annotated_frame, label_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                cv2.imshow("YOLOv8 Inference", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    count_juggles = CountJuggles()
    count_juggles.run()
