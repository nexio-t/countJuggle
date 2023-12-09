import cv2

# Replace with your video file path
video_path = "/Users/tomasgear/Desktop/Projects/Development/countJuggle/videos/edited/juggling_sample_1.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Failed to open video file.")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Video", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
