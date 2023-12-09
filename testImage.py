import cv2
from ultralytics import YOLO
import os

# Load the trained YOLO model
model_path = "/Users/tomasgear/Desktop/Projects/Development/countJuggle/countJuggles.pt"
model = YOLO(model_path)

# Print the current working directory
print("Current Working Directory:", os.getcwd())

# Check if the image file exists
image_path = "/Users/tomasgear/Desktop/Projects/Development/countJuggle/football_2.jpg"
print("Does the image exist?", os.path.exists(image_path))

# Load the image
image = cv2.imread(image_path)
if image is None:
    print(f"Error: Image not found at {image_path}")
else:
    # Perform object detection
    results = model(image, conf=0.60)

    # Extract and print detected objects and their labels
    for result in results.xyxy[0]:
        label = results.names[int(result[5])]
        print(f"Detected: {label}")

    # Draw bounding boxes and labels on the image
    annotated_image = results.plot()
    
    # Display the annotated image
    cv2.imshow("Detection", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()