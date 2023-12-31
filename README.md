# Juggle Counter ⚡︎⚽ ︎︎

![Output Example](./assets/output.gif)

## Overview
The Juggle Counter counts the number of times a player validly juggles soccer ball using <a href="https://github.com/ultralytics/ultralytics" target="_blank">YOLOv8's</a> pose estimation model and a separate fine-tuned YOLOv8 object detection model with the labeling assitance from <a href="https://public.roboflow.com/"  target="_blank">Roboflow</a>.

## Context 
I first set out to create this project to immerse myself cutting-edge computer vision models and libraries, inspired in part by this post I had seen by McKay Wrigley earlier this year. I first collected an assortment of soccer ball images from Kaggle, which were automatically labeled with Roboflow's <a href="https://blog.roboflow.com/autodistill/">Autodistill library</a>, saving me hours of manual labeling work. YOLOv8's object detection model was then fine-tuned with the assets. The rest consisted of generating an algorithm that combined ball tracking with player proximity tracking.  

## Features
- **Juggle Detection:** Uses advanced object detection from a fine-tuned YOLOv8 model to count soccer ball juggles accurately.
- **Human Pose Estimation:** Implements YOLOv8 pose detection model for detecting the player's poses, aiding in the juggle counting process.
- **Proximity and Trajectory Tracking:** Combines proximity-based and trajectory-based techniques for reliable juggle detection.

## Installation
1. **Clone the Repository:**
   ```
   git clone git@github.com:nexio-t/countJuggle.git
   ```
2. **Install Dependencies:**
   ```
   pip install -r requirements.txt
   ```
3. **Run the setup script:**
   ```
   python setupDirectories.py
   ```

## Assets
You can run this script against the video clips found <a href="https://drive.google.com/drive/folders/1TWDXXCVKoTqzt0nEnrsu77PWe4Nz3oRg" target="_blank">here</a>, or you can use your own assets. You must download the trained model best.pt <a href="https://drive.google.com/drive/folders/1Aa6gIt189lr_i8PW8bpKgNVFH9rYdM-q" target="_blank">here</a> and save in the root of the directory. 

## Usage
1. Place still clips of a player juggling in the `videos/input` directory.
2. Update the `video_path` variable in the `CountJuggles` class initializer with the path to your video file.
3. Update the `model` variable in the `CountJuggles` class initializer with the path to the model file, which I've provided above in the assets section. 
4. Run the script:
   ```
   python countJuggles.py
   ```
As the script is running, your video clip should open with the appropriate annotations, like so: 

<img src="./assets/annotation_example.png" alt="Annotation Example" width="300"/>

## Detailed Script Functionality

### Class: `CountJuggles`
- **Initialization:** Sets up the YOLO models and the video capture object.
- **Method `run`:** Processes each video frame, applies pose estimation, and detects juggles.
- **Juggle Counting Logic:** Uses a combination of proximity and trajectory analysis for accurate juggle detection.
- **Video Output:** Annotated video output showing the juggle count per frame.

### Key Parameters and Variables
- `juggle_detection_cooldown`: Time (in frames) to wait before considering a new juggle.
- `proximity_threshold`: Maximum distance between the ball and body parts for counting a juggle.
- `movement_threshold`: Minimum movement required to detect an upward trajectory of the ball.
- `frame_skip`: Number of frames to skip for each processing step to optimize performance.

## Customization
- Adjust the `proximity_threshold`, `movement_threshold`, and `frame_skip` parameters in the `CountJuggles` class to optimize performance based on your video's characteristics.

## Technologies Used
- **Python**
- **OpenCV (cv2)** 
- **YOLOv8** 
- **NumPy** 
- **Roboflow** 

## Limitations
- The accuracy of juggle counting is highly dependent on the video quality, lighting conditions, and of course the object detection accuracy. 
- Video frames must be still. Videos in which there are cuts and other edits will not work as the model loses track of the ball in frame. 
- Performance is subject to the limitations of the YOLO model and OpenCV processing capabilities.
- Expect bugs and miscounts. This script cannot handle all juggling scenarios.

## Samples 

You can view some output examples <a href="https://drive.google.com/drive/u/0/folders/1bHn0yV89h4GDTq0H_t-k7C0tni7mckdH" target="_blank">here</a>. 

## License
This project is released under the MIT License. Refer to the [LICENSE](LICENSE.txt) file for detailed information.