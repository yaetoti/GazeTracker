# Foveated Rendering R&D Project (Part 2: Gaze Tracker)

Part of a multidisciplinary research project that combines computer vision with a custom graphics engine. The goal of the project is to adapt foveated rendering methods for ray-tracing systems on PCs using an RGB camera to achieve an increase in rendering speed. Developed as a part of a master's thesis (grade: 100/100).

Gaze detector demonstration:

https://github.com/user-attachments/assets/9e13692b-5344-44d9-823e-45e1530d853a

Foveated rendering demonstration (I am looking at the mouse cursor):

https://github.com/user-attachments/assets/cf51a411-b808-4002-a362-b74ac2107873

The project's homepage and the beginning of the installation instructions can be found at the [link](https://github.com/yaetoti/RaytracerCUDA.git)

## Pipeline overview

To track the gaze coordinates on the screen, this project uses the MediaPipe Facemesh library and an MLP calibration model, which converts the extracted features into coordinates. The model is trained by a specific user, which resolves the issue of variations in screen size, position, and camera placement.

There are linear and classification calibration models. Regardless of which model you choose, the workflow for using them is virtually the same.

1. The first step is to record the screen coordinates and images of the face looking at the displayed point at that moment. The coordinates and image names are saved to the first metadata file.
2. In the second step, the recorded data is preprocessed: facial features are extracted, relative coordinates of markers are calculated, as well as distances between points and head rotations. The results are saved to the second metadata file.
3. The third step is calibrating the model based on the extracted features. This step is performed automatically, though the user can specify the number of iterations and fine-tune the model on new data. After that, the model weights are generated.
4. The fourth stage – displaying the predicted point on the screen in real time and calculating statistics. A service is also available that provides gaze coordinates to the ray tracer from the first part.

### Data conventions

**metadataStage0.txt**

```
# First line
<width> <height>
# Following lines
<image_name_with_extension> <x_normalized> <y_normalized>
```

**metadataStage1.txt**

```
# First line
<width> <height>
# Following lines
<image_name_with_extension> <x_normalized> <y_normalized> <feature0> <feature1> ... <featureN>
```

**Shared memory structure**
```
float x; // 4 bytes
float y; // 4 bytes
```

## Prerequisites

- Webcam
- **Python 3.10+** (with **torch**, **mediapipe**, **pygame**, **opencv-python**)

## Installation & Execution

Install the required Python dependencies:

```
pip install -r requirements.txt
```

The tracker requires a quick personalized calibration process. Follow these steps in exact order:

Go to the directory of the model you want:

```sh
cd MediaPipeClassification
# or
cd MediaPipeLinear
```

### 1 Recording
Run one of the recording scripts of your choice (prefixed with `N1`):

```sh
python N1_RecordLinear.py
# or
python N1_RecordClasses.py
```

You will be prompted to start recording from scratch or add data to an existing recording. A PyGame window will then open. Press the spacebar to start recording and follow the moving dot with your eyes, slightly moving your head (or move your eyes within the outlined area if you chose the second approach. In this case, you can also switch between zones using the TAB key in any order).

When you feel you have enough data, press ECS: the window will close, and the data will be saved automatically. Usually, about 300 frames are sufficient for each of the four main head rotation directions per screen segment. For the linear approach it's about 1000 frames.

Both too little and too much data can lead to noisy samples or overfitting, so try to find an option that works for you. At this stage, the tracker is very sensitive to data variability.

If you want to change the number of frames per segment, you'll need to modify the `images` variable in the source code. There's no handy UI since the project is in the proof-of-concept stage.

### 2 Postprocess

This step is performed automatically. Just like with recording, you will be prompted to update the existing data or restart the entire process.

```sh
python N2_Postprocess.py
```

### 3 Calibration

This step is automated, but you can continue to train the model as many times as you like, as long as it makes sense. The console displays the iteration number, the loss function, and the average accuracy deviation in pixels.

```sh
python N3_Calibration.py
```

To change the number of iterations per run, modify the `epochs` variable in the script.

### 4 Execution

You can test the model directly in user interface mode by running the command below. The statistics will be displayed to the console. 

```sh
python N4_Predict.py
```

Or you can run the gaze tracker service that sends the coordinates to the engine from the [first part](https://github.com/yaetoti/RaytracerCUDA/tree/gaze-detector)

```sh
python N4_Service.py
```
