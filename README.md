# Foveated Rendering R&D Project (Part 2: Gaze Tracker)

Part of a multidisciplinary research project that combines computer vision with a custom graphics engine. The goal of the project is to adapt foveated rendering methods for ray-tracing systems on PCs using an RGB camera to achieve an increase in rendering speed.

[video]

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

