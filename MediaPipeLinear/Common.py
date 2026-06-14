import math

import cv2
import mediapipe as mp
import numpy as np
import torch

mp_face_mesh = mp.solutions.face_mesh

PUPILS = [
  468, # left pupil
  473, # right pupil
]

LEFT_EYE = [
  33, 246, 161, 160, 159, 158, 157, 173,
  133, 155, 154, 153, 145, 144, 163, 7
]

RIGHT_EYE = [
  362, 398, 384, 385, 386, 387, 388, 466,
  263, 249, 390, 373, 374, 380, 381, 382
]

INDICES = PUPILS + LEFT_EYE + RIGHT_EYE
INPUT_SIZE = len(LEFT_EYE) + len(RIGHT_EYE) + 1
X_CLASSES = 3
Y_CLASSES = 3



def Classify(value, min_val, max_val, classes):
  if value < min_val:
    return 0
  if value >= max_val:
    return classes - 1

  step = (max_val - min_val) / classes
  class_index = int((value - min_val) / step)

  return class_index

def Declassify(cls, min_val, max_val, classes):
  return (cls / classes) * (max_val - min_val) + min_val

def classAABB(classX, classY, classesX, classesY, width, height):
  stepX = width / classesX
  stepY = height / classesY
  minX = Declassify(classX, 0, width, classesX)
  minY = Declassify(classY, 0, height, classesY)
  maxX = minX + stepX
  maxY = minY + stepY
  return minX, minY, maxX, maxY



def MinCoord(array):
  return [min(coord[0] for coord in array), min(coord[1] for coord in array)]

def MaxCoord(array):
  return [max(coord[0] for coord in array), max(coord[1] for coord in array)]

def NormalizeCoords(array):
  '''
  Normalizes coords in AABB
  :param array: coords
  :return: normalized coords
  '''

  minX = min(coord[0] for coord in array)
  minY = min(coord[1] for coord in array)
  maxX = max(coord[0] for coord in array)
  maxY = max(coord[1] for coord in array)

  return [[(x - minX) / (maxX - minX), (y - minY) / (maxY - minY)] for x, y in array]

def Normalize(x):
  return (x - x.min()) / (x.max() - x.min())


def PreprocessExpected(expected, width, height):
  '''
  Used for preprocessing expected coords in dataset
  :param expected: coords from stage0, normalized over width and height respectively
  :param width: recorded screen width
  :param height: recorded screen height
  :return: classes for model learning
  '''
  # gridSize = 200
  # classesW = int(width / gridSize)
  # classesH = int(height / gridSize)

  # classesW = X_CLASSES
  # classesH = Y_CLASSES
  # return [Classify(expected[0], 0, 1, classesW), Classify(expected[1], 0, 1, classesH)]

  return [expected[0], expected[1]]



class Preprocess:
  def __init__(self):
    self.faceMesh = mp_face_mesh.FaceMesh(
      max_num_faces=1,
      refine_landmarks=True,
      min_detection_confidence=0.5,
      min_tracking_confidence=0.5,
    )

  def __del__(self):
    self.faceMesh.close()

  def Handle(self, frame):
    '''
    Prepares model input data from an image
    :param frame: captured webcam frame (RGB)
    :return: input data for a model
    '''

    height, width, _ = frame.shape
    results = self.faceMesh.process(frame)
    if results.multi_face_landmarks:
      landmarks = results.multi_face_landmarks[0].landmark

      # extract normalized coords
      pupils = [[landmarks[index].x, landmarks[index].y] for index in PUPILS]
      leftEye = [[landmarks[index].x, landmarks[index].y] for index in LEFT_EYE]
      rightEye = [[landmarks[index].x, landmarks[index].y] for index in RIGHT_EYE]

      # normalize
      leftEyeNorm = NormalizeCoords(leftEye)
      rightEyeNorm = NormalizeCoords(rightEye)

      pupils[0] = [(pupils[0][0] - MinCoord(leftEye)[0]) / (MaxCoord(leftEye)[0] - MinCoord(leftEye)[0]),
                   (pupils[0][1] - MinCoord(leftEye)[1]) / (MaxCoord(leftEye)[1] - MinCoord(leftEye)[1])]
      pupils[1] = [(pupils[1][0] - MinCoord(rightEye)[0]) / (MaxCoord(rightEye)[0] - MinCoord(rightEye)[0]),
                   (pupils[1][1] - MinCoord(rightEye)[1]) / (MaxCoord(rightEye)[1] - MinCoord(rightEye)[1])]

      # calculate distances between pupil and eye contour points
      leftDistances = [math.hypot(pupils[0][0] - coord[0], pupils[0][1] - coord[1]) for coord in leftEyeNorm]
      rightDistances = [math.hypot(pupils[1][0] - coord[0], pupils[1][1] - coord[1]) for coord in rightEyeNorm]

      # calculate softmax for distances
      leftSoftmax = torch.softmax(-torch.tensor(leftDistances), 0).tolist()
      rightSoftmax = torch.softmax(-torch.tensor(rightDistances), 0).tolist()


      # Other features
      # Calculate eye distance
      innerLeftEye = (landmarks[362].x, landmarks[362].y)
      innerRightEye = (landmarks[173].x, landmarks[173].y)
      eyeDistance = math.hypot(innerLeftEye[0] - innerRightEye[0], innerLeftEye[1] - innerRightEye[1])

      forehead = (landmarks[10].x, landmarks[10].y)
      nose = (landmarks[1].x, landmarks[1].y)
      faceDistance = math.hypot(forehead[0] - nose[0], forehead[1] - nose[1])

      # print(leftDistances)
      # print(rightDistances)

      # print(leftSoftmax)
      # print(rightSoftmax)

      # # Test render to file
      # # convert normalized coords to pixel coordinates
      # pupils_px = [(int(coord[0] * width), int(coord[1] * height)) for coord in pupils]
      # leftEye_px = [(int(coord[0] * width), int(coord[1] * height)) for coord in leftEye]
      # rightEye_px = [(int(coord[0] * width), int(coord[1] * height)) for coord in rightEye]
      #
      # # draw circles
      # for coord in pupils_px:
      #   cv2.circle(frame, coord, 1, (0, 255, 0), -1)
      # for coord in leftEye_px:
      #   cv2.circle(frame, coord, 1, (255, 0, 0), -1)
      # for coord in rightEye_px:
      #   cv2.circle(frame, coord, 1, (0, 0, 255), -1)
      #
      # # save the modified frame
      # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      # cv2.imwrite("test.png", frame)


      # Test PnP roll pitch yaw
      model_points = np.array([
        (0.0, 0.0, 0.0),          # Nose tip
        (0.0, -330.0, -65.0),     # Chin
        (-225.0, 170.0, -135.0),  # Left eye left corner
        (225.0, 170.0, -135.0),   # Right eye right corner
        (-150.0, -150.0, -125.0), # Left mouth corner
        (150.0, -150.0, -125.0)   # Right mouth corner
      ], dtype="double")

      nose_tip = 1
      chin = 152
      left_eye_corner = 33
      right_eye_corner = 263
      left_mouth = 61
      right_mouth = 291

      image_points = np.array([
        (landmarks[nose_tip].x * width, landmarks[nose_tip].y * height),
        (landmarks[chin].x * width, landmarks[chin].y * height),
        (landmarks[left_eye_corner].x * width, landmarks[left_eye_corner].y * height),
        (landmarks[right_eye_corner].x * width, landmarks[right_eye_corner].y * height),
        (landmarks[left_mouth].x * width, landmarks[left_mouth].y * height),
        (landmarks[right_mouth].x * width, landmarks[right_mouth].y * height)
      ], dtype="double")

      focal_length = width
      center = (width / 2, height / 2)
      camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
      ], dtype="double")
      dist_coeffs = np.zeros((4, 1))

      success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs
      )

      rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

      def rotationMatrixToEulerAngles(R):
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6
        if not singular:
          x = math.atan2(R[2, 1], R[2, 2])
          y = math.atan2(-R[2, 0], sy)
          z = math.atan2(R[1, 0], R[0, 0])
        else:
          x = math.atan2(-R[1, 2], R[1, 1])
          y = math.atan2(-R[2, 0], sy)
          z = 0
        return np.degrees(x), np.degrees(y), np.degrees(z)

      pitch, yaw, roll = rotationMatrixToEulerAngles(rotation_matrix)



      #leftDistances = Normalize(torch.tensor(leftDistances)).tolist()
      #rightDistances = Normalize(torch.tensor(rightDistances)).tolist()



      #return leftSoftmax + rightSoftmax
      return leftSoftmax + rightSoftmax + [eyeDistance]
      #return leftSoftmax + rightSoftmax + [eyeDistance, faceDistance, *forehead, *nose, pitch, yaw, roll]
      #return leftSoftmax + rightSoftmax + [eyeDistance, faceDistance, *forehead, *nose]
      #return leftDistances + rightDistances
      #return int(pitch), int(yaw), int(roll)
      #return leftDistances + rightDistances + [pitch, yaw, roll]
      #return leftSoftmax + rightSoftmax + [pitch, yaw, roll]
    else:
      return None

if __name__ == "__main__":
  cap = cv2.VideoCapture(0)
  preprocess = Preprocess()

  while True:
    status, frame = cap.read()
    if not status:
      exit()

    rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    features = preprocess.Handle(rgbFrame)

    print(features)