from pathlib import Path
import pygame
from Utils.CameraStream import CameraStream
from N3_Calibrate import CalibrationModel
from Common import *
from Utils.EMA import EMA
from Utils.SharedMemoryWriter import SharedMemoryWriter
from Utils.Timer import Timer

# Params
assetsDir = Path("../Assets/Model0")
imagesDir = assetsDir / "Images"
weightsDir = assetsDir / "Weights"
weightFile = weightsDir / "model.pt"

def run_main_loop():
  emaX = EMA(0.9, 0)
  emaY = EMA(0.9, 0)

  # API
  gazeWriter = SharedMemoryWriter()
  gazeWriter.Write(0, 0)

  # Model
  cap = CameraStream(0)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  calibrator = CalibrationModel()
  calibrator.load_state_dict(torch.load(weightFile, map_location=device))
  calibrator.to(device)
  calibrator.eval()

  preprocess = Preprocess()

  # Pygame
  info = pygame.display.Info()
  width = info.current_w
  height = info.current_h

  clock = pygame.time.Clock()
  running = True
  fps = 144
  clock.tick(fps)

  last_frame_number = -1
  x = 0
  y = 0
  time = 0.0

  # Stats
  stats = []
  min_val = float('inf')
  max_val = float('-inf')
  avg_val = 0.0
  count = 0

  while running:
    dt = clock.tick(fps)
    time += dt

    # Face detection
    status, frame, frame_number = cap.read()
    if not status:
      continue

    if frame_number - last_frame_number > 0:
      processTimer = Timer()

      # Collect features
      frameRgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      features = preprocess.Handle(frameRgb)

      processTime = processTimer.GetElapsedTimeMs()
      calibrateTimer = Timer()

      if features is not None:
        last_frame_number = frame_number

        # Extract tensor for specified indices
        featuresTensor = torch.tensor(features, dtype = torch.float32, device = device).unsqueeze(0)

        outputX, outputY = calibrator(featuresTensor)
        outputX, outputY = torch.softmax(outputX.squeeze(), 0).cpu(), torch.softmax(outputY.squeeze(), 0).cpu()


        def SoftArgmax(inputX: torch.Tensor, inputY: torch.Tensor):
          xId = torch.tensor([x for x in range(X_CLASSES)], dtype=torch.float32)
          yId = torch.tensor([y for y in range(Y_CLASSES)], dtype=torch.float32)
          return (inputX * xId).sum().item() / X_CLASSES, (inputY * yId).sum().item() / Y_CLASSES

        def DeclassifyArgmax(inputX: torch.Tensor, inputY: torch.Tensor):
          indexX, indexY = torch.argmax(outputX).item(), torch.argmax(outputY).item()
          x, y = Declassify(indexX, 0, 1, X_CLASSES), Declassify(indexY, 0, 1, Y_CLASSES)
          return x, y

        x, y = SoftArgmax(outputX, outputY)
        #x, y = DeclassifyArgmax(outputX, outputY)

        # Update coords
        #x, y = Declassify(x, 0, 1, X_CLASSES), Declassify(y, 0, 1, Y_CLASSES)
        x = x * width + (width / X_CLASSES) / 2
        y = y * height + (height / Y_CLASSES) / 2
        x /= 1920
        y /= 1080
        x = emaX.update(x)
        y = emaY.update(y)
        # x += (1 / X_CLASSES) / 2
        # y += (1 / Y_CLASSES) / 2
        gazeWriter.Write(x, y)

        # indexX, indexY = torch.argmax(outputX).item(), torch.argmax(outputY).item()
        # x, y = Declassify(indexX, 0, 1, X_CLASSES), Declassify(indexY, 0, 1, Y_CLASSES)
        # x = x * width + (width / X_CLASSES) / 2
        # y = y * height + (height / Y_CLASSES) / 2



        # calibrateTime = calibrateTimer.GetElapsedTimeMs()

        # if time > 2.0:
        #   append_sample_data(stats, mouse_x, mouse_y, x, y, processTime, calibrateTime)
        #   calculate_and_print_stats(stats)
        #
        #   if len(stats) > 1000:
        #     exit()



if __name__ == '__main__':
  pygame.init()

  # Loop
  run_main_loop()

  pygame.quit()
